import numpy as np
import dtlpy as dl
import onnxruntime
import tempfile
import logging
import base64
import torch
import json
import time
import cv2
import os
from PIL import Image

# OWLv2 (Hugging Face)
from transformers import Owlv2Processor, Owlv2ForObjectDetection

logger = logging.getLogger("OWLV2-SAM-TOOLBAR")

# set max image size
Image.MAX_IMAGE_PIXELS = 933120000


class Runner(dl.BaseServiceRunner):
    """
    OWLv2 -> Boxes -> SAM2 Decoder (ONNX) -> Upload Boxes + Segmentations
    """

    def __init__(self, dl):
        self.device = "cpu"

        self.owl_model_id = "google/owlv2-base-patch16-ensemble"
        self.owl_processor = Owlv2Processor.from_pretrained(self.owl_model_id)
        self.owl_model = Owlv2ForObjectDetection.from_pretrained(self.owl_model_id).to(self.device)
        self.owl_model.eval()

        p = dl.projects.get("DataloopTasks")
        self.sam_service = p.services.get(service_name="global-sam")

        if os.path.isfile("/tmp/app/weights/sam2_hiera_small.decoder.onnx"):
            onnx_model_path = "/tmp/app/weights/sam2_hiera_small.decoder.onnx"
        else:
            onnx_model_path = "weights/sam2_hiera_small.decoder.onnx"
        
        # TODO: check why we are using a local path for the ONNX model
        self.sam_ort_session = onnxruntime.InferenceSession(
            onnx_model_path, providers=["CPUExecutionProvider"]
        )
        self.output_name = [a.name for a in self.sam_ort_session.get_outputs()]

        self.owl_score_threshold = 0.20
        self.max_detections = 300

    def run_and_upload(self, dl, item: dl.Item):
        annotations = self.run(dl, item)
        item.annotations.upload(annotations)

    def run(self, dl, item: dl.Item):
        """
        Run OWLv2 and the SAM decoder on the item
        """
        # get item's image
        if "bot.dataloop.ai" in dl.info()["user_email"]:
            raise ValueError("This function cannot run with a bot user")

        tic = time.time()
        with tempfile.TemporaryDirectory() as tempf:
            source = item.download(local_path=tempf, overwrite=True)

            # get labels from dataset
            labels = list(item.dataset.labels_flat_dict.keys())
            if len(labels) == 0:
                raise ValueError(
                    "Dataset recipe labels are empty. OWLv2 requires a label/prompt list "
                )

            detections = self.run_owlv2(source=source, labels=labels)
            collection = self.run_sam(
                dl=dl, item=item, detections=detections, labels=labels
            )

        logger.info(f"Full run took: {time.time() - tic:.2f} seconds")
        return collection.to_json()["annotations"]

    def run_sam(self, dl, item: dl.Item, detections, labels):
        """
        Run the SAM decoder on the item, using OWLv2 detections as prompts.
        `detections` is a list of dicts:
            {"cls": int, "conf": float, "id": int, "xyxy": np.ndarray shape (4,)}
        """
        tic = time.time()
        logger.info(f"Running SAM decoder on item: {item.id}")
        logger.info(f"current user: {dl.info()['user_email']}")

        ex = dl.services.execute(
            service_id=self.sam_service.id,
            function_name="get_sam_features",
            item_id=item.id,
            project_id=item.project.id,
        )
        ex = dl.executions.wait(execution=ex, timeout=60)
        if ex.latest_status["status"] not in ["success"]:
            raise ValueError(f"Execution failed. ex id: {ex.id}")

        logger.info(f"SAM feature extraction took: {time.time() - tic:.2f} seconds")

        item_bytes = dl.items.get(item_id=ex.output).download(save_locally=False)
        image_embedding_dict = json.load(item_bytes)

        image_embed = np.frombuffer(
            base64.b64decode(image_embedding_dict["image_embed"]), dtype=np.float32
        ).reshape([1, 256, 64, 64])
        high_res_feats_0 = np.frombuffer(
            base64.b64decode(image_embedding_dict["high_res_feats_0"]), dtype=np.float32
        ).reshape([1, 32, 256, 256])
        high_res_feats_1 = np.frombuffer(
            base64.b64decode(image_embedding_dict["high_res_feats_1"]), dtype=np.float32
        ).reshape([1, 64, 128, 128])

        embed_size = 64
        height = item.height
        width = item.width

        collection = dl.AnnotationCollection()

        for det in reversed(detections):
            c = int(det["cls"])
            d_conf = float(det["conf"])
            obj_id = det["id"]

            if c < 0 or c >= len(labels):
                continue

            name = labels[c]
            box = det["xyxy"]  # [x0,y0,x1,y1] in pixel coords

            logger.info(f"{name} {d_conf:.2f} {box}")

            feeds = {
                "image_embed": image_embed,
                "high_res_feats_0": high_res_feats_0,
                "high_res_feats_1": high_res_feats_1,
                "point_coords": np.array(
                    [
                        [
                            [box[0] / width * 1024, box[1] / height * 1024],
                            [box[2] / width * 1024, box[3] / height * 1024],
                        ]
                    ],
                    dtype=np.float32,
                ),
                "point_labels": np.array([[2, 3]], dtype=np.float32),

                "mask_input": torch.randn(
                    1, 1, 4 * embed_size, 4 * embed_size, dtype=torch.float
                ).cpu().numpy(),
                "has_mask_input": np.array([0], dtype=np.float32),
            }

            result = self.sam_ort_session.run([self.output_name[0]], feeds)
            mask = cv2.resize(result[0][0][0], (width, height))

            # Box annotation
            collection.add(
                annotation_definition=dl.Box(
                    label=name, left=box[0], top=box[1], right=box[2], bottom=box[3]
                ),
                object_id=obj_id,
                model_info={"name": "owlv2", "confidence": d_conf},
            )

            # Segmentation annotation
            collection.add(
                annotation_definition=dl.Segmentation(geo=mask > 0, label=name),
                object_id=obj_id,
                model_info={"name": "owlv2", "confidence": d_conf},
            )

        logger.info(f"Total SAM predictions took: {time.time() - tic:.2f} seconds")
        return collection

    def run_owlv2(self, source: str, labels: list):
        """
        Run OWLv2 on the image with provided text labels.
        Returns a list of detection dicts compatible with `run_sam`.
        """
        tic = time.time()
        image = Image.open(source).convert("RGB")
        w, h = image.size

        inputs = self.owl_processor(images=image, text=[labels], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.owl_model(**inputs)

        processed = self.owl_processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.owl_score_threshold,
            target_sizes=torch.tensor([[h, w]], device=self.device),
        )[0]

        boxes = processed["boxes"].detach().cpu().numpy()       # (N,4) xyxy pixels
        scores = processed["scores"].detach().cpu().numpy()     # (N,)
        label_ids = processed["labels"].detach().cpu().numpy()  # (N,) indices into `labels`

        # Keep only top-N detections by score (optional safety limit)
        if len(scores) > self.max_detections:
            idx = np.argsort(scores)[-self.max_detections:][::-1]
            boxes, scores, label_ids = boxes[idx], scores[idx], label_ids[idx]

        detections = []
        for i in range(len(scores)):
            x0, y0, x1, y1 = boxes[i].tolist()

            # Clamp to image bounds
            x0 = float(max(0, min(x0, w - 1)))
            y0 = float(max(0, min(y0, h - 1)))
            x1 = float(max(0, min(x1, w - 1)))
            y1 = float(max(0, min(y1, h - 1)))

            # Skip degenerate boxes
            if x1 <= x0 or y1 <= y0:
                continue

            detections.append(
                {
                    "cls": int(label_ids[i]),
                    "conf": float(scores[i]),
                    "id": i,
                    "xyxy": np.array([x0, y0, x1, y1], dtype=np.float32),
                }
            )

        logger.info(f"OWLv2 took: {time.time() - tic:.2f} seconds, detections={len(detections)}")
        return detections


def test():
    import dtlpy as dl

    dl.setenv("rc")
    # item = dl.items.get(item_id="")  # prod
    item = dl.items.get(item_id="")  # rc

    runner = Runner(dl=dl)
    annotations = runner.run(dl=dl, item=item)
    item.annotations.upload(annotations)


if __name__ == "__main__":
    test()
