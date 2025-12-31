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
from threading import Thread
from types import SimpleNamespace

from transformers import Owlv2Processor

logger = logging.getLogger("OWLV2-SAM-TOOLBAR")

Image.MAX_IMAGE_PIXELS = 933120000


class Runner(dl.BaseServiceRunner):

    def __init__(self, dl):
        self.device = "cpu"

        processor_path = "weights/owlv2-base-patch16-ensemble-ONNX"
        if os.path.isdir("/tmp/app/weights/owlv2-base-patch16-ensemble-ONNX"):
            processor_path = "/tmp/app/weights/owlv2-base-patch16-ensemble-ONNX"
        self.owl_processor = Owlv2Processor.from_pretrained(processor_path)

        owl_onnx_path = "weights/owlv2-base-patch16-ensemble-ONNX/onnx/model.onnx"
        if os.path.isfile("/tmp/app/weights/owlv2-base-patch16-ensemble-ONNX/onnx/model.onnx"):
            owl_onnx_path = "/tmp/app/weights/owlv2-base-patch16-ensemble-ONNX/onnx/model.onnx"

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.owl_session = onnxruntime.InferenceSession(
            owl_onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.owl_input_names = [inp.name for inp in self.owl_session.get_inputs()]
        self.owl_output_names = [out.name for out in self.owl_session.get_outputs()]

        p = dl.projects.get("DataloopTasks")
        self.sam_service = p.services.get(service_name="global-sam")

        if os.path.isfile("/tmp/app/weights/sam2_hiera_small.decoder.onnx"):
            onnx_model_path = "/tmp/app/weights/sam2_hiera_small.decoder.onnx"
        else:
            onnx_model_path = "weights/sam2_hiera_small.decoder.onnx"

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
        if "bot.dataloop.ai" in dl.info()["user_email"]:
            raise ValueError("This function cannot run with a bot user")

        tic = time.time()
        with tempfile.TemporaryDirectory() as tempf:
            source = item.download(local_path=tempf, overwrite=True)

            labels = list(item.dataset.labels_flat_dict.keys())
            if len(labels) == 0:
                raise ValueError(
                    "Dataset recipe labels are empty. OWLv2 requires a label/prompt list "
                )

            parallel_start = time.time()
            detections = None
            sam_features = None
            exception = None
            owlv2_time = None
            sam_features_time = None

            def run_owlv2_thread():
                nonlocal detections, exception, owlv2_time
                try:
                    owl_start = time.time()
                    detections = self.run_owlv2(source=source, labels=labels)
                    owlv2_time = time.time() - owl_start
                    logger.info(f"[TIMING] OWLv2 detection took: {owlv2_time:.2f}s")
                except Exception as e:
                    exception = e

            def run_sam_features_thread():
                nonlocal sam_features, exception, sam_features_time
                try:
                    sam_start = time.time()
                    sam_features = self.extract_sam_features(dl=dl, item=item)
                    sam_features_time = time.time() - sam_start
                    logger.info(f"[TIMING] SAM feature extraction took: {sam_features_time:.2f}s")
                except Exception as e:
                    exception = e

            thread1 = Thread(target=run_owlv2_thread)
            thread2 = Thread(target=run_sam_features_thread)

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()

            if exception:
                raise exception

            parallel_time = time.time() - parallel_start
            logger.info(f"[TIMING] Parallel execution (OWLv2 + SAM features) took: {parallel_time:.2f}s")

            decoder_start = time.time()
            collection = self.run_sam_decoder(
                item=item, detections=detections, labels=labels, sam_features=sam_features
            )
            decoder_time = time.time() - decoder_start
            logger.info(f"[TIMING] SAM decoder took: {decoder_time:.2f}s")

        total_time = time.time() - tic
        logger.info(f"[TIMING] Full run took: {total_time:.2f}s")
        logger.info(f"[TIMING] Breakdown - OWLv2: {owlv2_time:.2f}s | SAM features: {sam_features_time:.2f}s | SAM decoder: {decoder_time:.2f}s")
        return collection.to_json()["annotations"]

    def extract_sam_features(self, dl, item: dl.Item):
        ex = dl.services.execute(
            service_id=self.sam_service.id,
            function_name="get_sam_features",
            item_id=item.id,
            project_id=item.project.id,
        )
        ex = dl.executions.wait(execution=ex, timeout=60)
        if ex.latest_status["status"] not in ["success"]:
            raise ValueError(f"Execution failed. ex id: {ex.id}")

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

        return {
            "image_embed": image_embed,
            "high_res_feats_0": high_res_feats_0,
            "high_res_feats_1": high_res_feats_1,
        }

    def run_sam_decoder(self, item: dl.Item, detections, labels, sam_features):
        height = item.height
        width = item.width
        embed_size = 64

        collection = dl.AnnotationCollection()

        for det in reversed(detections):
            c = int(det["cls"])
            d_conf = float(det["conf"])
            obj_id = det["id"]

            if c < 0 or c >= len(labels):
                continue

            name = labels[c]
            box = det["xyxy"]

            feeds = {
                "image_embed": sam_features["image_embed"],
                "high_res_feats_0": sam_features["high_res_feats_0"],
                "high_res_feats_1": sam_features["high_res_feats_1"],
                "point_coords": np.array(
                    [[[box[0] / width * 1024, box[1] / height * 1024],
                      [box[2] / width * 1024, box[3] / height * 1024]]],
                    dtype=np.float32,
                ),
                "point_labels": np.array([[2, 3]], dtype=np.float32),
                "mask_input": torch.randn(1, 1, 4 * embed_size, 4 * embed_size, dtype=torch.float).cpu().numpy(),
                "has_mask_input": np.array([0], dtype=np.float32),
            }

            result = self.sam_ort_session.run([self.output_name[0]], feeds)
            mask = cv2.resize(result[0][0][0], (width, height))

            collection.add(
                annotation_definition=dl.Box(
                    label=name, left=box[0], top=box[1], right=box[2], bottom=box[3]
                ),
                object_id=obj_id,
                model_info={"name": "owlv2", "confidence": d_conf},
            )

            collection.add(
                annotation_definition=dl.Segmentation(geo=mask > 0, label=name),
                object_id=obj_id,
                model_info={"name": "owlv2", "confidence": d_conf},
            )

        return collection

    def run_sam(self, dl, item: dl.Item, detections, labels):
        sam_features = self.extract_sam_features(dl=dl, item=item)
        return self.run_sam_decoder(item=item, detections=detections, labels=labels, sam_features=sam_features)

    def run_owlv2(self, source: str, labels: list):
        image = Image.open(source).convert("RGB")
        w, h = image.size

        inputs = self.owl_processor(images=image, text=[labels], return_tensors="pt")

        onnx_inputs = {
            "pixel_values": inputs["pixel_values"].numpy(),
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }

        onnx_outputs = self.owl_session.run(self.owl_output_names, onnx_inputs)

        outputs = SimpleNamespace(
            logits=torch.from_numpy(onnx_outputs[0]),
            pred_boxes=torch.from_numpy(onnx_outputs[1]),
        )

        processed = self.owl_processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.owl_score_threshold,
            target_sizes=torch.tensor([[h, w]], device=self.device),
        )[0]

        boxes = processed["boxes"].detach().cpu().numpy()
        scores = processed["scores"].detach().cpu().numpy()
        label_ids = processed["labels"].detach().cpu().numpy()

        if len(scores) > self.max_detections:
            idx = np.argsort(scores)[-self.max_detections:][::-1]
            boxes, scores, label_ids = boxes[idx], scores[idx], label_ids[idx]

        detections = []
        for i in range(len(scores)):
            x0, y0, x1, y1 = boxes[i].tolist()

            x0 = float(max(0, min(x0, w - 1)))
            y0 = float(max(0, min(y0, h - 1)))
            x1 = float(max(0, min(x1, w - 1)))
            y1 = float(max(0, min(y1, h - 1)))

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

        return detections


def test():
    import dtlpy as dl
    logging.basicConfig(level=logging.INFO)
    dl.setenv("prod")
    item = dl.items.get(item_id="")

    runner = Runner(dl=dl)
    annotations = runner.run(dl=dl, item=item)
    item.annotations.upload(annotations)


if __name__ == "__main__":
    test()
