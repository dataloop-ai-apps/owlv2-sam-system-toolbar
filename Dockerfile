FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

ENV HF_HOME=/tmp/.cache/huggingface
ENV HF_HUB_CACHE=/tmp/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface/hub

RUN ${DL_PYTHON_EXECUTABLE} -m pip install -U --no-cache-dir \
    "numpy>=1.22,<2.0"

RUN ${DL_PYTHON_EXECUTABLE} -m pip install -U --no-cache-dir \
    pyyaml>=5.3.1 \
    onnxruntime \
    torch>=2.0.0 \
    transformers>=4.37.0 \
    huggingface-hub

# Download OWLv2 weights
RUN ${DL_PYTHON_EXECUTABLE} - <<'PY'
import os
from transformers import Owlv2Processor, Owlv2ForObjectDetection

model_id = "google/owlv2-base-patch16-ensemble"
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))
print("Pre-downloading:", model_id)

Owlv2Processor.from_pretrained(model_id)
Owlv2ForObjectDetection.from_pretrained(model_id)

print("Done downloading OWLv2 weights.")
PY

# Download SAM2 ONNX decoder weights
RUN mkdir -p /tmp/app/weights && \
    ${DL_PYTHON_EXECUTABLE} -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='SharpAI/sam2-hiera-small-onnx', \
                    filename='decoder.onnx', \
                    local_dir='/tmp/app/weights', \
                    local_dir_use_symlinks=False)" && \
    mv /tmp/app/weights/decoder.onnx /tmp/app/weights/sam2_hiera_small.decoder.onnx

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/owlv2-sam-system-toolbar:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/owlv2-sam-system-toolbar:0.0.1