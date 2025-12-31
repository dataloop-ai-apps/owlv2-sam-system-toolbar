FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

RUN ${DL_PYTHON_EXECUTABLE} -m pip install -U --no-cache-dir \
    "numpy>=1.22,<2.0" \
    pyyaml>=5.3.1 \
    onnxruntime \
    torch>=2.0.0 \
    transformers>=4.37.0 \
    huggingface-hub

RUN ${DL_PYTHON_EXECUTABLE} - <<'PY'
import os
from huggingface_hub import snapshot_download

weights_dir = "/tmp/app/weights"
os.makedirs(weights_dir, exist_ok=True)

# Download ONNX model + processor files from single repo
snapshot_download(
    repo_id="onnx-community/owlv2-base-patch16-ensemble-ONNX",
    local_dir=os.path.join(weights_dir, "owlv2-base-patch16-ensemble-ONNX"),
    local_dir_use_symlinks=False,
    allow_patterns=["onnx/model.onnx", "*.json", "*.txt"],
)
PY

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/owlv2-sam-system-toolbar:0.0.3 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/owlv2-sam-system-toolbar:0.0.3
