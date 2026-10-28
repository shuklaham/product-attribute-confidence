# Pod setup, run once on a fresh Runpod pod (runpod/pytorch image, CUDA 13 host) with the repo at /root/pac.
# Installs uv, the locked GPU dependencies and the pinned model revision, then checks that torch sees the GPU.
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=/root/venv HF_HOME=/root/hf UV_LINK_MODE=copy
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
cd /root/pac
uv sync --group gpu --locked
uv run --group gpu hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 >/dev/null
uv run --group gpu python -c "import torch, vllm, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok', torch.cuda.is_available(), '| vllm', vllm.__version__, '| transformers', transformers.__version__)"
echo SETUP_DONE
