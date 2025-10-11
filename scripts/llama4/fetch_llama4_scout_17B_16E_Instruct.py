from huggingface_hub import snapshot_download
import os

# Prefer /dev/shm for large temporary model files and tensors (tmpfs)
HF_SHM_CACHE = "~/.cache/huggingface/"
os.makedirs(HF_SHM_CACHE, exist_ok=True)

# Redirect HF/Transformers/torch caches to tmpfs
os.environ.setdefault("HF_HOME", HF_SHM_CACHE)              # general HF home
os.environ.setdefault("TRANSFORMERS_CACHE", HF_SHM_CACHE)    # transformers cache
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_SHM_CACHE, "datasets"))
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(HF_SHM_CACHE, "modules"))
os.environ.setdefault("XDG_CACHE_HOME", HF_SHM_CACHE)        # many libs use this
os.environ.setdefault("TORCH_HOME", os.path.join(HF_SHM_CACHE, "torch"))

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

# download snapshot into the tmpfs cache dir
path = snapshot_download(
    repo_id=model_id,
    cache_dir=HF_SHM_CACHE,
    token=token,          # None -> will use logged in CLI token
    repo_type="model",
)

print("Model downloaded to:", path)
