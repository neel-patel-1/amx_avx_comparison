from huggingface_hub import snapshot_download
import os

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

# download snapshot into the tmpfs cache dir
path = snapshot_download(
    repo_id=model_id,
    token=token,          # None -> will use logged in CLI token
    repo_type="model",
)
