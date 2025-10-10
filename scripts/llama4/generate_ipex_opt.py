from transformers import AutoProcessor, Llama4ForConditionalGeneration
from ipex_llm import optimize_model
import torch
import os
import time
import datetime

# Prefer /dev/shm for large temporary model files and tensors (tmpfs)
HF_SHM_CACHE = "/dev/shm/hf_cache"
os.makedirs(HF_SHM_CACHE, exist_ok=True)

# Redirect HF/Transformers/torch caches to tmpfs
os.environ.setdefault("HF_HOME", HF_SHM_CACHE)              # general HF home
os.environ.setdefault("TRANSFORMERS_CACHE", HF_SHM_CACHE)    # transformers cache
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_SHM_CACHE, "datasets"))
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(HF_SHM_CACHE, "modules"))
os.environ.setdefault("XDG_CACHE_HOME", HF_SHM_CACHE)        # many libs use this
os.environ.setdefault("TORCH_HOME", os.path.join(HF_SHM_CACHE, "torch"))

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_SHM_CACHE)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    cache_dir=HF_SHM_CACHE,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.float,
)

model = optimize_model(model, low_bit='sym_int8')

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {"type": "image", "url": url2},
            {"type": "text", "text": "Can you describe how these two images are similar, and how they differ?"},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# measure inference time
start_ts = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
start_perf = time.perf_counter()
print(f"Inference started at: {start_ts}")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)

end_perf = time.perf_counter()
end_ts = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
elapsed = end_perf - start_perf
print(f"Inference finished at: {end_ts}")
print(f"Elapsed time: {elapsed:.3f} seconds")

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)
print(outputs[0])