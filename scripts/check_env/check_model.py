"""Verify Qwen2-VL-2B loads on GPU and one forward pass runs on a dummy 256x256 image.

Run: uv run python tools/check_env/check_model.py
Requires HF_TOKEN for the HF resolver (model is public but gated downloads need auth).
"""
from __future__ import annotations
import os
import sys


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def main() -> int:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set (try: set -a; source .env; set +a)")

    print(f"Loading {MODEL_ID} ...")
    proc = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        token=hf_token,
    ).to("cuda").eval()
    print(f"  loaded, dtype={next(model.parameters()).dtype}, device={next(model.parameters()).device}")

    # Dummy 256x256 black image + "describe" prompt
    from PIL import Image
    img = Image.new("RGB", (256, 256), color=(0, 0, 0))
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img}, {"type": "text", "text": "describe"}],
    }]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = model(**inputs)
    print(f"  forward ok: logits {tuple(out.logits.shape)}")

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
