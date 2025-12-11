import time
start_time = time.perf_counter()
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from utils import time_it
from prompts import build_prompt
from model import load_model, predict

end_time = time.perf_counter()
print(f"[RUNTIME] Imports: {end_time - start_time:.4f} seconds\n")

@time_it
def register_attn_hooks(model):
    attention_maps = []

    def save_attn_hook(module, input, output):
        # Case 1: output is a BaseModelOutputWithPastAndCrossAttentions
        if hasattr(output, "attentions") and output.attentions is not None:
            for attn in output.attentions:
                attention_maps.append(attn.detach().cpu())

        # Case 2: output is a tuple (HF often returns tuples)
        elif isinstance(output, tuple):
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    attention_maps.append(item.detach().cpu())

    # Register hooks on relevant layers
    for name, module in model.named_modules():
        if "attn" in name.lower():
            module.register_forward_hook(save_attn_hook)

    return attention_maps


def visualize_attn(attention_maps, image):
    attn = attention_maps[-1][0]   # shape: [num_heads, T_text, T_image]

    # Average heads and pick the last generated token
    attn_map = attn.mean(0)[-1]  # shape: [T_image]

    # ------------------------------------------------------
    # 5. Convert image tokens into a spatial map
    # ------------------------------------------------------
    # Qwen-VL uses 16x16 patches → token count ~ (H/16 * W/16)
    patch = 16
    W, H = image.size
    patch_h = H // patch
    patch_w = W // patch
    if patch_h * patch_w != attn_map.shape[0]:
        raise ValueError(
            f"Token count mismatch: expected {patch_h*patch_w}, got {attn_map.shape[0]}"
        )

    attn_grid = attn_map.reshape(patch_h, patch_w)

    # Normalize for visualization
    attn_grid = attn_grid / attn_grid.max()

    # Upsample to original image size
    attn_resized = F.interpolate(
        torch.tensor(attn_grid).unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )[0,0].numpy()

    # ------------------------------------------------------
    # 6. Overlay heatmap on the image
    # ------------------------------------------------------
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.imshow(attn_resized, cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.show()
    plt.savefig("attention_overlay.png", bbox_inches='tight', pad_inches=0)


def main():
    accelerator, model, tokenizer = load_model()
    model.eval()

    attention_maps = register_attn_hooks(model)

    images = ['images/Processed-5cb29a4f-240d-4f1f-a063-b7fd922ee9e9-0.jpg']
    image = Image.open(images[0]).convert("RGB")
    question = "<image>\nWhat fraction of the shapes are squares?\nChoices:\n(A) 5/10\n(B) 3/7\n(C) 3/9\n(D) 5/9"
    inputs = build_prompt(tokenizer, [(question, images)], accelerator)
    out = predict(accelerator, model, tokenizer, inputs)
    
    visualize_attn(attention_maps, image)


if __name__ == "__main__":
    main()