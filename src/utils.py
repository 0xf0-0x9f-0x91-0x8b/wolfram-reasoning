import os
import time
import subprocess
from pathlib import Path

import csv
import matplotlib.pyplot as plt
import torch

# runtime
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[RUNTIME] {func.__name__}: {end - start:.4f} seconds\n")
        return result
    return wrapper

def gpu_memory():
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]
    output = subprocess.check_output(cmd).decode("utf-8").strip().split("\n")
    percentages = []
    for line in output:
        used, total = [int(x.strip()) for x in line.split(",")]
        pct = used / total * 100.0
        percentages.append(pct)
    print('[RUNTIME] GPU Usage:' + '|'.join([f'{pct:.1f}%' for pct in percentages]) + '\n')

def get_parent_dir(path: str) -> str:
    """
    Return the parent directory of the given file or directory path.
    """
    return str(Path(path).resolve().parent)

# images
def convert2img(tensor2d, out_path, cmap="viridis"):
    """
    tensor2d: shape (H, W) torch.Tensor or numpy array
    """
    if isinstance(tensor2d, torch.Tensor):
        tensor2d = tensor2d.cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(tensor2d, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# csv
def write_csv_header(csv_path):
    write_header = not os.path.exists(csv_path)
    writer = None
    if write_header:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "image", "answer", "text", "output_length", "prompt_length"])

def write_csv_rows(batch, texts, output_lengths, prompt_lengths, csv_path):
    output_lengths = output_lengths.detach().cpu().tolist()
    prompt_lengths = prompt_lengths.detach().cpu().tolist()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for ex, txt, out_len, prompt_len in zip(batch, texts, output_lengths, prompt_lengths):
            writer.writerow([
                ex["question"],
                ex["image"],
                ex["answer"],
                txt,
                out_len,
                prompt_len,
            ])