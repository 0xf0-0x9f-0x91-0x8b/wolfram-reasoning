import os
import ast
import re
import pandas as pd
from PIL import Image
import numpy as np

from utils import get_parent_dir

def extract_pred(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"<wolfram>(.*?)</wolfram>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def extract_gt(answer):
    if not isinstance(answer, str):
        return None
    m = re.search(r"\\boxed\{(.*?)}", answer)
    if m:
        return m.group(1).strip()
    return None

def has_errorfree_code(code_output):
    if not isinstance(code_output, str) or len(code_output.strip()) == 0:
        return False
    error_indicators = [
        "invalid",
        "not",
        "missing"
    ]
    code_output = code_output.strip().lower()
    for indicator in error_indicators:
        if indicator in code_output:
            return False
    return True

def same(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return a.strip().lower() == b.strip().lower()

def count_imgs(img_entry):
    if isinstance(img_entry, (list, tuple)):
        return len(img_entry)
    return 1 if isinstance(img_entry, str) else 0

def get_img_sizes(img_entry):
    """Return mean width and height for all images in this row."""
    if isinstance(img_entry, str):
        paths = [img_entry]
    elif isinstance(img_entry, (list, tuple)):
        paths = img_entry
    else:
        return np.nan, np.nan

    widths, heights = [], []
    for p in paths:
        with Image.open(p).convert("RGB") as im:
            widths.append(im.width)
            heights.append(im.height)

    return np.nanmean(widths), np.nanmean(heights)

def fmt(m, s):
    if pd.isna(m) or pd.isna(s):
        return "N/A"
    return f"{m:.2f} ± {s:.2f}"

def evaluate_results(df):
    # ----------------------------------------------------------
    # 1. Extract code <wolfram>...</wolfram>
    # ----------------------------------------------------------
    df["predicted"] = df["text"].apply(extract_pred)

    # ----------------------------------------------------------
    # 2. Extract ground-truth answer from \boxed{...}
    # ----------------------------------------------------------
    df["ground_truth"] = df["answer"].apply(extract_gt)

    # ----------------------------------------------------------
    # 3. Evaluate correctness (case-insensitive exact match)
    # ----------------------------------------------------------
    df["has_code"] = df["predicted"].apply(lambda x: x is not None)
    df["has_errorfree_code"] = df["wolfram_output"].apply(has_errorfree_code)
    df["is_correct"] = df.apply(lambda r: same(r["wolfram_output"], r["ground_truth"]), axis=1)

    # ----------------------------------------------------------
    # 4. Compute auxiliary columns
    # ----------------------------------------------------------
    df["image"] = df["image"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["num_images_per_query"] = df["image"].apply(count_imgs)
    df["image_width"], df["image_height"] = zip(*df["image"].apply(get_img_sizes))

    # ----------------------------------------------------------
    # 5. Compute category-wise statistics
    # ----------------------------------------------------------
    summary = (
        df.groupby("category")
        .agg(
            num_queries=("category", "size"),
            has_code=("has_code", lambda x: 100 * x.mean()),
            has_errorfree_code=("has_errorfree_code", lambda x: 100 * x.mean()),
            accuracy=("is_correct", lambda x: 100 * x.mean()),
            qlen_mean=("prompt_length", "mean"),
            qlen_std=("prompt_length", "std"),
            outlen_mean=("output_length", "mean"),
            outlen_std=("output_length", "std"),
            numimg_mean=("num_images_per_query", "mean"),
            numimg_std=("num_images_per_query", "std"),
            width_mean=("image_width", "mean"),
            width_std=("image_width", "std"),
            height_mean=("image_height", "mean"),
            height_std=("image_height", "std"),
        )
        .reset_index()
    )

    # ----------------------------------------------------------
    # 6. Compute overall statistics across ALL rows
    # ----------------------------------------------------------
    overall = pd.DataFrame({
        "category": ["Overall"],
        "num_queries": [len(df)],
        "has_code": [100 * df["has_code"].mean()],
        "has_errorfree_code": [100 * df["has_errorfree_code"].mean()],
        "accuracy": [100 * df["is_correct"].mean()],
        "qlen_mean": [df["prompt_length"].mean()],
        "qlen_std": [df["prompt_length"].std()],
        "outlen_mean": [df["output_length"].mean()],
        "outlen_std": [df["output_length"].std()],
        "numimg_mean": [df["num_images_per_query"].mean()],
        "numimg_std": [df["num_images_per_query"].std()],
        "width_mean": [df["image_width"].mean()],
        "width_std": [df["image_width"].std()],
        "height_mean": [df["image_height"].mean()],
        "height_std": [df["image_height"].std()],
    })

    # Append the overall row to the summary
    summary = pd.concat([summary, overall], ignore_index=True)

    # ----------------------------------------------------------
    # 7. Format mean/std columns into "mean ± std"
    # ----------------------------------------------------------
    summary["query_length"] = summary.apply(
        lambda r: fmt(r["qlen_mean"], r["qlen_std"]), axis=1
    )
    summary["output_length"] = summary.apply(
        lambda r: fmt(r["outlen_mean"], r["outlen_std"]), axis=1
    )
    summary["num_images_per_query"] = summary.apply(
        lambda r: fmt(r["numimg_mean"], r["numimg_std"]), axis=1
    )
    summary["image_width"] = summary.apply(
        lambda r: fmt(r["width_mean"], r["width_std"]), axis=1
    )
    summary["image_height"] = summary.apply(
        lambda r: fmt(r["height_mean"], r["height_std"]), axis=1
    )

    # Drop unformatted raw stats
    summary = summary[
        [
            "category",
            "num_queries",
            "has_code",
            "has_errorfree_code",
            "accuracy",
            "query_length",
            "output_length",
            "num_images_per_query",
            "image_width",
            "image_height",
        ]
    ].round(2)

    return summary, df

def load_results(results_path, eval_path):
    df = pd.read_csv(results_path, index_col=None)
    df_dataset = pd.read_csv("39Krelease.csv", index_col=None)
    df_dataset_unique = df_dataset.drop_duplicates(subset=["question"], keep="first")
    df = df.merge(
        df_dataset_unique[["question", "category"]],
        on="question",
        how="left"
    )
    if eval_path is not None:
        df_eval = pd.read_csv(eval_path, index_col=0)
        df = pd.concat([df, df_eval[["wolfram_output", "wolfram_error", "wolfram_returncode"]]], axis=1)
    return df

def main(results_path, eval_path):
    df = load_results(results_path, eval_path)
    summary, df = evaluate_results(df)
    summary_path = get_parent_dir(results_path) + '/summary.csv'
    if not os.path.exists(summary_path):
        summary.iloc[:, :7].round(2).to_csv(
            summary_path, index=False
        )

    return summary, df

if __name__ == "__main__":
    summary_df, results_df = main("./incontext_lrn_results/results_iter2.csv", "./incontext_lrn_results/eval_done.csv")
    breakpoint()