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
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"\\boxed\{(.*?)}", text, re.DOTALL | re.IGNORECASE)
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


def same(a, b):
    if a is None or b is None:
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
    # 1. Extract predicted answer inside <answer>...</answer>
    # ----------------------------------------------------------
    df["predicted"] = df["text"].apply(extract_pred)

    # ----------------------------------------------------------
    # 2. Extract ground-truth answer from \boxed{...}
    # ----------------------------------------------------------
    df["ground_truth"] = df["answer"].apply(extract_gt)

    # ----------------------------------------------------------
    # 3. Evaluate correctness (case-insensitive exact match)
    # ----------------------------------------------------------
    df["correct"] = df.apply(lambda r: same(r["predicted"], r["ground_truth"]), axis=1)

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
            accuracy=("correct", lambda x: 100 * x.mean()),
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
    ).round(2)

    # ----------------------------------------------------------
    # 6. Compute overall statistics across ALL rows
    # ----------------------------------------------------------
    overall = pd.DataFrame({
        "category": ["Overall"],
        "num_queries": [len(df)],
        "accuracy": [100 * df["correct"].mean()],
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
            "accuracy",
            "query_length",
            "output_length",
            "num_images_per_query",
            "image_width",
            "image_height",
        ]
    ]
    return summary, df


def load_results(path):
    df = pd.read_csv(path, index_col=None)
    df_dataset = pd.read_csv("39Krelease.csv", index_col=None)
    df_dataset_unique = df_dataset.drop_duplicates(subset=["question"], keep="first")
    df = df.merge(
        df_dataset_unique[["question", "category"]],
        on="question",
        how="left"
    )
    return df


def main(csv_path):
    df = load_results(csv_path)
    summary, df = evaluate_results(df)

    to_csv_path = get_parent_dir(csv_path) + '/summary.csv'
    if not os.path.exists(to_csv_path):
        summary.iloc[:, :5].round(2).to_csv(
            to_csv_path, index=False
        )

    return summary, df

if __name__ == "__main__":
    summary_df, results_df = main("./baseline_results/results_answers_iter1.csv")
    breakpoint()