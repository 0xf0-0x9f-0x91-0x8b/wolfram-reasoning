import os
import pandas as pd

from dataclasses import dataclass
from datasets import Dataset, load_from_disk

from utils import time_it

@dataclass
class QAExample:
    question: str
    answer: str
    image: list[str]

@time_it
def build_dataset(seed=42):
    train_path = "./train_ds"
    test_path = "./test_ds"
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_ds = load_from_disk(train_path)
        test_ds = load_from_disk(test_path)
        print(f'Train test split: {len(train_ds)}, {len(test_ds)}')
        return train_ds, test_ds

    df = pd.read_parquet("./39Krelease.parquet")
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "question": row["question"],
                "answer": row.get("answer", ""),
                "image": row["image"],
            }
        )

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.01, seed=seed)

    train_ds = split["train"]
    test_ds = split["test"]
    print(f'Train test split: {len(train_ds)}, {len(test_ds)}')
    train_ds.save_to_disk(train_path)
    test_ds.save_to_disk(test_path)

    return train_ds, test_ds


def fmt(m, s):
    return f"{m:.2f} ± {s:.2f}"


def analyze_dataset():
    import pandas as pd
    import ast

    df = pd.read_parquet("./39Krelease.parquet")
    df_unique = df.drop_duplicates(subset=["question"], keep="first")

    df_test = pd.read_csv('./baseline_results/results_answers_iter1.csv')
    df_test = df_test.merge(
        df_unique[["question", "category"]],
        on="question",
        how="left"
    )
    df_test["image"] = df_test["image"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_test["num_images"]  = df_test["image"].apply(len)

    df_train = df_unique[~df_unique["question"].isin(df_test["question"])]
    df_train["num_images"] = df_train["image"].apply(len)

    train_counts = df_train["category"].value_counts().rename("train")
    test_counts  = df_test["category"].value_counts().rename("test")

    img_stats = (
        df_train.groupby("category")["num_images"].agg(['mean', 'std'])
            .rename(columns={'mean': 'train_img_mean', 'std': 'train_img_std'})
    )
    img_stats_test = (
        df_test.groupby("category")["num_images"].agg(['mean', 'std'])
            .rename(columns={'mean': 'test_img_mean', 'std': 'test_img_std'})
    )

    category_table = (
        pd.concat([train_counts, test_counts, img_stats, img_stats_test], axis=1)
        .fillna(0)
    )

    overall = pd.DataFrame({
        "train": [df_train.shape[0]],
        "test":  [df_test.shape[0]],
        "train_img_mean": [df_train["num_images"].mean()],
        "train_img_std":  [df_train["num_images"].std()],
        "test_img_mean":  [df_test["num_images"].mean()],
        "test_img_std":   [df_test["num_images"].std()],
    }, index=["Overall"])

    category_table = pd.concat([category_table, overall])
    category_table["train_img"] = category_table.apply(
        lambda r: fmt(r["train_img_mean"], r["train_img_std"]), axis=1
    )
    category_table["test_img"] = category_table.apply(
        lambda r: fmt(r["test_img_mean"], r["test_img_std"]), axis=1
    )

    final = category_table[[
        "train", "test", "train_img", "test_img"
    ]].astype({"train": int, "test": int})
    final.to_csv("./baseline_results/dataset.csv")


def main():
    analyze_dataset()

if __name__ == "__main__":
    main()