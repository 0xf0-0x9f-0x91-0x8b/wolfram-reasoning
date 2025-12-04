import pandas as pd

from dataclasses import dataclass
from datasets import Dataset


@dataclass
class QAExample:
    question: str
    answer: str
    image: list[str]


def build_dataset(seed=42):
    df = pd.read_parquet("./39Krelease.parquet")
    df_geo = df[df["category"] == "(GradeSchool) Geometric"]
    df_circle = df_geo[df_geo["question"].str.contains("circle", case=False, na=False)]
    df_nottri = df_circle[~df_circle["question"].str.contains("triangle", case=False, na=False)]

    records = []
    for _, row in df_nottri.iterrows():
        question = row["question"]
        image = row["image"]
        answer = row.get("answer", "")
        records.append(
            {
                "question": question,
                "answer": answer,
                "image": image
            }
        )

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    return split['train'], split['test']