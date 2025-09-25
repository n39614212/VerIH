import key
import re
import os
import json
import tqdm
import random
import traceback
import numpy as np
from llm import OpenAIInterface, formate_json
from datasets import load_dataset
random.seed(42)


def generate_folio(split):
    dataset = load_dataset("yale-nlp/FOLIO")[split]
    columns = ["premises", "conclusion", "label"]
    selected_data = dataset.select_columns(columns)
    samples = [dict(item) for item in selected_data]
    for i in samples:
        if i["label"] == "False":
            i["label"] = False
        elif i["label"] in ["True", "Uncertain"]:
            i["label"] = True
        else: assert False
        i["type"]="folio"
    return samples


def generate_lsat(split):
    dataset = load_dataset("tasksource/lsat-ar")[split]
    columns = ["context", "answers", "label", "question"]
    selected_data = dataset.select_columns(columns)
    samples = [dict(item) for item in selected_data]
    for i in samples:
        assert len(i["answers"]) >= i["label"]

    model = OpenAIInterface(model="gpt-4o")
    sys_request = '''
You are a helpful assistant. When given a question and its answer, convert them into a single, complete declarative sentence that preserves the original meaning.
Return Format: a json object that contain the sentence, example: {"sentence":"xxxx"}'''
    user_template = '''
The question: {}
The answer: {}'''
    dataset = []
    for i in tqdm.tqdm(samples):
        label = random.choice([True, False])
        if label:
            ans = i["answers"][i["label"]]
        else:
            ans = random.choice(i["answers"])

        try:
            response = model.generate(sys_request, user_template.format(i["question"], ans), formate_json)
            dataset.append({
                "type": "lsat",
                "premises": i["context"],
                "conclusion": response["sentence"],
                "label": label
            })
        except Exception as _:
            traceback.print_exc()
    return dataset


def save_dataset(dataset, base_dir, file_name):
    os.makedirs(base_dir, exist_ok=True)
    random.shuffle(dataset)
    with open(os.path.join(base_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    base_dir = "dataset/logic"
    dataset = generate_folio("train")
    save_dataset(dataset, base_dir, "folio_train.json")
    dataset = generate_folio("validation")
    save_dataset(dataset, base_dir, "folio_test.json")

    # dataset = generate_lsat("train")
    # save_dataset(dataset, base_dir, "lsat_train.json")
    # print(OpenAIInterface.total_cost)
    # dataset = generate_lsat("validation")
    # save_dataset(dataset, base_dir, "lsat_test.json")
    # print(OpenAIInterface.total_cost)
