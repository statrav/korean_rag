import os
import json

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

config = load_config("config.json")

DATA_DIR = config.get("data_dir", "data")
TRAIN_FILE = os.path.join(DATA_DIR, config.get("train_file", "korean_language_rag_V1.0_train.json"))
DEV_FILE = os.path.join(DATA_DIR, config.get("dev_file", "korean_language_rag_V1.0_dev.json"))
TEST_FILE = os.path.join(DATA_DIR, config.get("test_file", "korean_language_rag_V1.0_test.json"))

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_documents(data):
    docs = []
    for item in data:
        question = item["input"]["question"]
        answer = item.get("output", {}).get("answer", None)
        if answer:
            docs.append(question + "\n" + answer)
        else:
            docs.append(question)
    return docs 