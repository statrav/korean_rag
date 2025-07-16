from langchain_community.embeddings import HuggingFaceEmbeddings

from utils.load_data import load_jsonl, build_documents, load_config
from models.retrieve import RetrieverManager
from models.generate import Generator
from models.generate_api import GeneratorAPI

import json
import os
import datetime

config = load_config("config.json")
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])

DATA_DIR = config.get("data_dir", "data")
TRAIN_FILE = f"{DATA_DIR}/" + config.get("train_file", "korean_language_rag_V1.0_train.json")
DEV_FILE = f"{DATA_DIR}/" + config.get("dev_file", "korean_language_rag_V1.0_dev.json")
TEST_FILE = f"{DATA_DIR}/" + config.get("test_file", "korean_language_rag_V1.0_test.json")

# split 선택 (dev, test, train)
split = config.get("split", "dev")
if split == "dev":
    EVAL_FILE = DEV_FILE
elif split == "test":
    EVAL_FILE = TEST_FILE
else:
    EVAL_FILE = TRAIN_FILE

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_output_path(split, mode):
    # mode: eval or submit
    timestamp = get_timestamp()
    filename = f"{split}_{mode}_results_{timestamp}.json"
    return os.path.join("outputs", filename)

if __name__ == "__main__":

    train_data = load_jsonl(TRAIN_FILE)
    documents = build_documents(train_data)
    print("# 학습 데이터 로드 및 문서화 성공")

    embedding_model = HuggingFaceEmbeddings(model_name=config["embed_model"])
    print("# 임베딩 모델 로드 성공")

    retriever_manager = RetrieverManager(
        retriever_type=config.get("retriever_type", "DenseRetriever"),
        documents=documents,
        embedding_model=embedding_model,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        top_k=config["retriever_top_k"]
    )
    retriever = retriever_manager.retriever
    print("# Retriever 생성 성공")

    # Generator 타입에 따라 선택
    generator_type = config.get("generator_type", "vllm")
    if generator_type == "api":
        generator = GeneratorAPI(config)
        print("# GeneratorAPI (Transformers) 생성 성공")
    else:
        generator = Generator(config)
        print("# Generator (vLLM) 생성 성공")
    
    print("# 평가 시작")
    eval_data = load_jsonl(EVAL_FILE)
    eval_results = []
    submit_results = []
    for i, item in enumerate(eval_data):
        query = item["input"]["question"]
        question_type = item["input"].get("question_type", "")
        context = "\n".join([doc.page_content for doc in retriever.retrieve(query, config["retriever_top_k"])] )
        preds = generator.generate(query, context, question_type)
        # 두 Generator 모두 동일한 인터페이스로 사용 가능
        answer = preds[0].outputs[0].text.strip()
        golden_answer = item.get("output", {}).get("answer", "").strip()
        # 평가용 저장
        eval_results.append({
            "id": item["id"],
            "input": {
                "question_type": question_type,
                "question": query
            },
            "output": {
                "answer": answer,
                "golden_answer": golden_answer
            }
        })
        # 제출용 저장
        submit_results.append({
            "id": item["id"],
            "input": {
                "question_type": question_type,
                "question": query
            },
            "output": {
                "answer": answer
            }
        })
        if i < 3:
            print(f"[{i+1}] 질문: {query}")
            print("답변:", answer)
            print("참고 문서:")
            for doc in retriever.retrieve(query, config["retriever_top_k"]):
                print("-", doc.page_content)
            print("\n" + "="*50 + "\n")

    # outputs 폴더 생성
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    # 평가용 저장
    eval_output_path = get_output_path(split, "eval")
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    # 제출용 저장
    submit_output_path = get_output_path(split, "submit")
    with open(submit_output_path, "w", encoding="utf-8") as f:
        json.dump(submit_results, f, ensure_ascii=False, indent=2)
    print(f"# 평가/제출 결과 저장 완료: {eval_output_path}, {submit_output_path}")