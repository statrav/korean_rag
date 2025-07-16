import json
import os
import re
import argparse

# 필요한 라이브러리 설치 안내:
# pip install rouge-score transformers evaluate (BERTScore/BLEURT를 사용하려면)
# BLEURT는 TensorFlow와 특정 모델이 필요하므로 설치가 복잡할 수 있습니다.

from rouge_score import rouge_scorer
# 실제 BERTScore와 BLEURT를 사용하려면 아래 주석을 해제하고 필요한 모델을 로드하세요.
# import evaluate

def calculate_exact_match(prediction, reference):
    """
    예측과 참조 간의 완전 일치(Exact Match) 점수를 계산합니다.
    "선택·교정 문장 이/가 옳다." 부분 평가에 사용됩니다.
    """
    # 양쪽 끝 공백 제거 및 소문자 변환으로 비교의 견고성 높임
    return 1 if prediction.strip().lower() == reference.strip().lower() else 0

def calculate_rouge(prediction, reference):
    """
    ROUGE 점수를 계산합니다. ROUGE-L의 F-measure를 반환합니다.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure

def calculate_bertscore(prediction, reference):
    """
    BERTScore를 계산하는 플레이스홀더 함수입니다.
    실제 환경에서는 'evaluate' 라이브러리를 사용하여 BERTScore 모델을 로드하고 사용해야 합니다.
    """
    # bertscore = evaluate.load("bertscore")
    # results = bertscore.compute(predictions=[prediction], references=[reference], lang="ko")
    # return results['f1'][0]
    return 0.5 # 데모를 위한 더미 점수

def calculate_bleurt(prediction, reference):
    """
    BLEURT 점수를 계산하는 플레이스홀더 함수입니다.
    실제 환경에서는 'evaluate' 라이브러리와 BLEURT 모델을 사용하여야 합니다.
    """
    # bleurt = evaluate.load("bleurt", module_type="metric")
    # results = bleurt.compute(predictions=[prediction], references=[reference])
    # return results['scores'][0]
    return 0.5 # 데모를 위한 더미 점수

def split_sentence_and_reason(text):
    """
    주어진 텍스트를 "선택·교정 문장 이/가 옳다." 부분과 "이유" 부분으로 분리합니다.
    다양한 모델 출력 형식에 대응하기 위해 정규 표현식과 추가 로직을 사용합니다.
    """
    # 1. 일반적인 "옳다." 또는 "옳습니다."로 끝나는 형태 파싱
    # 이 정규식은 "문장" + "가 옳다." 또는 "가 옳습니다." 형태를 찾습니다.
    # 예: "\"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳습니다."
    # 예: "\"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다."
    match = re.match(r'(.*?)(옳다\.|옳습니다\.)\s*(.*)', text, re.DOTALL)
    if match:
        sentence_part = match.group(1).strip() + match.group(2).strip()
        reason_part = match.group(3).strip()
        # "이유:" 같은 시작 부분이 있으면 제거
        if reason_part.lower().startswith("이유:"):
            reason_part = reason_part[len("이유:"):].strip()
        elif reason_part.lower().startswith("이 문장을 수정한 이유는 다음과 같습니다:"): # 624번 모델 출력 포맷 처리
            pass # 이 부분은 이유의 시작이므로 그대로 둡니다.
        return sentence_part, reason_part
    
    # 2. "문장."으로 끝나고 바로 뒤에 줄바꿈(들)으로 이유가 시작되는 형태 파싱 (예: 624번 모델 출력)
    # 즉, "...에요."\n\n이 문장을 수정한 이유는... 형태
    parts_by_newline = text.split('\n\n', 1)
    if len(parts_by_newline) > 1:
        sentence_part = parts_by_newline[0].strip()
        reason_part = parts_by_newline[1].strip()
        
        # 모델 출력의 문장 부분에 '가 옳다/옳습니다'가 없으면, 평가를 위해 정답 포맷에 맞춰 추가
        # 단, 실제 모델이 그렇게 출력하지 않았다면 Exact Match는 0점 처리될 것입니다.
        # 이 부분은 모델의 출력을 강제로 정답 형식에 맞추는 것이 아니라,
        # 정답과의 비교를 위해 임시로 일관된 형태를 만드는 시도입니다.
        if not (sentence_part.endswith('옳다.') or sentence_part.endswith('옳습니다.')):
            # "옳다." 또는 "옳습니다."가 붙지 않은 모델 출력의 경우, 비교를 위해 정답과 동일하게 뒤에 붙여봄
            # 하지만 이는 실제 모델 출력이 다르므로 EM은 0이 됩니다.
            # 이 로직은 `model_em_part`가 `golden_em_part`와 같은 형태로 보일 수 있게 정제하는 과정입니다.
            # 실제 EM 점수는 `calculate_exact_match` 함수에서 결정됩니다.
            pass # EM은 그대로 모델 출력을 사용하고, 정답과 다르다면 0점 처리되는 것이 맞습니다.
        
        return sentence_part, reason_part
    
    # 3. 그 외의 경우 (분리 패턴을 찾지 못했을 때) - 전체 텍스트를 문장으로 간주
    return text.strip(), ""


def evaluate_model(file_path):
    """
    모델의 성능을 JSON 파일과 평가 가이드라인에 따라 평가합니다.

    Args:
        file_path (str): 모델의 출력과 정답을 포함하는 JSON 파일의 경로입니다.
    """
    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다. 경로: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_exact_match_score = 0
    total_rouge_score = 0
    total_bertscore = 0
    total_bleurt_score = 0
    num_entries = len(data)

    print(f"--- 평가 시작 ({file_path}) ---")

    for i, entry in enumerate(data):
        model_output_full = entry['output']['answer'].strip()
        golden_output_full = entry['output']['golden_answer'].strip()

        # 모델 출력과 정답을 문장 부분과 이유 부분으로 분리
        model_sentence_part, model_reason_part = split_sentence_and_reason(model_output_full)
        golden_sentence_part, golden_reason_part = split_sentence_and_reason(golden_output_full)

        # 1. 문장 부분에 대한 완전 일치(Exact Match) 평가
        current_em_score = calculate_exact_match(model_sentence_part, golden_sentence_part)
        total_exact_match_score += current_em_score
        print(f"완전 일치(Exact Match) 점수: {current_em_score}")
        
        # 2. 이유 부분에 대한 ROUGE, BERTScore, BLEURT 평가
        current_rouge_score = calculate_rouge(model_reason_part, golden_reason_part)
        current_bert_score = calculate_bertscore(model_reason_part, golden_reason_part)
        current_bleurt_score = calculate_bleurt(model_reason_part, golden_reason_part)

        total_rouge_score += current_rouge_score
        total_bertscore += current_bert_score
        total_bleurt_score += current_bleurt_score

    if num_entries > 0:
        avg_exact_match = total_exact_match_score / num_entries
        avg_rouge = total_rouge_score / num_entries
        avg_bertscore = total_bertscore / num_entries
        avg_bleurt = total_bleurt_score / num_entries
        
        # 이유 부분 지표들의 평균
        avg_reason_metrics = (avg_rouge + avg_bertscore + avg_bleurt) / 3 

        print(f"\n--- 최종 평가 결과 ({file_path}) ---")
        print(f"총 처리된 항목 수: {num_entries}")
        print(f"평균 완전 일치(Exact Match) 점수 (문장 부분): {avg_exact_match:.4f}")
        print(f"평균 ROUGE 점수 (이유 부분): {avg_rouge:.4f}")
        print(f"평균 BERTScore (이유 부분): {avg_bertscore:.4f}")
        print(f"평균 BLEURT 점수 (이유 부분): {avg_bleurt:.4f}")
        print(f"이유 부분 지표들의 결합 평균 (ROUGE, BERTScore, BLEURT): {avg_reason_metrics:.4f}")
    else:
        print("평가할 항목이 없습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 출력과 정답을 포함하는 JSON 파일을 평가합니다.")
    parser.add_argument(
        "file_path", 
        type=str, 
        help="평가할 JSON 파일의 경로 (예: output_results.json)"
    )
    args = parser.parse_args()

    # 스크립트 실행 시 인자로 받은 파일 경로로 평가 함수 호출
    evaluate_model(args.file_path)