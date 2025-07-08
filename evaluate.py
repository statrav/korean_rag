import json
import os
# You'll need to install these libraries:
# pip install rouge-score transformers evaluate
# For BLEURT, it's more complex, often requiring TensorFlow and a specific model.
# Consider using the 'evaluate' library which might simplify integration.
# pip install bleurt (Note: BLEURT can be tricky to install and run due to TensorFlow dependencies)

# For demonstration, we'll use dummy functions for ROUGE, BERTScore, BLEURT
# In a real scenario, you would import and use the actual libraries.
from rouge_score import rouge_scorer
# import evaluate # for BERTScore and potentially BLEURT

def calculate_exact_match(prediction, reference):
    """
    Calculates Exact Match between a prediction and a reference.
    For the '선택·교정 문장 이/가 옳다.' part.
    """
    # Normalize by removing extra spaces and case for robust comparison
    return 1 if prediction.strip().lower() == reference.strip().lower() else 0

def calculate_rouge(prediction, reference):
    """
    Calculates ROUGE score.
    You'll need to instantiate a rouge_scorer and call its score() method.
    This is a placeholder.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    # Typically, you'd pick a specific ROUGE score (e.g., rougeL.fmeasure)
    # For this example, we'll return a simplified average or just rougeL.
    return scores['rougeL'].fmeasure # Returning F-measure for ROUGE-L

def calculate_bertscore(prediction, reference):
    """
    Calculates BERTScore.
    You'll typically load a BERTScore model using the 'evaluate' library.
    This is a placeholder.
    """
    # bertscore = evaluate.load("bertscore")
    # results = bertscore.compute(predictions=[prediction], references=[reference], lang="ko")
    # return results['f1'][0] # Return the F1 score
    return 0.0 # Placeholder

def calculate_bleurt(prediction, reference):
    """
    Calculates BLEURT score.
    This is a placeholder. BLEURT often requires specific model downloads and TensorFlow.
    """
    # bleurt = evaluate.load("bleurt", module_type="metric")
    # results = bleurt.compute(predictions=[prediction], references=[reference])
    # return results['scores'][0]
    return 0.0 # Placeholder

def evaluate_model(file_path):
    """
    Evaluates the model's performance based on the provided JSON file and PDF guidelines.

    Args:
        file_path (str): The path to the JSON file containing the model's outputs
                         and golden answers.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_exact_match_score = 0
    total_rouge_score = 0
    total_bertscore = 0
    total_bleurt_score = 0
    num_entries = len(data)

    for entry in data:
        model_answer = entry['output']['answer']
        golden_answer = entry['output']['golden_answer']

        # 1. Evaluate Exact Match for the leading sentence
        # The PDF specifies "정답은 '선택·교정 문장 이/가 옳다. 이유'의 형식으로 출력해야 한다." [cite: 19]
        # And "출력 형식 중 선택·교정 문장 이/가 옳다.'는 완전 일치(Exact Match)로 평가하며..." 
        # We need to extract this leading sentence for exact match.
        # This assumes the leading sentence ends with ".가 옳다." or ".가 옳습니다." and is followed by the reason.
        # This parsing might need refinement based on actual model output variations.
        
        # Split model_answer and golden_answer into sentence and reason parts
        # This is a heuristic and might need to be more robust depending on actual output format
        model_sentence_part = model_answer.split('. ')[0] + '.' if '. ' in model_answer else model_answer
        golden_sentence_part = golden_answer.split('. ')[0] + '.' if '. ' in golden_answer else golden_answer
        
        # For the reason part, we take everything after the first sentence.
        model_reason_part = '. '.join(model_answer.split('. ')[1:]) if '. ' in model_answer else ""
        golden_reason_part = '. '.join(golden_answer.split('. ')[1:]) if '. ' in golden_answer else ""

        total_exact_match_score += calculate_exact_match(model_sentence_part, golden_sentence_part)
        
        # 2. Evaluate ROUGE, BERTScore, BLEURT for the reason part
        # "이유'는 루지(ROUGE)와 버트스코어(BERTScore), 블루알트(BLEURT) 세 지표의 평균으로 모델의 성능을 정량 평가한다." 
        total_rouge_score += calculate_rouge(model_reason_part, golden_reason_part)
        total_bertscore += calculate_bertscore(model_reason_part, golden_reason_part)
        total_bleurt_score += calculate_bleurt(model_reason_part, golden_reason_part)

    if num_entries > 0:
        avg_exact_match = total_exact_match_score / num_entries
        avg_rouge = total_rouge_score / num_entries
        avg_bertscore = total_bertscore / num_entries
        avg_bleurt = total_bleurt_score / num_entries
        
        # Average of ROUGE, BERTScore, BLEURT for the reason part 
        avg_reason_metrics = (avg_rouge + avg_bertscore + avg_bleurt) / 3 

        print(f"--- Evaluation Results ({file_path}) ---")
        print(f"Total entries processed: {num_entries}")
        print(f"Average Exact Match (Sentence Part): {avg_exact_match:.4f}")
        print(f"Average ROUGE (Reason Part): {avg_rouge:.4f}")
        print(f"Average BERTScore (Reason Part): {avg_bertscore:.4f}")
        print(f"Average BLEURT (Reason Part): {avg_bleurt:.4f}")
        print(f"Combined Average for Reason Metrics (ROUGE, BERTScore, BLEURT): {avg_reason_metrics:.4f}")
    else:
        print("No entries to evaluate.")

# Example usage:
if __name__ == "__main__":
    # Create a dummy JSON file for testing based on your example
    dummy_data = [
        {
            "id": "623",
            "input": {
                "question_type": "선택형",
                "question": "\"나는 그를 본 적이 있음을 {기억해냈다/기억해 냈다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요."
            },
            "output": {
                "answer": "\"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳습니다. \n\n'-아들/-았었다'는 형식으로 표현하면 더 자연스럽고 일반적으로 사용됩니다. 따라서 이 문장에서는 '기억해 냈다'가 더 적합합니다. 반면에 '기억해냈다'는 '기억하다 + 내다'로 분석될 수 있어 조금 불편하게 느껴질 수 있습니다. 또한, 이 용언 조합은 실제 언어 사용에서 드물게 사용되며, 정형화되지 않은 표현으로 간주되기 때문입니다. 따라서 '기억해 냈다'가 더 도움이 될 것 같습니다. \n따라서, '기억해 냈다'는自然而流畅地表达出了已经将某事记在了心里的意思，符合韩语的常用表达方式.",
                "golden_answer": "\"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳다. '기억해 냈다'는 '기억하-+-아+냈다'의 구성이다. 이처럼 '본용언+-아/-어+보조 용언' 구성인 경우 본용언과 보조 용언을 붙여 쓰는 것이 허용되지만, 이러한 구성을 갖더라도 앞말이 3음절 이상의 합성어나 파생어라면 보조 용언을 붙여 쓰는 것이 허용되지 않는다. '기억하다'는 '기억'과 '-하다'가 결합한 파생어이며 '기억해'는 3음절이다. 따라서 '기억해'와 '냈다'는 띄어 써야 한다."
            }
        },
        {
            "id": "624",
            "input": {
                "question_type": "교정형",
                "question": "다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"오늘은 퍼즐 마추기를 해 볼 거예요.\""
            },
            "output": {
                "answer": "\"오늘은 퍼즐 맞히기를 해볼 거예요.\"\n\n이 문장을 수정한 이유는 다음과 같습니다:\n\n1. **\"마추기\"** → **\"맞히기\"**: \"마추기\"는 일반적으로 운동이나 경기에 사용되는 용어이며, 퍼즐과 관련성이 없습니다. 반면 \"맞히기\"는 퍼즐을 해결하거나 답을 찾는 행동을 정확하게 표현합니다.\n\n2. **\"해볼\"** → **\"해보려고\" 또는 단순히 **\"해볼\"**: \"해볼\" 자체는 상황적이고 자연스러운 표현입니다만, 여기서는 조금 더 구체적인 의도를 드러내기 위해 \"해보려고\"나 \"해볼까\"처럼 조사를 추가하는 것이 좋습니다. 하지만 이 경우 \"해볼\" 역시 문제 없으므로 선택 사항입니다.\n\n따라서 가장 자연스럽게 들리는 표현은 \"오늘은 퍼즐 맞히기를 해볼 거예요.\"라고 할 수 있습니다.",
                "golden_answer": "\"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다. '제자리에 맞게 붙이다, 주문하다, 똑바르게 하다, 비교하다' 등의 뜻이 있는 말은 '마추다'가 아닌 '맞추다'로 적는다."
            }
        }
    ]

    dummy_file_name = "evaluation_data.json"
    with open(dummy_file_name, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, ensure_ascii=False, indent=2)

    evaluate_model(dummy_file_name)

    # Clean up the dummy file
    os.remove(dummy_file_name)