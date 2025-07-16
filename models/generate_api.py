import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_data import load_config

config = load_config("config.json")

# vLLM 호환을 위한 응답 구조체
class APIOutput:
    def __init__(self, text):
        self.text = text

class APIRequestOutput:
    def __init__(self, text):
        self.outputs = [APIOutput(text)]

class APIResponse:
    def __init__(self, text):
        self.request_output = APIRequestOutput(text)
    
    def __getitem__(self, index):
        if index == 0:
            return self.request_output
        raise IndexError("Index out of range")

class GeneratorAPI:
    def __init__(self, config):
        # A.X-4.0-Light 모델 설정
        self.model_name = config.get("api_model_name", "skt/A.X-4.0-Light")
        
        print(f"{self.model_name} 모델을 로딩중...")
        
        # 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # 생성 파라미터
        self.max_new_tokens = config.get("max_new_tokens", 1024)
        self.temperature = config.get("temperature", 0.6)
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", 40)
        self.repetition_penalty = config.get("repetition_penalty", 1.18)
        
        print(f"{self.model_name} 모델 로딩 완료!")

    def _system_prompt(self, context: str, question_type: str):
        # question_type: '선택형' or '교정형'
        if question_type == "선택형":
            task_specific_guidance = """
            - 질문에서 주어진 여러 선택지 중 어문 규범에 맞는 올바른 표현을 하나만 선택해야 합니다.
            - 선택한 표현이 왜 올바른지 그 이유를 한국어 어문 규범에 근거하여 명확하게 설명해야 합니다.
            """
            response_format_example = """
            예시 답변 형식:
            "선택된 표현"이 옳다. [선택한 이유]
            """
        elif question_type == "교정형":
            task_specific_guidance = """
            - 질문의 문장에서 어문 규범에 부합하지 않는 부분을 찾아 올바르게 교정해야 합니다.
            - 교정한 문장이 왜 올바른지 그 이유를 한국어 어문 규범에 근거하여 명확하게 설명해야 합니다.
            """
            response_format_example = """
            예시 답변 형식:
            "교정된 문장"이 옳다. [교정한 이유]
            """
        else:
            # 예상치 못한 question_type에 대한 기본 또는 오류 처리
            task_specific_guidance = """
            - 질문의 내용에 따라 어문 규범에 맞는 답변을 생성해야 합니다.
            - 답변의 이유를 반드시 포함해야 합니다.
            """
            response_format_example = """
            예시 답변 형식:
            "올바른 문장/표현"이 옳다. [이유]
            """

        return f"""# Role: 당신은 한국어 어문 규범 전문가입니다. 주어진 질문에 대해 국어 지식을 참조하여 가장 정확하고 상세한 답변을 제공하는 것이 당신의 역할입니다.

# Task:
- 사용자의 질문 유형({question_type})에 따라 다음 지시사항을 따르세요:
{task_specific_guidance.strip()}
- 모든 답변은 한국어 어문 규범(한글 맞춤법, 표준어 사정 원칙, 문장 부호 규정, 외래어 표기법 등)에 근거해야 합니다.
- 답변의 첫 부분은 반드시 "선택·교정 문장 이/가 옳다." 형식으로 시작해야 합니다.
- 답변의 뒷부분에는 선택 또는 교정한 이유를 상세하게 설명해야 합니다.

# Reference (참고 자료 및 예시):
- 아래에 제공되는 참고 자료는 질문에 답변하는 데 필요한 국어 지식의 예시입니다. 이 자료를 활용하여 답변을 생성하십시오.
{context}

# Response Type (답변 형식):
- 참고 자료의 답변 예시와 동일한 말투와 형식을 사용해야 합니다[cite: 11, 13].
{response_format_example.strip()}
- 답변은 간결하고 명확하게 작성하며, 비문이나 어색한 표현을 사용하지 않습니다."""

    def _prepare_messages(self, query: str, context: str, question_type: str):
        system_prompt = self._system_prompt(context, question_type)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        return messages

    def generate(self, query: str, context: str, question_type: str):
        """       
        Args:
            query (str): 사용자 질문
            context (str): 검색된 참고 자료
            question_type (str): 질문 유형 ('선택형' 또는 '교정형')
            
        Returns:
            APIResponse: vLLM 호환 응답 객체 (preds[0].outputs[0].text로 접근 가능)
        """
        try:
            messages = self._prepare_messages(query, context, question_type)
            
            # 채팅 템플릿 적용
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(self.model.device)
            
            # attention_mask 생성 (경고 해결)
            attention_mask = torch.ones_like(input_ids).to(self.model.device)
            
            # 응답 생성
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # 입력 프롬프트 길이 계산
            len_input_prompt = len(input_ids[0])
            
            # 응답 디코딩 (입력 프롬프트 제외)
            response_text = self.tokenizer.decode(
                output[0][len_input_prompt:], 
                skip_special_tokens=True
            )
            
            # vLLM 호환 구조로 반환
            return APIResponse(response_text.strip())
            
        except Exception as e:
            print(f"모델 생성 중 오류 발생: {e}")
            error_message = f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
            return APIResponse(error_message)

    def check_model_status(self):
        try:
            # 간단한 테스트 메시지로 모델 상태 확인
            test_messages = [
                {"role": "user", "content": "안녕하세요"}
            ]
            
            input_ids = self.tokenizer.apply_chat_template(
                test_messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(self.model.device)
            
            attention_mask = torch.ones_like(input_ids).to(self.model.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            print("✅ A.X-4.0-Light 모델 정상 작동 확인")
            return True
            
        except Exception as e:
            print(f"❌ A.X-4.0-Light 모델 상태 확인 실패: {e}")
            return False 