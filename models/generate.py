import json
import os # os 모듈 추가
from vllm import LLM, SamplingParams
from vllm.logits_process import get_bad_words_logits_processors
from utils.load_data import load_config

# VLLM_USE_V1 환경 변수를 '0'으로 설정
os.environ["VLLM_USE_V1"] = "0"

config = load_config("config.json")

class Generator:
    def __init__(self, config):

        self.llm = LLM(
            model=config["generate_model"],
            trust_remote_code=True,
            gpu_memory_utilization=config["gpu_memory_utilization"],
            tensor_parallel_size=config["tensor_parallel_size"],
        )

        tokenizer = self.llm.get_tokenizer()
        chinese = self._get_chinese(tokenizer)
        logits_processors = get_bad_words_logits_processors(chinese, tokenizer)

        self.sampling_params = SamplingParams(
            max_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            repetition_penalty=config["repetition_penalty"],
            # logits_processors=logits_processors,
        )

    def _get_chinese(self, tokenizer):
        chinese_ranges = [
            (0x4E00, 0x9FFF),   # CJK Unified Ideographs
            (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
            (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
            (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
        ]
        chinese = []
        for start, end in chinese_ranges:
            for code in range(start, end + 1):
                char = chr(code)
                chinese.append(char)
        return chinese
    
    def _system_prompt(self, context: str, question_type: str):
        # question_type은 '선택형' 또는 '교정형'이 될 것입니다.
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

        return f"""
        <|im_start|>system
        # Role: 당신은 한국어 어문 규범 전문가입니다. 주어진 질문에 대해 국어 지식을 참조하여 가장 정확하고 상세한 답변을 제공하는 것이 당신의 역할입니다.

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
        - 답변은 간결하고 명확하게 작성하며, 비문이나 어색한 표현을 사용하지 않습니다.

        <|im_im_end|>
        """
    
    def _user_prompt(self, query: str):
        return f"""
        <|im_start|>user
        {query}
        <|im_end|>
        """

    def _chat_prompt(self, query: str, context: str, question_type: str):
        system_prompt = self._system_prompt(context, question_type)
        user_prompt = self._user_prompt(query)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "<|im_start|>assistant\n"},
        ]
        
    def generate(self, query: str, context: str, question_type: str):
        chat_prompt = self._chat_prompt(query, context, question_type)
        response = self.llm.chat(chat_prompt, self.sampling_params, use_tqdm=True)
        return response