import argparse
import json
import os
import re
import torch
from typing import List, Dict
from datetime import datetime
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset,load_from_disk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import nltk
from transformers import AutoTokenizer

# OpenAI API 추가
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not available. GPT models will not work.")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def get_available_gpus():
    """사용 가능한 GPU 개수를 자동으로 감지"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"감지된 GPU 개수: {gpu_count}")
        return gpu_count
    else:
        print("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        return 0

def clear_gpu_memory():
    """모든 GPU 메모리를 클리어 - CUDA OOM 에러 방지"""
    if torch.cuda.is_available():
        print("GPU 메모리 클리어 중...")
        
        try:
            # 현재 GPU 메모리 사용량 출력
            gpu_count = torch.cuda.device_count()
            total_memory_before = 0
            total_memory_after = 0
            
            for i in range(gpu_count):
                try:
                    allocated_before = torch.cuda.memory_allocated(device=i) / 1024**3
                    reserved_before = torch.cuda.memory_reserved(device=i) / 1024**3
                    total_memory_before += reserved_before
                    
                    print(f"  GPU {i} 클리어 전: Allocated={allocated_before:.2f}GB, Reserved={reserved_before:.2f}GB")
                except RuntimeError as e:
                    print(f"  GPU {i} 메모리 상태 확인 실패: {e}")
            
            # 모든 GPU의 캐시 클리어 (안전한 방법)
            try:
                torch.cuda.empty_cache()  # 모든 디바이스의 캐시 클리어
                torch.cuda.synchronize()  # GPU 동기화
                print("  전체 GPU 캐시 클리어 완료")
            except RuntimeError as e:
                print(f"  GPU 캐시 클리어 중 오류: {e}")
                # 개별 GPU별로 시도
                for i in range(gpu_count):
                    try:
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()
                        print(f"  GPU {i} 개별 캐시 클리어 완료")
                    except RuntimeError as e2:
                        print(f"  GPU {i} 개별 캐시 클리어 실패: {e2}")
            
            # 클리어 후 메모리 사용량 출력
            for i in range(gpu_count):
                try:
                    allocated_after = torch.cuda.memory_allocated(device=i) / 1024**3
                    reserved_after = torch.cuda.memory_reserved(device=i) / 1024**3
                    total_memory_after += reserved_after
                    
                    print(f"  GPU {i} 클리어 후: Allocated={allocated_after:.2f}GB, Reserved={reserved_after:.2f}GB")
                except RuntimeError as e:
                    print(f"  GPU {i} 클리어 후 메모리 상태 확인 실패: {e}")
            
            memory_freed = total_memory_before - total_memory_after
            if memory_freed > 0:
                print(f"총 {memory_freed:.2f}GB 메모리가 해제되었습니다.")
            else:
                print("GPU 메모리 클리어를 시도했습니다.")
                
        except Exception as e:
            print(f"GPU 메모리 클리어 중 예외 발생: {e}")
    else:
        print("CUDA를 사용할 수 없어 GPU 메모리 클리어를 건너뜁니다.")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_model_config(model_name):
    """
    모델별 설정을 반환하는 함수 (test_0811.py와 동일)
    """
    model_name_lower = model_name.lower()
    
    # EEVE 모델 체크
    if "eeve" in model_name_lower:
        return {
            "prompt_template": "Human: {prompt}\\nAssistant: {response}",
            "stop_criteria": ["Human:", "\\n\\nHuman:", "Assistant:", "\\n\\nAssistant:"],
            "model_type": "eeve"
        }
    
    # Tri 모델 체크
    if "tri" in model_name_lower:
        return {
            "prompt_template": "<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n{response}<|im_end|>",
            "stop_criteria": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
            "model_type": "tri"
        }
    
    # A.X 모델 체크
    if "a.x" in model_name_lower or "ax" in model_name_lower:
        return {
            "prompt_template": "<s>[INST] {prompt} [/INST] {response}</s>",
            "stop_criteria": ["[INST]", "[/INST]", "<s>", "</s>"],
            "model_type": "ax"
        }
    
    # Default configuration for other models
    return {
        "prompt_template": "질문: {prompt}\\n답변: {response}",
        "stop_criteria": ["질문:", "\\n질문:", "답변:", "\\n답변:"],
        "model_type": "default"
    }

def get_type_specific_system_prompt(eval_type):
    """
    Type별 시스템 프롬프트 반환 (test_0811.py와 동일)
    """
    system_prompts = {
        1: '''## 응답 출력 지침
- 출력 형식: 아래 JSON 형식 그대로 사용 (추가 텍스트/설명/기호 금지)
{"_original": "정식 답변", "_short": "간단 요약 문장"}
- _original: **반드시 3문장 이하, 70자 이내**의 간결한 답변
- _short: **2문장 이하, 20자 이내, 핵심 내용 위주의 자연스러운 대화체 요약** (개조식 금지)
예시:
{"_original":"콜레스테롤 관리가 중요해요. 운동과 식단을 신경 써보세요. 추가로 궁금한 점 있으신가요?","_short":"콜레스테롤 관리가 중요해요."}

## 역할: 건강 전문 상담사 AI
- 친절하고 전문적이며, 누구나 이해할 수 있는 건강 정보 제공
- 진단이나 치료는 하지 않으며, 필요 시 의료 전문가나 진료과 안내

## 답변 지침
1. 길이 제한
- 모든 응답은 반드시 **3문장 이하, 70자 이내**로 제한
- 부연설명 없이, 짧고 명확하게 핵심 정보만 전달
2. 대화 유도 및 추가 질문
- 짧은 응답 후, 관련 후속 질문 또는 증상 구체화 질문 제시
- 예: "언제부터 이런 증상이 있었나요?"
3. 전문의 안내
- 필요한 경우에 한해, 의심 증상에 따라 해당 진료 과 권유
- 예: "이런 증상은 내과 진료가 도움이 될 수 있어요."
4. 범위 제한
- 건강과 무관한 질문은 정중히 불가 안내 후 건강 주제로 유도
- 예: "그 부분은 답변드리기 어려워요. 건강 관련 궁금한 점 있으신가요?"
5. 한국어 사용
- 항상 자연스럽고 정확한 한국어 사용
- 요청 없는 한 영어 및 전문 용어 사용 금지''',
        2: '''## 응답 출력 지침
- 출력 형식: 아래 JSON 형식 그대로 사용 (추가 텍스트/설명/기호 금지)
{"_answer": "정식 답변", "_follow_up": "추가 질문"}
- _answer: 3문장 이하, 반드시 **45자 이내** 문장은 자연스럽게 이어져야 하며, 불필요하게 잘라내거나 딱딱하게 표현하지 않음. (개조식 금지)
- _follow_up: 답변 흐름에 맞는 자연스럽고 부드러운 후속 질문 한 문장

## 역할: 건강 전문 상담사 AI
- 친절하고 전문적이며, 일상 대화처럼 친절하고 자연스러운 어투로 건강 정보를 제공
- 전문 용어는 꼭 필요한 경우에만 사용하고, 이해하기 쉬운 표현을 우선
- 진단이나 치료는 하지 않으며, 필요 시 관련 의료 전문가나 진료과 안내

## 답변 지침
1. 길이 제한
- _answer: 45자 이내, 3문장 이하
- _follow_up: 1문장
- 핵심 정보 전달을 우선하되, 문장 간 자연스러운 연결 유지
2. _answer 작성
- 핵심 내용을 짧고 명확하게, 부드러운 문장 흐름으로 작성
3. follow_up 작성
- 답변과 자연스럽게 이어지는 후속 질문 제시
- 예: "언제부터 그러셨나요?" / "조금 더 설명해주실래요?" / "좀 더 자세히 설명드릴까요?" 등
4. 전문의 안내
- 필요한 경우, 의심 증상에 따라 해당 진료 과를 간단히 안내
- 예: "내과 진료가 도움이 될 수 있어요."
5. 건강과 무관한 질문은 정중히 불가 안내 후 건강 주제로 유도
- 예: "그 부분은 답변드리기 어려워요. 건강 관련 궁금한 점 있으신가요?"
6. 언어 규칙
- 항상 자연스럽고 정확한 한국어 사용
- 요청 없는 한 영어·전문 용어 사용 금지''',
        3: '''## 역할: 건강 전문 상담사 AI
- 친절하고 전문적이면서도 일상 대화처럼 자연스럽게 건강 정보를 제공
- 전문 용어는 꼭 필요한 경우에만 사용하고, 이해하기 쉬운 표현을 우선
- 진단이나 치료는 하지 않으며, 필요 시 관련 의료 전문가나 진료과 안내

## 답변 지침
1. 길이 제한
- 모든 응답은 반드시 **2문장 이하, 50자 이내**
- 핵심 내용만 간결하게 포함, 부연 설명 최소화
2. 답변 스타일
- 자연스럽고 친근한 어투로 응답
- 전문적이지만 쉽게 이해할 수 있는 수준으로 작성
3. 전문의 안내
- 필요한 경우, 진료과를 간단히 권유
- 예: "내과 진료 받아보세요." / "피부과에 가보세요."
4. 건강과 무관한 질문은 정중히 불가 안내
- 예: "건강 관련 질문만 도와드릴 수 있어요."
5. 언어 규칙
- 한국어로만 응답
- 전문 용어 사용 금지'''
    }
    
    return system_prompts.get(eval_type, None)

def generate_prompt(input_text: str, use_system_prompt: bool = False, system_prompt: str = None, model_name: str = None) -> str:
    """
    입력 텍스트를 프롬프트 형태로 변환하는 함수 (모델별 템플릿 적용)
    """
    # Get model configuration
    model_config = get_model_config(model_name) if model_name else get_model_config("default")
    
    if use_system_prompt and system_prompt:
        if model_config["model_type"] == "eeve":
            formatted_prompt = f"{system_prompt}\n\nHuman: {input_text}\nAssistant: "
        elif model_config["model_type"] == "tri":
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        elif model_config["model_type"] == "ax":
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"{system_prompt}\n\n질문: {input_text}\n답변: "
    else:
        if model_config["model_type"] == "eeve":
            formatted_prompt = f"Human: {input_text}\nAssistant: "
        elif model_config["model_type"] == "tri":
            formatted_prompt = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        elif model_config["model_type"] == "ax":
            formatted_prompt = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"다음 질문에 대해 정확하고 간결하게 답변하세요.\n\n질문: {input_text}\n답변: "
    
    return formatted_prompt

def load_kormedmcqa_dataset():
    """KorMedMCQA 데이터셋 로딩"""
    # 모든 subset과 dev split을 합치기
    subsets = ['doctor', 'nurse', 'dentist', 'pharm']
    questions = []
    
    for subset in subsets:
        try:
            dataset = load_dataset("sean0042/KorMedMCQA", subset)
            if 'dev' in dataset:
                dev_data = dataset['dev']
                for item in dev_data:
                    # A, B, C, D, E 선택지를 리스트로 변환
                    choices = []
                    choice_keys = ['A', 'B', 'C', 'D', 'E']
                    for key in choice_keys:
                        if key in item and item[key] is not None:
                            choices.append(item[key])
                    
                    questions.append({
                        'question': item['question'],
                        'choices': choices,
                        'answer': item['answer'],
                        'subset': subset
                    })
                print(f"Loaded {len(dev_data)} questions from {subset} subset")
        except Exception as e:
            print(f"Error loading {subset} subset: {e}")
            continue
    
    print(f"Total loaded {len(questions)} questions from KorMedMCQA")
    return questions

def load_genmedgpt_dataset():
    """GenMedGPT 데이터셋 로딩"""
    dataset_path = "/home/project/rapa/dataset/genmed_dataset"
    
    # Load from local path
    try:
        with open(os.path.join(dataset_path, "test.json"), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            questions.append({
                'question': item['input_refined'],
                'answer': item['output_short']
            })
        
        print(f"Loaded {len(questions)} questions from GenMedGPT")
        return questions
    except FileNotFoundError:
        # Fallback to HuggingFace dataset
        dataset = load_from_disk(dataset_path)
        test_data = dataset['test']
        
        questions = []
        for item in test_data:
            questions.append({
                'question': item['input_refined'],
                'answer': item['output_short']
            })
        
        print(f"Loaded {len(questions)} questions from GenMedGPT")
        return questions

def generate_kormedmcqa_prompt(question: str, choices: List[str], use_system_prompt: bool = False, system_prompt: str = None, model_name: str = None) -> str:
    """KorMedMCQA용 프롬프트 생성 (모델별 템플릿 적용)"""
    # 선택지를 1, 2, 3, 4, 5 형태로 포맷팅
    choices_text = ""
    choice_labels = ['1', '2', '3', '4', '5']
    for i, choice in enumerate(choices):
        if i < len(choice_labels):
            choices_text += f"{choice_labels[i]}. {choice}\n"
    
    # Get model configuration
    model_config = get_model_config(model_name) if model_name else get_model_config("default")
    
    if use_system_prompt and system_prompt:
        if model_config["model_type"] == "eeve":
            formatted_prompt = f"{system_prompt}\n\nHuman: {question}\n\n선택지:\n{choices_text}\n정답을 하나만 선택하세요. (답변은 반드시 1,2,3,4,5 중 숫자 하나로만 출력):\nAssistant: "
        elif model_config["model_type"] == "tri":
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}\n\n선택지:\n{choices_text}\n정답을 하나만 선택하세요. (답변은 반드시 1,2,3,4,5 중 숫자 하나로만 출력):<|im_end|>\n<|im_start|>assistant\n"
        elif model_config["model_type"] == "ax":
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}\n\n선택지:\n{choices_text}\n정답을 하나만 선택하세요. (답변은 반드시 1,2,3,4,5 중 숫자 하나로만 출력):<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"{system_prompt}\n\n질문: {question}\n\n선택지:\n{choices_text}\n정답을 하나만 선택하세요. (답변은 반드시 1,2,3,4,5 중 숫자 하나로만 출력): "
    else:
        if model_config["model_type"] == "eeve":
            formatted_prompt = f"Human: 다음 질문에 대해서 주어진 선택지 중 정답을 하나만 선택하세요. (답변은 반드시 1,2,3,4,5 중 숫자 하나로만 출력)\n\n질문: {question}\n\n선택지:\n{choices_text}\n답변:\nAssistant: "
        elif model_config["model_type"] == "tri":
            formatted_prompt = f"<|im_start|>user\n다음 질문에 대해서 주어진 선택지 중 정답을 하나만 선택하세요. (답변은 반드시 1,2,3,4,5 중 숫자 하나로만 출력)\n\n질문: {question}\n\n선택지:\n{choices_text}\n답변:<|im_end|>\n<|im_start|>assistant\n"
        elif model_config["model_type"] == "ax":
            formatted_prompt = f"<|im_start|>user\n다음 질문에 대해서 주어진 선택지 중 정답을 하나만 선택하세요. (답변은 반드시 1,2,3,4,5 중 숫자 하나로만 출력)\n\n질문: {question}\n\n선택지:\n{choices_text}\n답변:<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"다음 질문에 대해서 주어진 선택지 중 정답을 하나만 선택하세요. (답변은 반드시 1,2,3,4,5 중 숫자 하나로만 출력)\n\n질문: {question}\n\n선택지:\n{choices_text}\n답변: "
    
    return formatted_prompt

def generate_genmedgpt_prompt(question: str, use_system_prompt: bool = False, system_prompt: str = None, model_name: str = None) -> str:
    """GenMedGPT용 프롬프트 생성 (모델별 템플릿 적용)"""
    # Get model configuration
    model_config = get_model_config(model_name) if model_name else get_model_config("default")
    
    if use_system_prompt and system_prompt:
        if model_config["model_type"] == "eeve":
            formatted_prompt = f"{system_prompt}\n\nHuman: {question}\nAssistant: "
        elif model_config["model_type"] == "tri":
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        elif model_config["model_type"] == "ax":
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"{system_prompt}\n\n질문: {question}\n답변: "
    else:
        if model_config["model_type"] == "eeve":
            formatted_prompt = f"Human: {question}\nAssistant: "
        elif model_config["model_type"] == "tri":
            formatted_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        elif model_config["model_type"] == "ax":
            formatted_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"다음 질문에 대해서 답변을 생성하세요.\n\n질문: {question}\n답변: "
    
    return formatted_prompt

def run_inference(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    """Run inference on a batch of prompts using vLLM"""
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract generated text from outputs
    generated_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
    
    return generated_texts

def extract_choice_from_response(response: str, choices: List[str]) -> str:
    """모델 응답에서 선택지(1, 2, 3, 4, 5) 추출 - 개선된 버전"""
    response = response.strip()
    choice_labels = ['1', '2', '3', '4', '5']
    
    # 1. 명확한 숫자 선택지 패턴 먼저 확인
    patterns = [
        r'\b([1-5])번\b',  # "3번" 형태
        r'([1-5])번입니다',  # "3번입니다" 형태  
        r'([1-5])번이다',  # "3번이다" 형태
        r'([1-5])번이\s*정답',  # "3번이 정답" 형태
        r'정답은\s*([1-5])번',  # "정답은 3번" 형태
        r'답은\s*([1-5])번',  # "답은 3번" 형태
        r'정답\s*:\s*([1-5])번',  # "정답: 3번" 형태
        r'답\s*:\s*([1-5])번',  # "답: 3번" 형태
        r'\b([1-5])\b',  # 단독으로 나타나는 1-5
        r'답변[:\s]*([1-5])',  # "답변: 1" 형태
        r'정답[:\s]*([1-5])',  # "정답: 1" 형태
        r'선택[:\s]*([1-5])',  # "선택: 1" 형태
        r'^([1-5])',  # 문장 시작에 나타나는 1-5
        r'([1-5])\s*입니다',  # "3 입니다" 형태
        r'([1-5])\s*이다',  # "3 이다" 형태
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    # 2. 선택지 내용과 양방향 포함 관계 확인
    best_match_choice = None
    best_match_score = 0
    
    for i, choice in enumerate(choices):
        choice_upper = choice.upper().strip()
        response_upper = response.upper().strip()
        
        # 공백과 구두점 제거하여 더 유연한 매칭
        choice_clean = re.sub(r'[^\w가-힣]', '', choice_upper)
        response_clean = re.sub(r'[^\w가-힣]', '', response_upper)
        
        # 길이가 너무 짧은 경우 스킵 (의미없는 매칭 방지)
        if len(choice_clean) < 2:
            continue
        
        match_score = 0
        
        # Case 1: 선택지 내용이 응답에 포함되는 경우
        if choice_clean in response_clean:
            match_score = len(choice_clean) / len(response_clean)
        
        # Case 2: 응답 내용이 선택지에 포함되는 경우
        elif response_clean in choice_clean:
            match_score = len(response_clean) / len(choice_clean)
        
        # Case 3: 부분 매칭 - 선택지와 응답의 공통 부분 확인
        else:
            # 긴 단어들로 분할하여 매칭 확인
            choice_words = [word for word in re.findall(r'[가-힣]+|\w+', choice_clean) if len(word) >= 2]
            response_words = [word for word in re.findall(r'[가-힣]+|\w+', response_clean) if len(word) >= 2]
            
            if choice_words and response_words:
                # 공통 단어 개수 기반 매칭
                common_words = set(choice_words) & set(response_words)
                if common_words:
                    match_score = len(common_words) / max(len(choice_words), len(response_words))
        
        # 최고 점수 업데이트
        if match_score > best_match_score and match_score > 0.3:  # 임계값 설정
            best_match_score = match_score
            best_match_choice = choice_labels[i]
    
    # 3. 숫자와 선택지 내용이 함께 있는 경우 확인
    if not best_match_choice:
        for i, choice in enumerate(choices):
            choice_clean = re.sub(r'[^\w가-힣]', '', choice.upper().strip())
            response_clean = re.sub(r'[^\w가-힣]', '', response.upper().strip())
            
            # 숫자와 선택지 내용이 함께 나타나는지 확인 (어떤 문자가 사이에 와도 됨)
            number_pattern = f".*{choice_labels[i]}.*{re.escape(choice_clean[:min(len(choice_clean), 10)])}.*"
            reverse_pattern = f".*{re.escape(choice_clean[:min(len(choice_clean), 10)])}.*{choice_labels[i]}.*"
            
            if (re.search(number_pattern, response_clean, re.IGNORECASE) or 
                re.search(reverse_pattern, response_clean, re.IGNORECASE)):
                best_match_choice = choice_labels[i]
                break
    
    # 4. 최적 매칭이 있으면 반환
    if best_match_choice:
        return best_match_choice
    
    # 5. Fallback: 기존 단순 포함 관계 확인
    for i, choice in enumerate(choices):
        if choice.upper() in response.upper():
            return choice_labels[i]
    
    # 6. 매칭 실패 시 None 반환 (더 이상 기본값 'A' 사용하지 않음)
    return None

def find_best_choice_match(response: str, choices: List[str]) -> tuple:
    """응답과 선택지들 간의 최적 매칭을 찾는 보조 함수"""
    response_clean = re.sub(r'[^\w가-힣]', '', response.upper().strip())
    
    matches = []
    for i, choice in enumerate(choices):
        choice_clean = re.sub(r'[^\w가-힣]', '', choice.upper().strip())
        
        if len(choice_clean) < 2:
            continue
            
        # 다양한 매칭 방법들
        scores = []
        
        # 1. 완전 포함 관계
        if choice_clean in response_clean:
            scores.append(('inclusion_choice_in_response', len(choice_clean) / len(response_clean)))
        
        if response_clean in choice_clean:
            scores.append(('inclusion_response_in_choice', len(response_clean) / len(choice_clean)))
        
        # 2. 단어 단위 매칭
        choice_words = set(re.findall(r'[가-힣]{2,}|\w{2,}', choice_clean))
        response_words = set(re.findall(r'[가-힣]{2,}|\w{2,}', response_clean))
        
        if choice_words and response_words:
            common_words = choice_words & response_words
            if common_words:
                word_match_score = len(common_words) / len(choice_words | response_words)
                scores.append(('word_matching', word_match_score))
        
        # 3. 부분 문자열 매칭 (3글자 이상)
        choice_substrings = set()
        response_substrings = set()
        
        for length in range(3, min(len(choice_clean), len(response_clean)) + 1):
            for start in range(len(choice_clean) - length + 1):
                choice_substrings.add(choice_clean[start:start + length])
            for start in range(len(response_clean) - length + 1):
                response_substrings.add(response_clean[start:start + length])
        
        common_substrings = choice_substrings & response_substrings
        if common_substrings:
            max_common_length = max(len(s) for s in common_substrings)
            substring_score = max_common_length / max(len(choice_clean), len(response_clean))
            scores.append(('substring_matching', substring_score))
        
        # 최고 점수 선택
        if scores:
            best_score = max(scores, key=lambda x: x[1])
            matches.append((i, best_score[1], best_score[0]))
    
    return matches

def is_correct_mcqa_response(response: str, correct_choice: str, choices: List[str]) -> bool:
    """MCQA 응답이 정답인지 확인 - 개선된 버전"""
    response = response.strip()
    
    # correct_choice를 숫자 형태로 변환
    if isinstance(correct_choice, int):
        correct_choice_num = str(correct_choice)
    elif isinstance(correct_choice, str):
        # A,B,C,D,E 형태의 경우 1,2,3,4,5로 변환
        if correct_choice.upper() in ['A', 'B', 'C', 'D', 'E']:
            correct_choice_num = str(ord(correct_choice.upper()) - ord('A') + 1)
        else:
            correct_choice_num = correct_choice.strip()
    else:
        correct_choice_num = str(correct_choice)
    
    # 1. 예측된 선택지 추출 (개선된 함수 사용)
    predicted_choice = extract_choice_from_response(response, choices)
    
    # 2. 매칭이 실패한 경우 (None 반환) 틀린 것으로 처리
    if predicted_choice is None:
        return False
    
    # 3. 직접 비교
    if predicted_choice == correct_choice_num:
        return True
    
    # 4. 추가 검증: 정답 선택지와의 직접적인 내용 매칭
    try:
        correct_choice_index = int(correct_choice_num) - 1
        if 0 <= correct_choice_index < len(choices):
            correct_choice_text = choices[correct_choice_index]
            
            # 개선된 매칭 함수를 사용하여 재확인
            matches = find_best_choice_match(response, choices)
            if matches:
                # 가장 높은 점수의 매칭 확인
                best_match = max(matches, key=lambda x: x[1])
                if best_match[0] == correct_choice_index and best_match[1] > 0.3:
                    return True
    except (ValueError, IndexError):
        pass
    
    return False

def calculate_classification_metrics(predictions: List[str], labels: List[str], choices_list: List[List[str]]) -> Dict:
    """분류 성능 메트릭 계산 (MCQA용) - 개선된 버전"""
    correct_predictions = 0
    total = len(labels)
    
    # 각 예측에 대해 정답 여부 판별
    prediction_results = []
    for pred, label, choices in zip(predictions, labels, choices_list):
        is_correct = is_correct_mcqa_response(pred, label, choices)
        prediction_results.append(is_correct)
        if is_correct:
            correct_predictions += 1
    
    # 정확도 계산
    accuracy = correct_predictions / total if total > 0 else 0.0
    
    # MCQA에서는 각 문제가 독립적이고 정답/오답만 있으므로
    # Precision = Recall = Accuracy (macro average 기준)
    # 이는 각 클래스(정답 선택지)가 균등하다고 가정할 때의 계산
    
    # 더 정확한 계산을 위해 선택지별 메트릭 계산
    choice_labels = ['1', '2', '3', '4', '5']
    
    # 실제 정답과 예측된 답의 분포 계산
    true_choices = []
    pred_choices = []
    
    for pred, label, choices in zip(predictions, labels, choices_list):
        # 정답 선택지 정규화 (숫자 형태로)
        if isinstance(label, int):
            true_choice = str(label)
        elif isinstance(label, str):
            # A,B,C,D,E 형태의 경우 1,2,3,4,5로 변환
            if label.upper() in ['A', 'B', 'C', 'D', 'E']:
                true_choice = str(ord(label.upper()) - ord('A') + 1)
            else:
                true_choice = label.strip()
        else:
            true_choice = str(label)
        true_choices.append(true_choice)
        
        # 예측된 선택지 추출
        pred_choice = extract_choice_from_response(pred, choices)
        # None인 경우 빈 문자열로 처리 (어떤 선택지와도 매칭되지 않도록)
        if pred_choice is None:
            pred_choice = 'NONE'
        pred_choices.append(pred_choice)
    
    # 선택지별 정밀도, 재현율, F1 계산
    from collections import defaultdict
    
    # 각 선택지별 TP, FP, FN 계산
    choice_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for true_choice, pred_choice in zip(true_choices, pred_choices):
        if true_choice == pred_choice:
            choice_metrics[true_choice]['tp'] += 1
        else:
            choice_metrics[true_choice]['fn'] += 1
            choice_metrics[pred_choice]['fp'] += 1
    
    # 전체 정밀도, 재현율, F1 계산 (macro average)
    precisions = []
    recalls = []
    f1s = []
    
    for choice in choice_labels:
        tp = choice_metrics[choice]['tp']
        fp = choice_metrics[choice]['fp']
        fn = choice_metrics[choice]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # Macro average 계산
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = sum(f1s) / len(f1s)
    
    return {
        'accuracy': accuracy,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'correct_predictions': correct_predictions,
        'total_predictions': total,
    }

def calculate_text_similarity_metrics(predictions: List[str], references: List[str], tokenizer=None) -> Dict:
    """텍스트 유사도 메트릭 계산"""
    # BLEU scores
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    # 토크나이저가 없으면 기본 split 사용
    if tokenizer is None:
        print("Warning: Tokenizer not provided. Using default split() for tokenization.")
        tokenize = lambda text: text.split()
    else:
        tokenize = tokenizer.tokenize

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = [tokenize(ref)]
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    # ROUGE scores (Rouge는 토크나이저를 내부적으로 사용하므로 그대로 둠)
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = rouge.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    # METEOR scores
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        try:
            pred_tokens = tokenize(pred)
            ref_tokens = tokenize(ref)
            meteor = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(meteor)
        except Exception as e:
            # print(f"Could not calculate METEOR score for a sentence: {e}")
            meteor_scores.append(0.0)
    
    # BERTScore (BERTScore는 자체 토크나이저를 사용하므로 그대로 둠)
    try:
        P, R, F1 = bert_score(predictions, references, lang='ko', verbose=False)
        bert_f1 = F1.mean().item()
        bert_precision = P.mean().item()
        bert_recall = R.mean().item()
    except Exception as e:
        # print(f"Could not calculate BERTScore: {e}")
        bert_f1 = 0.0
        bert_precision = 0.0
        bert_recall = 0.0
    
    return {
        'bleu': avg_bleu,
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge2': sum(rouge2_scores) / len(rouge2_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores),
        'meteor': sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0,
        'bert_score_f1': bert_f1,
        'bert_score_precision': bert_precision,
        'bert_score_recall': bert_recall
    }

def is_openai_model(model_name: str) -> bool:
    """OpenAI 모델인지 확인"""
    openai_models = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']
    return any(model in model_name.lower() for model in openai_models)

def run_openai_inference(client: OpenAI, prompts: List[str], model_name: str, temperature: float = 0.0, max_tokens: int = 256) -> List[str]:
    """OpenAI API를 사용한 추론"""
    generated_texts = []
    
    for prompt in tqdm(prompts, desc="Processing with OpenAI API"):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0
            )
            generated_text = response.choices[0].message.content
            generated_texts.append(generated_text)
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            generated_texts.append("")
    
    return generated_texts

def generate_gpt4o_reference_answers(questions_data: List[Dict], result_path: str, system_prompt: str = None) -> List[str]:
    """GPT-4o로 reference 답변을 생성하고 저장"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package is required for GPT-4o reference generation")
    
    # Check if GPT-4o answers already exist
    gpt4o_file = os.path.join(result_path, "gpt4o_reference_answers.json")
    
    if os.path.exists(gpt4o_file):
        print(f"Loading existing GPT-4o reference answers from {gpt4o_file}")
        with open(gpt4o_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            answer_list = [item['answers'] for item in data]
            return answer_list
    
    print("Generating GPT-4o reference answers...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # Generate prompts
    prompts = []
    for item in questions_data:
        prompt = generate_genmedgpt_prompt(item['question'], True, system_prompt, "gpt-4o")
        prompts.append(prompt)
    
    # Generate answers using GPT-4o
    gpt4o_answers = run_openai_inference(
        client, 
        prompts, 
        "gpt-4o", 
        temperature=0.0, 
        max_tokens=512
    )
    
    # Save GPT-4o answers
    os.makedirs(result_path, exist_ok=True)
    gpt4o_data = {
        'model': 'gpt-4o',
        'total_questions': len(questions_data),
        'answers': gpt4o_answers,
        'questions': [item['question'] for item in questions_data]
    }
    
    with open(gpt4o_file, 'w', encoding='utf-8') as f:
        json.dump(gpt4o_data, f, ensure_ascii=False, indent=2)
    
    print(f"GPT-4o reference answers saved to {gpt4o_file}")
    return gpt4o_answers

def main():
    parser = argparse.ArgumentParser(description="Medical QA evaluation script")
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--cache_dir', type=str, default='', help='Directory to cache model')
    parser.add_argument('--result_path', type=str, default='./result', required=True, help='Path to save results')
    parser.add_argument('--max_length', type=int, default=256, help='Max length')
    
    # Multi-GPU arguments
    parser.add_argument('--tensor_parallel_size', type=int, default=None, help='Number of GPUs to use for tensor parallelism (auto-detected if not specified)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='GPU memory utilization ratio (0.0-1.0)')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, required=True, choices=['kormedmcqa', 'genmedgpt', 'example'], 
                       help='Dataset to evaluate on')
    
    # Sampling params arguments 추가
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling')
    
    # System prompt arguments
    parser.add_argument("--use_system_prompt", type=str2bool, default=False, help="Whether to use system prompt")
    parser.add_argument("--system_prompt_file", type=str, default='./system_prompt.txt', 
                       help="Path to file containing system prompt")
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="System prompt text (alternative to file)")
    
    # OpenAI API arguments
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key (can also use OPENAI_API_KEY env var)")
    
    # Add GPT-4o reference generation argument
    parser.add_argument("--generate_gpt4o_reference", type=str2bool, default=False,
                       help="Generate GPT-4o reference answers for GenMedGPT (only when model is gpt-4o)")
    
    # Type argument for different evaluation modes (from test_0811.py)
    parser.add_argument("--type", type=int, choices=[1, 2, 3], default=None,
                       help="Evaluation type: 1 (structured), 2 (shortened), 3 (shortened_50)")
    
    
    args = parser.parse_args()
    
    # GPU 자동 감지 및 tensor_parallel_size 설정
    if args.tensor_parallel_size is None:
        available_gpus = get_available_gpus()
        if available_gpus > 0:
            args.tensor_parallel_size = available_gpus
            print(f"tensor_parallel_size가 자동으로 {args.tensor_parallel_size}로 설정되었습니다.")
        else:
            args.tensor_parallel_size = 1
            print("GPU를 사용할 수 없어 tensor_parallel_size를 1로 설정합니다.")
    else:
        print(f"사용자 지정 tensor_parallel_size: {args.tensor_parallel_size}")
    
    # Add current date to result path
    current_date = datetime.now().strftime("%m%d")
    if not args.result_path.endswith(current_date):
        args.result_path = os.path.join(args.result_path, current_date)
    
    print(f"Results will be saved to: {args.result_path}")
    
    print(f"Loading model: {args.model_name}")
    
    
    # Check if model is OpenAI model
    use_openai = is_openai_model(args.model_name)
    
    tokenizer = None  # Initialize tokenizer

    if (use_openai):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is required for GPT models. Install with: pip install openai")
        
        # Initialize OpenAI client
        api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --openai_api_key argument")
        
        client = OpenAI(api_key=api_key)
        llm = None
        print("OpenAI client initialized successfully!")
    else:
        # Load vLLM model and tokenizer
        print(f"Initializing vLLM with {args.tensor_parallel_size} GPU(s)")
        print(f"GPU memory utilization: {args.gpu_memory_utilization}")
        
        if args.cache_dir != "":
            llm = LLM(
                model=args.model_name,
                download_dir=args.cache_dir,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=True,
                dtype='bfloat16',
                load_format="auto",
                max_model_len=4096,
                disable_custom_all_reduce=True,
                enforce_eager=True
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        else:
            llm = LLM(
                model=args.model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=True,
                dtype='bfloat16',
                load_format="auto",
                disable_custom_all_reduce=True,
                enforce_eager=True
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        client = None
        print("vLLM model loaded successfully!")
        if tokenizer:
            print("Tokenizer loaded successfully!")
        
        # 모델 로드 후 메모리 사용량 출력
        if torch.cuda.is_available():
            print("\n모델 로드 후 GPU 메모리 사용량:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(device=i) / 1024**3
                reserved = torch.cuda.memory_reserved(device=i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                utilization = (reserved / total) * 100
                print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved ({utilization:.1f}%)")
    
    # Load system prompt if specified
    system_prompt = None
    if args.use_system_prompt:
        if args.system_prompt_file:
            try:
                with open(args.system_prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt = f.read().strip()
                print(f"시스템 프롬프트를 파일에서 로드했습니다: {args.system_prompt_file}")
            except FileNotFoundError:
                print(f"시스템 프롬프트 파일을 찾을 수 없습니다: {args.system_prompt_file}")
                args.use_system_prompt = False
        elif args.system_prompt:
            system_prompt = args.system_prompt
            print("시스템 프롬프트를 인자에서 로드했습니다.")
        else:
            print("시스템 프롬프트가 지정되지 않았습니다. 기본 프롬프트를 사용합니다.")
            args.use_system_prompt = False
    
    # Set type-specific system prompt if type is specified (but not for KorMedMCQA)
    if args.type is not None and args.dataset != 'kormedmcqa':
        type_system_prompt = get_type_specific_system_prompt(args.type)
        if type_system_prompt:
            system_prompt = type_system_prompt
            args.use_system_prompt = True
            print(f"Type {args.type}에 해당하는 시스템 프롬프트를 사용합니다.")
    elif args.type is not None and args.dataset == 'kormedmcqa':
        print(f"KorMedMCQA 데이터셋에서는 타입별 시스템 프롬프트를 사용하지 않습니다.")
    
    # Load dataset based on selection
    if args.dataset == 'kormedmcqa':
        questions_data = load_kormedmcqa_dataset()
        use_mcqa_format = True
    elif args.dataset == 'genmedgpt':
        questions_data = load_genmedgpt_dataset()
        use_mcqa_format = False
        # Force system prompt usage for GenMedGPT
        if not args.use_system_prompt:
            print("GenMedGPT 평가에는 시스템 프롬프트를 사용합니다.")
            args.use_system_prompt = True
        
        # Generate or load GPT-4o reference answers for GenMedGPT
        if args.model_name.lower() == "gpt-4o" and args.generate_gpt4o_reference:
            # This run is for generating GPT-4o reference answers
            print("Generating GPT-4o reference answers for GenMedGPT dataset...")
        elif args.dataset == 'genmedgpt':
            # Load GPT-4o reference answers for comparison
            try:
                gpt4o_reference_answers = generate_gpt4o_reference_answers(questions_data, args.result_path, system_prompt)
                print(f"Loaded {len(gpt4o_reference_answers)} GPT-4o reference answers")
                
                # Replace original answers with GPT-4o answers for evaluation
                for i, item in enumerate(questions_data):
                    if i < len(gpt4o_reference_answers):
                        item['answer'] = gpt4o_reference_answers[i]
                        item['original_answer'] = item.get('answer', '')  # Keep original if exists
                        
            except Exception as e:
                print(f"Error loading GPT-4o reference answers: {e}")
                print("Using original GenMedGPT answers as reference")
    else:  # example
        # 기존 예시 질문들 사용
        questions_data = [
            {"question": q} for q in [
                "고혈압 관리 방법은?",
                "고혈압에 좋은 음식은?",
                "혈압을 낮추는 운동은 어떤 것이 있나요?",
                "요즘 가슴이 좀 답답하고 숨이 차는 것 같아요. 무슨 문제일까요?",
                "최근에 두통이 심하고 목과 어깨가 뻣뻣한 증상이 있어요.",
                "당뇨가 있으면 어떤 음식을 피해야 하나요?",
                "당뇨 관리 방법은?",
                "당뇨에 좋은 음식은?",
                "당뇨에 좋은 운동은?",
                "당뇨에 대한 검진은 어떻게 하나요?",
                "안녕하세요, 며칠 전부터 심한 두통과 열이 있습니다. 너무 걱정이 되어서 선생님을 만나러 왔습니다.",
                "발가락이 붉고 감염된 이상하게 생긴 피부가 있습니다. 또한 많이 아프고 정상이 아닌 것 같아요.",
                "제 눈이 교차하는 것 같고 선명하게 보이지 않습니다. 무슨 증상일까요?",
                "어제부터 목에 통증이 있습니다. 원인이 무엇일까요?",
                "한동안 부비동에 통증이 있었습니다. 무슨 병일까요?",
                "저는 웨스트나일 바이러스 진단을 받았는데 어떤 약을 복용해야 하나요?"
            ]
        ]
        use_mcqa_format = False

    print(f"Number of questions for inference: {len(questions_data)}")

    # Prepare inputs for inference with dataset-specific prompt format
    inputs = []
    for item in questions_data:
        if use_mcqa_format and args.dataset == 'kormedmcqa':
            formatted_prompt = generate_kormedmcqa_prompt(
                item['question'], 
                item['choices'], 
                args.use_system_prompt, 
                system_prompt,
                args.model_name
            )
        elif args.dataset == 'genmedgpt':
            formatted_prompt = generate_genmedgpt_prompt(
                item['question'],
                args.use_system_prompt,
                system_prompt,
                args.model_name
            )
        else:  # example
            formatted_prompt = generate_prompt(item['question'], args.use_system_prompt, system_prompt, args.model_name)
        
        inputs.append(formatted_prompt)
    
    # Get model configuration for stop criteria
    model_config = get_model_config(args.model_name)
    
    # Set up sampling parameters with model-specific and dataset-specific stop tokens
    base_stop_tokens = model_config["stop_criteria"]  # Model-specific stop tokens
    
    if args.dataset == 'kormedmcqa':
        # KorMedMCQA는 선택지만 선택하면 되므로 더 짧은 답변
        dataset_stop_tokens = ["\n질문:", "\n\n", "선택지:", "답변:", "정답:", "\n선택:"]
        effective_max_length = min(args.max_length, 64)  # 더 짧게 제한
    elif args.dataset == 'genmedgpt':
        # GenMedGPT는 의료 답변이므로 적당한 길이
        dataset_stop_tokens = ["\n질문:", "\n\n질문:", "질문:", "\n답변:", "사용자:", "의사:"]
        effective_max_length = args.max_length
    else:  # example
        dataset_stop_tokens = ["\n질문:", "\n\n질문:", "질문:", "\n답변:"]
        effective_max_length = args.max_length
    
    # Combine model-specific and dataset-specific stop tokens
    stop_tokens = list(set(base_stop_tokens + dataset_stop_tokens))
    
    print(f"Sampling parameters:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Max tokens: {effective_max_length}")
    print(f"  Stop tokens: {stop_tokens}")
    
    print("Running inference...")
    
    # Run inference based on model type
    if use_openai:
        predictions = run_openai_inference(
            client, 
            inputs, 
            args.model_name, 
            temperature=args.temperature, 
            max_tokens=effective_max_length
        )
    else:
        # vLLM inference
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=effective_max_length,
            stop=stop_tokens,
        )
        
        # Run inference in batches
        batch_size = 32  # Adjust based on your GPU memory
        predictions = []
        
        for i in tqdm(range(0, len(inputs), batch_size), desc="Processing batches"):
            batch_inputs = inputs[i:i+batch_size]
            batch_predictions = run_inference(llm, batch_inputs, sampling_params)
            predictions.extend(batch_predictions)
    
    # Prepare results
    results = []
    for i in range(len(questions_data)):
        if args.dataset == 'kormedmcqa':
            # MCQA 정답 판별
            is_correct = is_correct_mcqa_response(
                predictions[i], 
                questions_data[i]['answer'], 
                questions_data[i]['choices']
            )
            
            # 예측된 선택지 추출
            predicted_choice = extract_choice_from_response(predictions[i], questions_data[i]['choices'])
            
            result = {
                "question": questions_data[i]['question'],
                "formatted_prompt": inputs[i],
                "choices": questions_data[i]['choices'],
                "correct_answer": questions_data[i]['answer'],
                "predicted_choice": predicted_choice,
                "model_prediction": predictions[i],
                "subset": questions_data[i].get('subset', ''),
                "is_correct": is_correct
            }
        
        elif args.dataset == 'genmedgpt':
            # GenMedGPT는 텍스트 생성 태스크
            result = {
                "question": questions_data[i]['question'],
                "formatted_prompt": inputs[i],
                "reference_answer": questions_data[i]['answer'],  # GPT-4o reference or original
                "model_prediction": predictions[i],
                "original_answer": questions_data[i].get('original_answer', ''),  # Keep original if exists
            }
        
        else:  # example dataset
            result = {
                "question": questions_data[i]['question'],
                "formatted_prompt": inputs[i],
                "model_prediction": predictions[i],
            }
        
        results.append(result)
    
    # 메트릭 계산을 위해 vLLM 모델을 해제하여 GPU 메모리 확보
    if not use_openai and 'llm' in locals() and llm is not None:
        print("메트릭 계산을 위해 vLLM 모델을 메모리에서 해제합니다...")
        try:
            del llm
            
            # Python 가비지 컬렉션 강제 실행
            import gc
            gc.collect()
            
            # GPU 메모리 클리어
            clear_gpu_memory()
            
            print("vLLM 모델 해제 완료")
        except Exception as e:
            print(f"vLLM 모델 해제 중 오류: {e}")
            # 그래도 메모리 클리어는 시도
            clear_gpu_memory()
    
    # Calculate metrics based on dataset
    if args.dataset == 'kormedmcqa':
        print("Calculating classification metrics for KorMedMCQA...")
        
        # Extract predicted choices
        predicted_choices = [result['model_prediction'] for result in results]
        correct_answers = [result['correct_answer'] for result in results]
        choices_list = [result['choices'] for result in results]
        
        # Calculate metrics
        classification_metrics = calculate_classification_metrics(predicted_choices, correct_answers, choices_list)
        
        # Add metrics to results
        metrics_summary = {
            'dataset': args.dataset,
            'total_questions': len(results),
            'correct_predictions': classification_metrics['correct_predictions'],
            'metrics': {
                'accuracy': classification_metrics['accuracy'],
                'precision': classification_metrics['precision'],
                'recall': classification_metrics['recall'],
                'f1': classification_metrics['f1']
            },
            'model_name': args.model_name,
            'answer_mapping': '1=1, 2=2, 3=3, 4=4, 5=5',
            'sampling_params': {
                'temperature': args.temperature,
                'top_p': args.top_p,
                'max_length': effective_max_length,
                'stop_tokens': stop_tokens
            }
        }
        
        print(f"Classification Metrics:")
        print(f"  Correct: {classification_metrics['correct_predictions']}/{classification_metrics['total_predictions']}")
        print(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {classification_metrics['precision']:.4f}")
        print(f"  Recall (macro): {classification_metrics['recall']:.4f}")
        print(f"  F1 (macro): {classification_metrics['f1']:.4f}")
    
    elif args.dataset == 'genmedgpt':
        print("Calculating text similarity metrics for GenMedGPT...")
        
        predictions_text = [result['model_prediction'] for result in results]
        references_text = [result['reference_answer'] for result in results]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir if args.cache_dir else None)
        except:
            tokenizer = None

        # Calculate metrics
        similarity_metrics = calculate_text_similarity_metrics(predictions_text, references_text, tokenizer)
        
        metrics_summary = {
            'dataset': args.dataset,
            'total_questions': len(results),
            'metrics': similarity_metrics,
            'model_name': args.model_name,
            'reference_model': 'gpt-4o',
            'sampling_params': {
                'temperature': args.temperature,
                'top_p': args.top_p,
                'max_length': effective_max_length,
                'stop_tokens': stop_tokens
            }
        }
        
        print(f"Text Similarity Metrics (vs GPT-4o):")
        print(f"  BLEU: {similarity_metrics['bleu']:.4f}")
        print(f"  ROUGE-1: {similarity_metrics['rouge1']:.4f}")
        print(f"  ROUGE-2: {similarity_metrics['rouge2']:.4f}")
        print(f"  ROUGE-L: {similarity_metrics['rougeL']:.4f}")
        print(f"  METEOR: {similarity_metrics['meteor']:.4f}")
        print(f"  BERTScore F1: {similarity_metrics['bert_score_f1']:.4f}")
    
    else:  # example dataset
        metrics_summary = {
            'dataset': args.dataset,
            'total_questions': len(results),
            'model_name': args.model_name,
            'sampling_params': {
                'temperature': args.temperature,
                'top_p': args.top_p,
                'max_length': effective_max_length,
                'stop_tokens': stop_tokens
            }
        }
    
    # Generate filename based on model name and cache_dir condition
    if use_openai:
        model_name_for_file = args.model_name.replace('/', '_').replace('\\', '_').replace('-', '_')
    elif args.cache_dir == "":
        # Remove specific paths from model_name if cache_dir is empty
        model_name_clean = args.model_name.replace('/home/project/rapa/final_model', '')
        model_name_clean = model_name_clean.replace('/home/project/rapa/ckpt', '')
        if model_name_clean.startswith('/'):
            model_name_clean = model_name_clean[1:]
        model_name_for_file = model_name_clean.replace('/', '_').replace('\\', '_')
    else:
        # Use model_name as is (for HuggingFace models)
        model_name_for_file = args.model_name.replace('/', '_').replace('\\', '_')
    
    if (model_name_for_file.startswith('_')):
        model_name_for_file = model_name_for_file[1:]
    
    # Add dataset suffix to filename
    system_prompt_suffix = "_system" if args.use_system_prompt else ""
    # Add type suffix to filename
    type_suffix = f"_type{args.type}" if args.type is not None else ""
    
    # Create result path with model name and dataset
    result_dir = os.path.join(args.result_path, args.dataset)
    result_filename = f"inference_{model_name_for_file}{system_prompt_suffix}{type_suffix}_results.json"
    final_result_path = os.path.join(result_dir, result_filename)
    
    # Save results to JSON file with metrics
    final_results = {
        'metrics_summary': metrics_summary,
        'detailed_results': results
    }
    
    print(f"Saving results to: {final_result_path}")
    os.makedirs(result_dir, exist_ok=True)
    
    with open(final_result_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"Inference completed! Results saved to {final_result_path}")
    print(f"Total examples processed: {len(results)}")


if __name__ == '__main__':
    main()
