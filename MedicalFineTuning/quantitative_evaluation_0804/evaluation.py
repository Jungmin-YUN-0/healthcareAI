import argparse
import json
import os
import re
from typing import List, Dict
from datetime import datetime
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset,load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import nltk

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def generate_prompt(input_text: str, use_system_prompt: bool = False, system_prompt: str = None) -> str:
    """
    입력 텍스트를 프롬프트 형태로 변환하는 함수
    """
    if use_system_prompt and system_prompt:
        formatted_prompt = f"{system_prompt}\n\n질문: {input_text}\n답변: "
    else:
        formatted_prompt = f"다음 질문에 대해서 답변을 생성하세요.\n\n질문: {input_text}\n답변: "
    
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

def generate_kormedmcqa_prompt(question: str, choices: List[str], use_system_prompt: bool = False, system_prompt: str = None) -> str:
    """KorMedMCQA용 프롬프트 생성"""
    # 선택지를 A, B, C, D, E 형태로 포맷팅
    choices_text = ""
    choice_labels = ['A', 'B', 'C', 'D', 'E']
    for i, choice in enumerate(choices):
        if i < len(choice_labels):
            choices_text += f"{choice_labels[i]}. {choice}\n"
    
    if use_system_prompt and system_prompt:
        formatted_prompt = f"{system_prompt}\n\n질문: {question}\n\n선택지:\n{choices_text}\n답변 (A, B, C, D, E 중 하나만 선택): "
    else:
        formatted_prompt = f"다음 질문에 대해서 주어진 선택지 중 정답을 하나만 선택하세요.\n\n질문: {question}\n\n선택지:\n{choices_text}\n답변 (A, B, C, D, E 중 하나만 선택): "
    
    return formatted_prompt

def generate_genmedgpt_prompt(question: str, use_system_prompt: bool = False, system_prompt: str = None) -> str:
    """GenMedGPT용 프롬프트 생성"""
    if use_system_prompt and system_prompt:
        formatted_prompt = f"{system_prompt}\n\n질문: {question}\n답변: "
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
    """모델 응답에서 선택지(A, B, C, D, E) 추출 - 개선된 버전"""
    response = response.strip().upper()
    choice_labels = ['A', 'B', 'C', 'D', 'E']
    
    # 1. 명확한 선택지 패턴 먼저 확인
    patterns = [
        r'\b([A-E])\b',  # 단독으로 나타나는 A-E
        r'답변[:\s]*([A-E])',  # "답변: A" 형태
        r'정답[:\s]*([A-E])',  # "정답: A" 형태
        r'선택[:\s]*([A-E])',  # "선택: A" 형태
        r'^([A-E])',  # 문장 시작에 나타나는 A-E
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
        
        # 공백과 구두점 제거하여 더 유연한 매칭
        choice_clean = re.sub(r'[^\w가-힣]', '', choice_upper)
        response_clean = re.sub(r'[^\w가-힣]', '', response)
        
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
    
    # 3. 최적 매칭이 있으면 반환, 없으면 원래 방식으로 fallback
    if best_match_choice:
        return best_match_choice
    
    # 4. Fallback: 기존 단순 포함 관계 확인
    for i, choice in enumerate(choices):
        if choice.upper() in response:
            return choice_labels[i]
    
    # 5. 기본값 반환
    return 'A'

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
    response = response.strip().upper()
    
    # correct_choice가 숫자인 경우 문자로 변환
    if isinstance(correct_choice, int):
        correct_choice = chr(ord('A') + correct_choice - 1)
    correct_choice = correct_choice.upper()
    
    # 1. 예측된 선택지 추출 (개선된 함수 사용)
    predicted_choice = extract_choice_from_response(response, choices)
    
    # 2. 직접 비교
    if predicted_choice == correct_choice:
        return True
    
    # 3. 추가 검증: 정답 선택지와의 직접적인 내용 매칭
    correct_choice_index = ord(correct_choice) - ord('A')
    if 0 <= correct_choice_index < len(choices):
        correct_choice_text = choices[correct_choice_index]
        
        # 개선된 매칭 함수를 사용하여 재확인
        matches = find_best_choice_match(response, choices)
        if matches:
            # 가장 높은 점수의 매칭 확인
            best_match = max(matches, key=lambda x: x[1])
            if best_match[0] == correct_choice_index and best_match[1] > 0.3:
                return True
    
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
    choice_labels = ['A', 'B', 'C', 'D', 'E']
    
    # 실제 정답과 예측된 답의 분포 계산
    true_choices = []
    pred_choices = []
    
    for pred, label, choices in zip(predictions, labels, choices_list):
        # 정답 선택지 정규화
        if isinstance(label, int):
            true_choice = chr(ord('A') + label - 1)
        else:
            true_choice = label.strip().upper()
        true_choices.append(true_choice)
        
        # 예측된 선택지 추출
        pred_choice = extract_choice_from_response(pred, choices)
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
        'choice_distribution': {
            'true_choices': dict(pd.Series(true_choices).value_counts()) if 'pd' in globals() else {},
            'pred_choices': dict(pd.Series(pred_choices).value_counts()) if 'pd' in globals() else {}
        }
    }

def calculate_text_similarity_metrics(predictions: List[str], references: List[str]) -> Dict:
    """텍스트 유사도 메트릭 계산"""
    # BLEU scores
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    # ROUGE scores
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
            meteor = meteor_score([ref.split()], pred.split())
            meteor_scores.append(meteor)
        except:
            meteor_scores.append(0.0)
    
    # BERTScore
    try:
        P, R, F1 = bert_score(predictions, references, lang='ko', verbose=False)
        bert_f1 = F1.mean().item()
        bert_precision = P.mean().item()
        bert_recall = R.mean().item()
    except:
        bert_f1 = 0.0
        bert_precision = 0.0
        bert_recall = 0.0
    
    return {
        'bleu': avg_bleu,
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge2': sum(rouge2_scores) / len(rouge2_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores),
        'meteor': sum(meteor_scores) / len(meteor_scores),
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
        prompt = generate_genmedgpt_prompt(item['question'], True, system_prompt)
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
    
    args = parser.parse_args()
    
    # Add current date to result path
    current_date = datetime.now().strftime("%m%d")
    if not args.result_path.endswith(current_date):
        args.result_path = os.path.join(args.result_path, current_date)
    
    print(f"Results will be saved to: {args.result_path}")
    
    print(f"Loading model: {args.model_name}")
    
    # Check if model is OpenAI model
    use_openai = is_openai_model(args.model_name)
    
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
        # Load vLLM model with conditional cache_dir handling
        if args.cache_dir != "":
            llm = LLM(
                model=args.model_name,
                download_dir=args.cache_dir,
                gpu_memory_utilization=0.8,
                trust_remote_code=True,
                dtype='bfloat16',
                load_format="auto",
            )
        else:
            llm = LLM(
                model=args.model_name,
                trust_remote_code=True,
                dtype='bfloat16',
                load_format="auto",
            )
        
        client = None
        print("vLLM model loaded successfully!")
    
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
                system_prompt
            )
        elif args.dataset == 'genmedgpt':
            formatted_prompt = generate_genmedgpt_prompt(
                item['question'],
                args.use_system_prompt,
                system_prompt
            )
        else:  # example
            formatted_prompt = generate_prompt(item['question'], args.use_system_prompt, system_prompt)
        
        inputs.append(formatted_prompt)
    
    # Set up sampling parameters with dataset-specific stop tokens
    stop_tokens = None
    if args.dataset == 'kormedmcqa':
        # KorMedMCQA는 선택지만 선택하면 되므로 더 짧은 답변
        stop_tokens = ["\n질문:", "\n\n", "선택지:", "답변:", "정답:", "\n선택:"]
        effective_max_length = min(args.max_length, 64)  # 더 짧게 제한
    elif args.dataset == 'genmedgpt':
        # GenMedGPT는 의료 답변이므로 적당한 길이
        stop_tokens = ["\n질문:", "\n\n질문:", "질문:", "\n답변:", "사용자:", "의사:"]
        effective_max_length = args.max_length
    else:  # example
        stop_tokens = ["\n질문:", "\n\n질문:", "질문:", "\n답변:"]
        effective_max_length = args.max_length
    
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
                "choices": questions_data[i]['choices'],
                "correct_answer": questions_data[i]['answer'],
                "predicted_choice": predicted_choice,
                "model_prediction": predictions[i],
                "subset": questions_data[i].get('subset', ''),
                "raw_model_response": predictions[i],
                "is_correct": is_correct
            }
        
        elif args.dataset == 'genmedgpt':
            # GenMedGPT는 텍스트 생성 태스크
            result = {
                "question": questions_data[i]['question'],
                "reference_answer": questions_data[i]['answer'],  # GPT-4o reference or original
                "model_prediction": predictions[i],
                "raw_model_response": predictions[i],
                "original_answer": questions_data[i].get('original_answer', ''),  # Keep original if exists
            }
        
        else:  # example dataset
            result = {
                "question": questions_data[i]['question'],
                "model_prediction": predictions[i],
                "raw_model_response": predictions[i],
            }
        
        results.append(result)
    
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
            'answer_mapping': 'A=1, B=2, C=3, D=4, E=5',
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
        
        # 원본 모델 응답도 저장
        for result in results:
            result['raw_model_response'] = result['model_prediction']
        
        # Calculate metrics
        similarity_metrics = calculate_text_similarity_metrics(predictions_text, references_text)
        
        metrics_summary = {
            'dataset': args.dataset,
            'total_questions': len(results),
            'metrics': similarity_metrics,
            'model_name': args.model_name,
            'reference_model': 'gpt-4o' if any('original_genmedgpt_answer' in r for r in results) else 'original_genmedgpt',
            'sampling_params': {
                'temperature': args.temperature,
                'top_p': args.top_p,
                'max_length': effective_max_length,
                'stop_tokens': stop_tokens
            }
        }
        
        print(f"Text Similarity Metrics (vs {'GPT-4o' if any('original_genmedgpt_answer' in r for r in results) else 'Original GenMedGPT'}):")
        print(f"  BLEU: {similarity_metrics['bleu']:.4f}")
        print(f"  ROUGE-1: {similarity_metrics['rouge1']:.4f}")
        print(f"  ROUGE-2: {similarity_metrics['rouge2']:.4f}")
        print(f"  ROUGE-L: {similarity_metrics['rougeL']:.4f}")
        print(f"  METEOR: {similarity_metrics['meteor']:.4f}")
        print(f"  BERTScore F1: {similarity_metrics['bert_score_f1']:.4f}")
    
    else:  # example dataset
        # 원본 모델 응답도 저장
        for result in results:
            result['raw_model_response'] = result['model_prediction']
        
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
    # Add sampling params suffix to filename
    sampling_suffix = f"_temp{args.temperature}_tp{args.top_p}"
    
    # Create result path with model name and dataset
    result_dir = os.path.join(args.result_path, args.dataset)
    result_filename = f"inference_{model_name_for_file}{system_prompt_suffix}{sampling_suffix}_results.json"
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
