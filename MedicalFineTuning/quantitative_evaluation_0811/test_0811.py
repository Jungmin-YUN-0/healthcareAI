import argparse
import json
import os
import logging
from datetime import datetime
from typing import List, Dict
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import nltk
from transformers import AutoTokenizer

# Suppress vLLM verbose logging
logging.getLogger("vllm").setLevel(logging.WARNING)

def get_model_config(model_name):
    """
    모델별 설정을 반환하는 함수
    """
    model_configs = {
        "yanolja/EEVE-Korean-10.8B-v1.0": {
            "prompt_template": "Human: {prompt}\nAssistant: {response}",
            "stop_criteria": ["Human:", "\n\nHuman:", "Assistant:", "\n\nAssistant:"],
            "model_type": "eeve"
        },
        "trillionlabs/Tri-7B": {
            "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
            "stop_criteria": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
            "model_type": "tri"
        },
        "skt/A.X-4.0-Light": {
            "prompt_template": "<s>[INST] {prompt} [/INST] {response}</s>",
            "stop_criteria": ["[INST]", "[/INST]", "<s>", "</s>"],
            "model_type": "ax"
        }
    }
    
    for model_key in model_configs:
        if model_key in model_name or model_name in model_key:
            return model_configs[model_key]
    
    # Default configuration for other models
    return {
        "prompt_template": "질문: {prompt}\n답변: {response}",
        "stop_criteria": ["질문:", "\n질문:", "답변:", "\n답변:"],
        "model_type": "default"
    }

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
            formatted_prompt = f"<|im_start|>system\nYou are Trillion, created by TrillionLabs. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        elif model_config["model_type"] == "ax":
            formatted_prompt = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"다음 질문에 대해 정확하고 간결하게 답변하세요.\n\n질문: {input_text}\n답변: "
    
    return formatted_prompt

def calculate_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    예측과 정답 간의 다양한 메트릭을 계산하는 함수
    """
    # NLTK 데이터 다운로드 (필요시)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    # BLEU score 계산
    smoothing_function = SmoothingFunction().method1
    bleu_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = pred.split()
        gt_tokens = [gt.split()]  # Reference는 list of lists 형태
        
        try:
            bleu = sentence_bleu(gt_tokens, pred_tokens, smoothing_function=smoothing_function)
            bleu_scores.append(bleu)
        except:
            bleu_scores.append(0.0)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    # ROUGE scores 계산
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        try:
            scores = scorer.score(gt, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        except:
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)
    
    # METEOR score 계산
    meteor_scores = []
    for pred, gt in zip(predictions, ground_truths):
        try:
            meteor = meteor_score([gt.split()], pred.split())
            meteor_scores.append(meteor)
        except:
            meteor_scores.append(0.0)
    
    # BERTScore 계산
    try:
        P, R, F1 = bert_score(predictions, ground_truths, lang="ko", verbose=False)
        bert_precision = P.mean().item()
        bert_recall = R.mean().item()
        bert_f1 = F1.mean().item()
    except:
        bert_precision = 0.0
        bert_recall = 0.0
        bert_f1 = 0.0
    
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

def run_inference(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    """Run inference on a batch of prompts using vLLM"""
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract generated text from outputs
    generated_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
    
    return generated_texts


def main():
    parser = argparse.ArgumentParser(description="Medical QA evaluation script")
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--cache_dir', type=str, required=True, help='Directory to cache model')
    parser.add_argument('--result_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--max_length', type=int, default=1024, help='Max length')
    parser.add_argument('--data_path', type=str, default="", help='Path to test dataset (empty for predefined questions)')
    
    # Type argument for different evaluation modes
    parser.add_argument("--type", type=int, choices=[1, 2, 3], required=True,
                       help="Evaluation type: 1 (structured), 2 (shortened), 3 (shortened_50)")
    
    # Sampling params arguments 추가
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling')
    
    # System prompt arguments
    parser.add_argument("--use_system_prompt", type=str2bool, default=False, help="Whether to use system prompt")
    parser.add_argument("--system_prompt_file", type=str, default='./system_prompt.txt', 
                       help="Path to file containing system prompt")
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="System prompt text (alternative to file)")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    # Add current date to result path
    current_date = datetime.now().strftime("%m%d")
    if not args.result_path.endswith(current_date):
        args.result_path = os.path.join(args.result_path, current_date)
    # Load vLLM model with conditional cache_dir handling
    if args.cache_dir != "":
        llm = LLM(
            model=args.model_name,
            download_dir=args.cache_dir,
            trust_remote_code=True,
            dtype='bfloat16',
            load_format="auto",  # 또는 "safetensors"
            max_model_len=4096,
            disable_custom_all_reduce=True,
            enforce_eager=True  # Disable torch compile
        )
    else:
        llm = LLM(
            model=args.model_name,
            trust_remote_code=True,
            dtype='bfloat16',
            load_format="auto",  # 또는 "safetensors"
            disable_custom_all_reduce=True,
            enforce_eager=True  # Disable torch compile
        )
    
    print("Model loaded successfully!")
    
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
    
    # Load test dataset or use predefined questions
    if args.data_path == "":
        print("Using predefined test questions")
        questions = [
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
        ground_truth = None
        output_key = None
    else:
        print(f"Loading test dataset from: {args.data_path}")
        from datasets import load_from_disk
        dataset = load_from_disk(args.data_path)
        test_dataset = dataset['test']
        
        # Define output key based on type
        output_keys = {
            1: 'output_structured',
            2: 'output_shortened', 
            3: 'output_shortened_50'
        }
        output_key = output_keys[args.type]
        
        # Extract questions and ground truth from test dataset
        questions = [example['input'] for example in test_dataset]
        ground_truth = [example[output_key] for example in test_dataset]
    
    # Define system prompts based on type (same as train_0811.py)
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
    
    # Set system prompt based on type
    if args.type in system_prompts:
        system_prompt = system_prompts[args.type]
        args.use_system_prompt = True
    
    print(f"Number of questions for inference: {len(questions)}")
    if output_key:
        print(f"Using output field: {output_key}")
    print(f"Evaluation type: {args.type}")

    # Get model configuration for stop criteria
    model_config = get_model_config(args.model_name)
    
    # Prepare inputs for inference with custom prompt format
    inputs = []
    for question in questions:
        formatted_prompt = generate_prompt(question, args.use_system_prompt, system_prompt, args.model_name)
        inputs.append(formatted_prompt)
    
    # Set up sampling parameters with model-specific stop criteria
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=512,  # Adjust based on your needs
        stop=model_config["stop_criteria"],  # Model-specific stop tokens
    )
    
    print("Running inference...")
    
    # Run inference one by one (single batch)
    predictions = []
    
    # for i, input_prompt in enumerate(tqdm(inputs, desc=f"Processing inputs (total: {len(inputs)})", unit="input")):
    prediction = run_inference(llm, inputs, sampling_params)
    predictions.extend(prediction)
    
    # # Parse predictions based on evaluation type
    # print("Parsing predictions...")
    # parsed_predictions = []
    # for pred in predictions:
    #     parsed_pred = parse_response_by_type(pred, args.type)
    #     parsed_predictions.append(parsed_pred)
    
    # # Parse ground truth based on evaluation type (if needed)
    # parsed_ground_truth = []
    # for gt in ground_truth:
    #     if args.type in [1, 2] and isinstance(gt, str) and gt.startswith('{"'):
    #         # Ground truth is already in JSON format, parse it
    #         parsed_gt = parse_response_by_type(gt, args.type)
    #         parsed_ground_truth.append(parsed_gt)
    #     else:
    #         # Ground truth is plain text
    #         parsed_ground_truth.append(gt)
    
    # Calculate evaluation metrics (only if ground truth exists)
    if ground_truth is not None:
        print("Calculating evaluation metrics...")
        metrics = calculate_metrics(predictions, ground_truth)
    else:
        print("No ground truth available - skipping metrics calculation")
        metrics = {}
    
    # Prepare results
    results = []
    for i in range(len(questions)):
        result = {
            "question": inputs[i],
            "model_prediction": predictions[i],
            "input_text": questions[i]
        }
        if ground_truth is not None:
            result["ground_truth"] = ground_truth[i]
        results.append(result)
    
    # Generate filename based on model name and cache_dir condition
    if args.cache_dir == "":
        # Remove specific paths from model_name if cache_dir is empty
        model_name_clean = args.model_name.replace('/home/project/rapa/final_model', '')
        model_name_clean = model_name_clean.replace('/home/project/rapa/ckpt', '')
        if model_name_clean.startswith('/'):
            model_name_clean = model_name_clean[1:]
        model_name_for_file = model_name_clean.replace('/', '_').replace('\\', '_')
    else:
        # Use model_name as is (for HuggingFace models)
        model_name_for_file = args.model_name.replace('/', '_').replace('\\', '_')
    
    if model_name_for_file.startswith('_'):
        model_name_for_file = model_name_for_file[1:]
    
    # Add system prompt suffix to filename
    system_prompt_suffix = "_system" if args.use_system_prompt else ""
    # Add type suffix to filename
    type_suffix = f"_type{args.type}"
    
    # Create result path with model name (distinguish predefined vs dataset)
    if args.data_path == "":
        # For predefined questions, add "predefined" to path
        result_dir = os.path.join(args.result_path, "predefined")
        result_filename = f"inference_{model_name_for_file}{system_prompt_suffix}{type_suffix}_predefined.json"
    else:
        # For dataset evaluation
        result_dir = args.result_path
        result_filename = f"inference_{model_name_for_file}{system_prompt_suffix}{type_suffix}_results.json"
    
    final_result_path = os.path.join(result_dir, result_filename)
    
    # Prepare final results with metrics
    final_results = {
        "model_name": args.model_name,
        "evaluation_type": args.type,
        "total_examples": len(results),
        "results": results
    }
    
    if output_key:
        final_results["output_field"] = output_key
    if metrics:
        final_results["metrics"] = metrics
    
    # Save results to JSON file
    print(f"Saving results to: {final_result_path}")
    os.makedirs(result_dir, exist_ok=True)
    
    with open(final_result_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print(f"INFERENCE RESULTS SUMMARY")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Evaluation Type: {args.type}")
    if output_key:
        print(f"Output Field: {output_key}")
    print(f"Total Examples: {len(results)}")
    
    if metrics:
        print("-"*50)
        print("METRICS:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper()}: {value:.4f}")
    else:
        print("No evaluation metrics (no ground truth)")
    print("="*50)
    
    print(f"\nInference completed! Results saved to {final_result_path}")
    print(f"Total examples processed: {len(results)}")


if __name__ == '__main__':
    main()
