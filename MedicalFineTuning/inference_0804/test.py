import argparse
import json
import os
from datetime import datetime
from typing import List, Dict
from vllm import LLM, SamplingParams
from tqdm import tqdm

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
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            dtype='bfloat16',
            load_format="auto",  # 또는 "safetensors"

        )
    else:
        llm = LLM(
            model=args.model_name,
            trust_remote_code=True,
            dtype='bfloat16',
            load_format="auto",  # 또는 "safetensors"

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
    
    # The user's questions for inference
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
    print(f"Number of questions for inference: {len(questions)}")

    # Prepare inputs for inference with custom prompt format
    inputs = []
    for question in questions:
        formatted_prompt = generate_prompt(question, args.use_system_prompt, system_prompt)
        inputs.append(formatted_prompt)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_length,  # Adjust based on your needs
        stop=None,  # Add stop tokens if needed
    )
    
    print("Running inference...")
    
    # Run inference in batches
    batch_size = 32  # Adjust based on your GPU memory
    predictions = []
    
    for i in tqdm(range(0, len(inputs), batch_size), desc="Processing batches"):
        batch_inputs = inputs[i:i+batch_size]
        batch_predictions = run_inference(llm, batch_inputs, sampling_params)
        predictions.extend(batch_predictions)
    
    # Prepare results
    results = []
    for i in range(len(questions)):
        result = {
            "question": questions[i],
            "model_prediction": predictions[i]
        }
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
    # Add sampling params suffix to filename
    sampling_suffix = f"_temp{args.temperature}_tp{args.top_p}"
    
    # Create result path with model name
    result_dir = args.result_path
    result_filename = f"inference_{model_name_for_file}{system_prompt_suffix}{sampling_suffix}_results_example.json"
    final_result_path = os.path.join(result_dir, result_filename)
    
    # Save results to JSON file
    print(f"Saving results to: {final_result_path}")
    os.makedirs(result_dir, exist_ok=True)
    
    with open(final_result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Inference completed! Results saved to {final_result_path}")
    print(f"Total examples processed: {len(results)}")


if __name__ == '__main__':
    main()
