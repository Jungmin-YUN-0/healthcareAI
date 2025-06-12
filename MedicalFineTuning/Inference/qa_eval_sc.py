import argparse
import torch
import pandas as pd
import os
import re
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from vllm import LLM, SamplingParams
import datetime
import torch._dynamo
from collections import Counter

def apply_majority_voting(responses, num_samples):
    """Apply majority voting by selecting the most frequent answer"""
    if num_samples <= 1:
        # Extract number from first response
        if responses:
            match = re.search(r'\b[1-5]\b', responses[0])
            return int(match.group()) if match else -1
        return -1
    
    # Extract numbers from responses
    extracted_numbers = []
    for response in responses:
        match = re.search(r'\b[1-5]\b', response)
        if match:
            extracted_numbers.append(int(match.group()))
        else:
            extracted_numbers.append(-1)  # Invalid response
    # print(f"Extracted numbers: {extracted_numbers}")
    # Count occurrences and return most frequent
    if extracted_numbers:
        counter = Counter(extracted_numbers)
        most_common = counter.most_common(1)[0][0]
        # print(f"Most common number: {most_common}")
        return most_common if most_common != -1 else -1
    
    return -1

def main(args):
    torch._dynamo.config.suppress_errors = True

    dataset = load_from_disk(args.data_path)
    # dataset = dataset.shuffle(seed=42).select(range(2)) #############
    # dataset 
    question = dataset['question']
    answer = dataset['answer']

    if args.cache_dir != "":
        llm = LLM(
            model=args.model_name,
            download_dir=args.cache_dir,
            hf_overrides={"cache_dir": args.cache_dir},
            #cache_dir=args.cache_dir,
            trust_remote_code=True,
            dtype='bfloat16',
        )
    else:
        llm = LLM(
            model=args.model_name,
            trust_remote_code=True,
            dtype='bfloat16',
        )

    # Modify sampling parameters for self-consistency
    if args.use_self_consistency:
        sampling_params = SamplingParams(
            temperature=args.temperature,  # Use temperature > 0 for diversity
            max_tokens=512,
            stop_token_ids=[128009, 2],
            stop=["</s>", "<|eot_id|>", "<|end_of_text|>", "\nObservation:", "\nThought:", "Question:", "Context:", "Q:", "A:", "[Question]", "[Answer]", "[Explanation]"],
            n=args.num_samples,  # Generate multiple samples
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop_token_ids=[128009, 2],
            stop=["</s>", "<|eot_id|>", "<|end_of_text|>", "\nObservation:", "\nThought:", "Question:", "Context:", "Q:", "A:", "[Question]", "[Answer]", "[Explanation]"]
        )

    instruction = "You are a knowledgeable medical expert. Based on the following question and multiple-choice options (1 to 5), choose the single most appropriate answer. Your response should be concise and medically accurate. If the question requires clinical judgment, select the answer that is generally correct under standard medical guidelines."

    input_prompts = []
    for idx in tqdm(range(len(question))):
        prompt = (
            f"[Question]: {question[idx]}\n[Options]:\n"
            f"1. {dataset['A'][idx]}\n2. {dataset['B'][idx]}\n3. {dataset['C'][idx]}\n4. {dataset['D'][idx]}\n5. {dataset['E'][idx]}\n\n[Answer]:"
        )
        input_prompts.append(f"{instruction}\n\n{prompt}")

    predicted_outputs = []
    try:
        print(f"Start processing {len(input_prompts)} prompts...")
        
        if args.use_self_consistency:
            # Process each prompt individually for self-consistency
            for prompt in tqdm(input_prompts, desc="Processing with Self-Consistency"):
                request_outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                
                # Extract multiple responses for this prompt
                responses = []
                for output in request_outputs[0].outputs:
                    responses.append(output.text.strip())
                print(f"Responses for prompt: {responses}")
                # Apply majority voting (returns integer)
                final_response = apply_majority_voting(responses, args.num_samples)
                # print(f"Final response after voting: {final_response}")
                predicted_outputs.append(final_response)
        else:
            # Original single-response generation
            request_outputs = llm.generate(input_prompts, sampling_params)
            for request_output in request_outputs:
                generated_text = request_output.outputs[0].text.strip()
                # Extract number from single response
                match = re.search(r'\b[1-5]\b', generated_text)
                predicted_outputs.append(int(match.group()) if match else -1)
                
    except Exception as e:
        print(f"Error during generation: {e}")
        return

    # predicted_outputs already contains integers, no need for additional processing
    predicted_outputs_ = predicted_outputs

    acc = accuracy_score(answer, predicted_outputs_)
    precision = precision_score(answer, predicted_outputs_, average='weighted', zero_division=0)
    recall = recall_score(answer, predicted_outputs_, average='weighted', zero_division=0)
    f1 = f1_score(answer, predicted_outputs_, average='weighted', zero_division=0)

    model_name_short = args.model_name.split('/')[-1]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    sc_suffix = f"_sc{args.num_samples}" if args.use_self_consistency else ""
    txt_path = os.path.join(args.result_path, f'{model_name_short}_QA_Evaluation{sc_suffix}_{timestamp}.txt')
    csv_path = os.path.join(args.result_path, f'{model_name_short}_QA_Evaluation{sc_suffix}_{timestamp}.csv')

    sc_info = f"Self-Consistency   : {'Enabled' if args.use_self_consistency else 'Disabled'}\n"
    if args.use_self_consistency:
        sc_info += f"Number of Samples  : {args.num_samples}\nTemperature        : {args.temperature}\n"

    result_text = f"""QA Evaluation Report
    =========================================
    Model Name : {args.model_name}
    Timestamp  : {timestamp}
    {sc_info}=========================================

    Metrics:
    -----------------------------------------
    Accuracy    : {acc:.4f}
    Precision   : {precision:.4f}
    Recall      : {recall:.4f}
    F1 Score    : {f1:.4f}
    -----------------------------------------
    """

    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"Saved evaluation report: {txt_path}")
    except Exception as e:
        print(f"Failed to save TXT: {e}")

    df_result = pd.DataFrame({
        'question': dataset['question'],
        'answer' : dataset['answer'],
        'option1': dataset['A'],
        'option2': dataset['B'],
        'option3': dataset['C'],
        'option4': dataset['D'],
        'option5': dataset['E'],
        'predicted_output': predicted_outputs_,
    })

    try:
        df_result.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved result CSV: {csv_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Medical QA evaluation script")
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed dataset')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--cache_dir', type=str, required=True, help='Directory to cache model')
    parser.add_argument('--result_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID to use')
    
    # Self-consistency arguments
    parser.add_argument('--use_self_consistency', type=bool, default=False, help='Enable self-consistency with majority voting')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to generate for self-consistency (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling when using self-consistency (default: 0.7)')

    args = parser.parse_args()
    main(args)
