import argparse
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score
import datetime
from vllm import LLM, SamplingParams
import torch._dynamo
from utils import context_to_string
from collections import Counter
import math

def normalized_weighted_sum(responses_with_logprobs):
    """Calculate normalized weighted sum based on log probabilities"""
    answer_scores = {}
    
    for response_data in responses_with_logprobs:
        response_text = response_data['text']
        cumulative_logprob = response_data['cumulative_logprob']
        token_count = response_data['token_count']
        
        if token_count == 0:
            continue
            
        # 2. 길이로 정규화
        normalized_log_prob = cumulative_logprob / token_count
        
        # 3. exp() 변환하여 실제 확률로
        normalized_prob = math.exp(normalized_log_prob)
        
        # 4. 답변별로 가중치 합산
        answer_scores[response_text] = answer_scores.get(response_text, 0) + normalized_prob
    # print('answer_scores',answer_scores)
    if not answer_scores:
        return responses_with_logprobs[0]['text'] if responses_with_logprobs else ""
    
    return max(answer_scores.items(), key=lambda x: x[1])[0]

def apply_self_consistency(responses, num_samples):
    """Apply self-consistency by selecting response with highest normalized log probability"""
    if num_samples <= 1:
        return responses[0]['text'] if responses else ""
    
    return normalized_weighted_sum(responses)

def main(args):
    torch._dynamo.config.suppress_errors = True

    dataset = load_from_disk(args.data_path)
    answer = dataset['answer']
    context = dataset['context']
    context_str_list = [context_to_string(c) for c in context]
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
            #tensor_parallel_size=2
        )

    # Modify sampling parameters for self-consistency
    if args.use_self_consistency:
        sampling_params = SamplingParams(
            temperature=args.temperature,  # Use temperature > 0 for diversity
            max_tokens=512,
            stop_token_ids=[128009, 2],
            stop=["</s>", "<|eot_id|>", "<|end_of_text|>", "\nObservation:", "\nThought:", "Question:", "Context:", "Q:", "A:", "\nQuestion:", "\nInformation:", "\nExplanation:", "\npatient:", "\nPatient:", "\nDoctor:", "\ndoctor:", "\n"],
            n=args.num_samples,  # Generate multiple samples
            logprobs=1,  # Enable log probability calculation
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop_token_ids=[128009, 2],
            stop=["</s>", "<|eot_id|>", "<|end_of_text|>", "\nObservation:", "\nThought:", "Question:", "Context:", "Q:", "A:", "\nQuestion:", "\nInformation:", "\nExplanation:", "\npatient:", "\nPatient:", "\nDoctor:", "\ndoctor:", "\n"]
        )

    input_prompts = []
    predicted_outputs = []

    instruction = """You are a kind and professional doctor engaged in a multi-turn conversation with a patient. Below is the dialogue history between you and the patient. Based on the most recent patient response, generate the next appropriate response as a doctor.\nYour response should:\nBe brief and conversational (1–2 sentences)\nAcknowledge or empathize with the patient’s previous message\nAsk a relevant follow-up question or give simple medical advice to guide the conversation\n"""

    for cnt in tqdm(context_str_list):
        vllm_input = f'''{instruction}\n\n[Dialogue History] {cnt}\n[Next Doctor Response] doctor:'''
        input_prompts.append(vllm_input)

    try:
        print(f"Start processing ({len(input_prompts)}) prompts...")
        
        if args.use_self_consistency:
            # Process each prompt individually for self-consistency
            for prompt in tqdm(input_prompts, desc="Processing with Self-Consistency"):
            
                request_outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                
                # Extract multiple responses with log probabilities for this prompt
                responses_with_logprobs = []
                for output in request_outputs[0].outputs:
                    # Calculate token count from logprobs
                    token_count = len(output.logprobs) if output.logprobs else 0
                    
                    response_data = {
                        'text': output.text.strip(),
                        'cumulative_logprob': output.cumulative_logprob,
                        'token_count': token_count
                    }
                    responses_with_logprobs.append(response_data)
                # print('responses_with_logprobs: ',responses_with_logprobs)
                # Apply self-consistency (probability-based selection)
                final_response = apply_self_consistency(responses_with_logprobs, args.num_samples)
                predicted_outputs.append(final_response)
        else:
            # Original single-response generation
            request_outputs = llm.generate(input_prompts, sampling_params)
            for request_output in request_outputs:
                generated_text = request_output.outputs[0].text.strip()
                predicted_outputs.append(generated_text)
                
    except Exception as e:
        print(f"Error: {e}")
        return
    
    if args.cache_dir != None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_text(text):
        return tokenizer.tokenize(text)

    smoothing_fn = SmoothingFunction().method1
    bleu_scores = [sentence_bleu([tokenize_text(true)], tokenize_text(pred), smoothing_function=smoothing_fn) for true, pred in zip(answer, predicted_outputs)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(true, pred) for true, pred in zip(answer, predicted_outputs)]
    avg_rouge1 = sum([score["rouge1"].fmeasure for score in rouge_scores]) / len(rouge_scores)
    avg_rouge2 = sum([score["rouge2"].fmeasure for score in rouge_scores]) / len(rouge_scores)
    avg_rougeL = sum([score["rougeL"].fmeasure for score in rouge_scores]) / len(rouge_scores)

    meteor_scores = [meteor_score([tokenize_text(true)], tokenize_text(pred)) for true, pred in zip(answer, predicted_outputs)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    P, R, F1 = score(predicted_outputs, answer, model_type="klue/bert-base", num_layers=12, lang='ko', verbose=True)
    avg_bertscore_f1 = F1.mean().item()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    extracted_name = args.model_name.split('/')[-1]
    
    sc_suffix = f"_sc{args.num_samples}" if args.use_self_consistency else ""
    output_txt_path = f'{args.result_path}/{extracted_name}_dialogue_generation_results{sc_suffix}_{timestamp}.txt'
    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    sc_info = f"Self-Consistency   : {'Enabled' if args.use_self_consistency else 'Disabled'}\n"
    if args.use_self_consistency:
        sc_info += f"Number of Samples  : {args.num_samples}\nTemperature        : {args.temperature}\n"

    report_content = f"""Dialogue Generation Evaluation Report\n=========================================\nModel Name : {args.model_name}\nReport Date: {current_timestamp}\n{sc_info}=========================================\n\nMetrics:\n-----------------------------------------\nBLEU Score        : {avg_bleu:.4f}\nROUGE-1 F-measure : {avg_rouge1:.4f}\nROUGE-2 F-measure : {avg_rouge2:.4f}\nROUGE-L F-measure : {avg_rougeL:.4f}\nMETEOR Score      : {avg_meteor:.4f}\nBERTScore F1      : {avg_bertscore_f1:.4f}\n-----------------------------------------\n"""

    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\nEvaluation report successfully saved to: {output_txt_path}")
    except Exception as e:
        print(f"Error saving report: {e}")

    df_dialogue_details = pd.DataFrame({
        'Input_Prompt': context_str_list,
        'Ground_Truth_Output': answer,
        'Predicted_Output': predicted_outputs
    })

    output_csv_details_path = f'{args.result_path}/{extracted_name}_dialogue_generation_results{sc_suffix}_{timestamp}.csv'

    try:
        df_dialogue_details.to_csv(output_csv_details_path, index=False, encoding='utf-8-sig')
        print(f"\nDialogue generation details saved to: {output_csv_details_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dialogue generation and evaluation script")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the preprocessed dataset')
    parser.add_argument('--model_name', type=str, required=True, help='HuggingFace model name or local path')
    parser.add_argument('--result_path', type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('--cache_dir', type=str, required=True, help='cache directory for HuggingFace model')
    
    # Self-consistency arguments
    parser.add_argument('--use_self_consistency', type=bool, default=True, required=True, help='Enable self-consistency with majority voting')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to generate for self-consistency (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling when using self-consistency (default: 0.7)')

    args = parser.parse_args()
    main(args)
