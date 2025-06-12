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
        #request_outputs = llm.generate(input_prompts, sampling_params)
        #request_outputs = llm.chat(
        request_outputs = llm.generate(
            input_prompts, 
            sampling_params,
            #chat_template_kwargs={"enable_thinking": True},  # Set to False to strictly disable thinking
        )


        for i, request_output in enumerate(request_outputs):
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
    output_txt_path = f'{args.result_path}/{extracted_name}_dialogue_generation_results_{timestamp}.txt'
    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report_content = f"""Dialogue Generation Evaluation Report\n=========================================\nModel Name : {args.model_name}\nReport Date: {current_timestamp}\n=========================================\n\nMetrics:\n-----------------------------------------\nBLEU Score        : {avg_bleu:.4f}\nROUGE-1 F-measure : {avg_rouge1:.4f}\nROUGE-2 F-measure : {avg_rouge2:.4f}\nROUGE-L F-measure : {avg_rougeL:.4f}\nMETEOR Score      : {avg_meteor:.4f}\nBERTScore F1      : {avg_bertscore_f1:.4f}\n-----------------------------------------\n"""

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

    output_csv_details_path = f'{args.result_path}/{extracted_name}_dialogue_generation_results_{timestamp}.csv'

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

    args = parser.parse_args()
    main(args)
