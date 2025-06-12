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


def main(args):
    torch._dynamo.config.suppress_errors = True

    dataset = load_from_disk(args.data_path)
    #dataset = dataset.shuffle(seed=42).select(range(1000)) #############

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
        request_outputs = llm.generate(input_prompts, sampling_params)
        for request_output in request_outputs:
            generated_text = request_output.outputs[0].text.strip()
            predicted_outputs.append(generated_text)
    except Exception as e:
        print(f"Error during generation: {e}")
        return

    predicted_outputs_ = []
    for text in predicted_outputs:
        match = re.search(r'\b[1-5]\b', text)
        predicted_outputs_.append(int(match.group()) if match else -1)  # -1 for invalid

    acc = accuracy_score(answer, predicted_outputs_)
    precision = precision_score(answer, predicted_outputs_, average='weighted', zero_division=0)
    recall = recall_score(answer, predicted_outputs_, average='weighted', zero_division=0)
    f1 = f1_score(answer, predicted_outputs_, average='weighted', zero_division=0)

    model_name_short = args.model_name.split('/')[-1]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    txt_path = os.path.join(args.result_path, f'{model_name_short}_QA_Evaluation_{timestamp}.txt')
    csv_path = os.path.join(args.result_path, f'{model_name_short}_QA_Evaluation_{timestamp}.csv')

    result_text = f"""QA Evaluation Report
    =========================================
    Model Name : {args.model_name}
    Timestamp  : {timestamp}
    =========================================

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

    args = parser.parse_args()
    main(args)
