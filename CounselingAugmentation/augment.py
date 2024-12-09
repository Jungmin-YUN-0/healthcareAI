from __future__ import annotations
import os
import json
import argparse
from itertools import permutations
import warnings
import openai # tested on openai==1.43.0
import pandas as pd # pandas==1.5.2
from tqdm.auto import tqdm # tqdm==4.64.1
warnings.filterwarnings("ignore")

# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def construct_prompt(prompt: str, summaries: list[str], dialogues: list[str], input_summary: str) -> str:
    messages = []
    messages.append({
        "role": "system",
        "content": [{
            "text": prompt,
            "type": "text",
        }]
    })

    assert len(summaries) == len(dialogues)
    for s, d in zip(summaries, dialogues):
        messages.append({
            "role": "user",
            "content": [{
                "text": s,
                "type": "text",
            }]
        })

        messages.append({
            "role": "assistant",
            "content": [{
                "text": d,
                "type": "text",
            }]
        })

    messages.append({
        "role": "user",
        "content": [{
            "text": input_summary,
            "type": "text",
        }]
    })

    return messages

def load_example(input_path: str) -> pd.DataFrame:
    # Load example data
    data = pd.read_csv(input_path, header=None)
    summary, dialogue = "", ""

    # Parse summary
    summary += "## 기본 정보\n"
    for i in range(0, 9):
        summary += data.iloc[i, 0].replace("기본 정보-", "- ") + ": " + data.iloc[i, 1] + "\n"

    summary += "## 주호소문제\n" + "### 주호소문제\n" + data.iloc[9, 1] + "\n" + "### 스트레스 요인 및 상황\n" + data.iloc[10, 1] + "\n"
    summary += "## 생활 및 관계\n" + "### 생활 패턴\n" + data.iloc[11, 1] + "\n" + "### 직장/학교 생활\n" + data.iloc[12, 1] + "\n" + "### 가족/친구 관계\n" + data.iloc[13, 1] + "\n"
    summary += "## 이전 병력 사항\n" + "### 정신과 이용 경험\n" + data.iloc[14, 1] + "\n" + "### 상담 서비스 이용 경험\n" + data.iloc[15, 1] + "\n" + "### 심리평가 경험\n" + data.iloc[16, 1] + "\n"

    summary += "## 증상 파악\n"
    for i in range(17, 25):
        summary += data.iloc[i, 0].replace("증상 파악-", "- ") + ": " + data.iloc[i, 1] + "\n"

    summary += "## 기타/추가 사항\n" + data.iloc[25, 1] + "\n"

    if len(data) == 26:
        return summary, dialogue
    else:
        # Parse dialogue
        for i in range(26, len(data)):
            dialogue += data.iloc[i, 0].replace("상담-", "- ") + ": " + data.iloc[i, 1] + "\n"

    return summary, dialogue

def construct_output(input_path: str, generated_dialogue: str) -> str:
    # pd.DataFrame
    input_summary = pd.read_csv(input_path, header=None)

    # Parse dialogue
    for turn in generated_dialogue.split("\n"):
        try:
            role = turn.split(": ")[0]
            content = turn.split(": ")[1]
        except:
            return -1 # Error

        if "상담자" in role:
            input_summary = input_summary.append(pd.Series(["상담-상담자", content]), ignore_index=True)
        elif "내담자" in role:
            input_summary = input_summary.append(pd.Series(["상담-내담자", content]), ignore_index=True)

    return input_summary


def main(args: argparse.Namespace) -> None:
    # Load example data
    example_path_list = sorted([path for path in os.listdir(args.example_path) if path.endswith(".csv")]) # Only use .csv files

    summaries, dialogues = [], []
    for example_path in example_path_list:
        s, d = load_example(args.example_path + example_path)
        summaries.append(s)
        dialogues.append(d)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load prompt
    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    # Augment data
    input_list = sorted([path for path in os.listdir(args.input_path) if path.endswith(".csv")]) # Only use .csv files

    for input_path in tqdm(input_list, total=len(input_list), desc="Generating dialogue data"):
        input_summary, _ = load_example(args.input_path + input_path)

        if args.use_permutation:
            permutations_list = list(permutations(range(len(summaries)), 2))
        else:
            permutations_list = [(0, 1)]

        for p, n in zip(permutations_list, range(len(permutations_list))):
            i, j = p[0], p[1]
            input = construct_prompt(prompt, [summaries[i], summaries[j]], [dialogues[i], dialogues[j]], input_summary)

            response = client.chat.completions.create(
                model=args.generation_model,
                messages=input,
                temperature=args.generation_temperature,
                top_p=args.generation_top_p,
                max_tokens=args.generation_max_tokens,
                response_format={
                    "type": "text"
                },
            )

            generated_dialogue = response.choices[0].message.content
            # Parse generated dialogue and concatenate with input summary
            output = construct_output(args.input_path + input_path, generated_dialogue)
            if type(output) == int:
                continue
            # Save output
            input_number = input_path.split(".csv")[0].split("_")[1]
            output.to_csv(args.output_path + f"output_{input_number}_{n+1}.csv", header=False, index=False)

    print("Finished dialogue data generation")
    # Save setup
    setup = {
        "example_path": args.example_path,
        "input_path": args.input_path,
        "output_path": args.output_path,
        "prompt_path": args.prompt_path,
        "use_permutation": args.use_permutation,
        "generation_model": args.generation_model,
        "generation_temperature": args.generation_temperature,
        "generation_top_p": args.generation_top_p,
        "generation_max_tokens": args.generation_max_tokens,
    }
    with open(args.output_path + "setup.json", "w", encoding="utf-8") as f:
        json.dump(setup, f, indent=4)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment data using OpenAI API")
    parser.add_argument("--example_path", type=str, default="./example/", help="Path to example data")
    parser.add_argument("--input_path", type=str, default="./input/", help="Path to input data")
    parser.add_argument("--output_path", type=str, default="./output/", help="Path to output data")
    parser.add_argument("--prompt_path", type=str, default="./example/prompt_1.txt", help="Path to prompt data")
    parser.add_argument("--use_permutation", type=bool, default=False, help="Use permutation for augmentation")
    parser.add_argument("--generation_model", type=str, default="gpt-4o", help="Generation model for OpenAI API")
    parser.add_argument("--generation_temperature", type=float, default=1.0, help="Generation temperature for OpenAI API")
    parser.add_argument("--generation_top_p", type=float, default=1.0, help="Generation top p for OpenAI API")
    parser.add_argument("--generation_max_tokens", type=int, default=2048, help="Generation max tokens for OpenAI API")
    args = parser.parse_args()
    main(args)
