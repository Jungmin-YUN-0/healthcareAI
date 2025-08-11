from __future__ import annotations
import os
import json
import argparse
import warnings
from itertools import combinations
from pathlib import Path

import openai
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# API 클라이언트 설정
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
client = openai.OpenAI(api_key=api_key)


def construct_prompt(prompt: str, summaries: list[str], dialogues: list[str], input_summary: str) -> list:
    messages = []
    messages.append({
        "role": "system",
        "content": [{"text": prompt, "type": "text"}]
    })

    assert len(summaries) == len(dialogues)
    for s, d in zip(summaries, dialogues):
        messages.append({
            "role": "user",
            "content": [{"text": s, "type": "text"}]
        })
        messages.append({
            "role": "assistant",
            "content": [{"text": d, "type": "text"}]
        })

    messages.append({
        "role": "user",
        "content": [{"text": input_summary, "type": "text"}]
    })
    return messages


def load_example(input_path: str):
    # Load example data
    try:
        data = pd.read_csv(input_path, header=None, encoding='utf-8')
    except:
        data = pd.read_csv(input_path, header=None, encoding='cp949')

    summary, dialogue = "", ""

    # 요약 정보 파싱
    summary_parts = [
        "## 기본 정보\n" + "\n".join([f"{data.iloc[i, 0].replace('기본 정보-', '- ')}: {data.iloc[i, 1]}" for i in range(9)]),
        "## 주호소문제\n" + f"### 주호소문제\n{data.iloc[9, 1]}\n### 스트레스 요인 및 상황\n{data.iloc[10, 1]}",
        "## 생활 및 관계\n" + f"### 생활 패턴\n{data.iloc[11, 1]}\n### 직장/학교 생활\n{data.iloc[12, 1]}\n### 가족/친구 관계\n{data.iloc[13, 1]}",
        "## 이전 병력 사항\n" + f"### 정신과 이용 경험\n{data.iloc[14, 1]}\n### 상담 서비스 이용 경험\n{data.iloc[15, 1]}\n### 심리평가 경험\n{data.iloc[16, 1]}",
        "## 증상 파악\n" + "\n".join([f"{data.iloc[i, 0].replace('증상 파악-', '- ')}: {data.iloc[i, 1]}" for i in range(17, 25)]),
        "## 기타/추가 사항\n" + str(data.iloc[25, 1])
    ]
    summary = "\n\n".join(summary_parts)

    # 대화 정보 파싱
    dialogue = ""
    if len(data) > 26:
        dialogue_lines = [f"{row[0].replace('상담-', '')}: {row[1]}" for _, row in data.iloc[26:].iterrows()]
        dialogue = "\n".join(dialogue_lines)

    return summary, dialogue


def construct_output(input_path: str, generated_dialogue: str) -> pd.DataFrame | None:
    input_summary = pd.read_csv(input_path, header=None)

    for turn in generated_dialogue.split("\n"):
        if len(turn) != 0:
            try:
                role, content = turn.split(": ")[0], turn.split(": ")[1]
            except:
                continue
            if "상담자" in role:
                new_row = pd.DataFrame([["상담-상담자", content]])
                input_summary = pd.concat([input_summary, new_row], ignore_index=True)
            elif "내담자" in role:
                new_row = pd.DataFrame([["상담-내담자", content]])
                input_summary = pd.concat([input_summary, new_row], ignore_index=True)
    if len(input_summary) == 0:
        print("Error: No dialogue generated.")
        return -1
    else:
        return input_summary


def main(args: argparse.Namespace) -> None:
    summaries, dialogues = [], []
    if args.shot_mode > 0:
        example_path_list = sorted([path for path in os.listdir(args.example_path) if path.endswith(".csv")])
        for example_path in example_path_list:
            s, d = load_example(os.path.join(args.example_path, example_path))
            summaries.append(s)
            dialogues.append(d)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load prompt
    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    # Augment data
    input_list = sorted([path for path in os.listdir(args.input_path) if path.endswith(".csv")]) # Only use .csv files
    
    for filename in tqdm(input_list, total=len(input_list), desc="입력 파일 처리 중"):
        current_input_path = os.path.join(args.input_path, filename)
        input_summary, _ = load_example(current_input_path)
        
        tasks = []
        if args.shot_mode == 0:
            api_input = construct_prompt(prompt, [], [], input_summary)
            tasks.append(api_input)
        elif args.shot_mode == 1:
            for i in range(len(summaries)):
                api_input = construct_prompt(prompt, [summaries[i]], [dialogues[i]], input_summary)
                tasks.append(api_input)
        elif args.shot_mode == 2:
            combinations_list = list(combinations(range(len(summaries)), 2))
            for p in combinations_list:
                i, j = p[0], p[1]
                api_input = construct_prompt(prompt, [summaries[i], summaries[j]], [dialogues[i], dialogues[j]], input_summary)
                tasks.append(api_input)
        
        for api_input in tqdm(tasks, desc=f"{filename} 대화 생성 중", leave=False):
            try:
                response = client.chat.completions.create(
                    model=args.generation_model,
                    messages=api_input,
                    temperature=args.generation_temperature,
                    top_p=args.generation_top_p,
                    max_tokens=args.generation_max_tokens,
                    response_format={"type": "text"},
                )
                generated_dialogue = response.choices[0].message.content.replace('**', '').replace(r'\n\n', r'\n')
                output_df = construct_output(current_input_path, generated_dialogue)
                
                output_index = len([name for name in os.listdir(args.output_path) if name.startswith("output_") and name.endswith(".csv")]) + 1
                output_filename = f"output_{output_index}.csv"

                output_df.to_csv(os.path.join(args.output_path, output_filename), header=False, index=False, encoding='utf-8-sig')
            
            except Exception as e:
                print(f"Error: {e}")
                continue
            
    print("Finished dialogue data generation")
    # Save setup
    setup = {
        "example_path": args.example_path,
        "input_path": args.input_path,
        "output_path": args.output_path,
        "prompt_path": args.prompt_path,
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

    # 경로 관련
    parser.add_argument("--example_path", type=str, default="./example/", help="Path to example data")
    parser.add_argument("--input_path", type=str, default="./input/", help="Path to input data")
    parser.add_argument("--output_path", type=str, default="./output/", help="Path to output data")
    parser.add_argument("--prompt_path", type=str, default="./example/prompt_1.txt", help="Path to prompt data")
    
    # 생성 모델 관련
    parser.add_argument("--generation_model", type=str, default="gpt-4o", help="Generation model for OpenAI API")
    parser.add_argument("--generation_temperature", type=float, default=1.0, help="Generation temperature for OpenAI API")
    parser.add_argument("--generation_top_p", type=float, default=1.0, help="Generation top p for OpenAI API")
    parser.add_argument("--generation_max_tokens", type=int, default=2048, help="Generation max tokens for OpenAI API")
    
    # 실행 모드 관련
    parser.add_argument(
        "--shot_mode", 
        type=int, 
        default=1, 
        choices=[0, 1, 2], 
        help="Set the shot mode: 0 for zero-shot, 1 for one-shot, 2 for few-shot (2-shot combinations)."
    )

    args = parser.parse_args()
    main(args)

