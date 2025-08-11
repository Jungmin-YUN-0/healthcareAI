from __future__ import annotations

import os
import json
import argparse
import warnings
from pathlib import Path
from ast import literal_eval
from itertools import combinations
import openai
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv 

load_dotenv()
warnings.filterwarnings("ignore")

# API 클라이언트 설정
def setup_api_client() -> openai.OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    return openai.OpenAI(api_key=api_key)


def load_structured_summary(file_path: str) -> str:
    try:
        data = pd.read_csv(file_path, header=None, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, header=None, encoding='cp949')
    
    sections = {
        "기본 정보": "\n".join([f"- {data.iloc[i, 0].replace('기본 정보-', '')}: {data.iloc[i, 1]}" for i in range(9)]),
        "주호소문제": f"### 주호소문제\n{data.iloc[9, 1]}\n### 스트레스 요인 및 상황\n{data.iloc[10, 1]}",
        "생활 및 관계": f"### 생활 패턴\n{data.iloc[11, 1]}\n### 직장/학교 생활\n{data.iloc[12, 1]}\n### 가족/친구 관계\n{data.iloc[13, 1]}",
        "이전 병력 사항": f"### 정신과 이용 경험\n{data.iloc[14, 1]}\n### 상담 서비스 이용 경험\n{data.iloc[15, 1]}\n### 심리평가 경험\n{data.iloc[16, 1]}",
        "증상 파악": "\n".join([f"- {data.iloc[i, 0].replace('증상 파악-', '')}: {data.iloc[i, 1]}" for i in range(17, 25)]),
        "기타/추가 사항": data.iloc[25, 1]
    }

    summary_text = "\n\n".join([f"## {title}\n{content}" for title, content in sections.items()])
    return summary_text


def generate_and_save_data(client: openai.OpenAI, messages: list, args: argparse.Namespace) -> None:
    """OpenAI API 호출 -> 데이터 생성, CSV 파일로 결과 저장"""
    response = client.chat.completions.create(
        model=args.generation_model,
        messages=messages,
        temperature=args.generation_temperature,
        top_p=args.generation_top_p,
        max_tokens=args.generation_max_tokens,
    )
    
    content = response.choices[0].message.content
    try:
        data_dict = literal_eval(content)
        df_output = pd.DataFrame(data_dict.items(), columns=["key", "value"])
        
        output_dir = Path(args.output_path)
        output_index = len(list(output_dir.glob("output_*.csv"))) + 1
        output_file = output_dir / f"output_{output_index}.csv"
        df_output.to_csv(output_file, index=False, header=None, encoding="utf-8")
        
    except (ValueError, SyntaxError) as e:
        print(f"모델 응답 파싱 또는 파일 저장 중 오류 발생: {e}")
        print(f"오류 발생 응답: {content}")


def save_setup(args: argparse.Namespace) -> None:
    """실행에 사용된 설정들을 JSON 파일로 저장"""
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_file = output_path / "setup.json"
    
    with open(setup_file, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


def main(args: argparse.Namespace) -> None:
    client = setup_api_client()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    message_sets_to_process = []
    
    if args.shot_mode == 0:
        print("Running in Zero-shot mode...")
        messages = [{"role": "system", "content": prompt}]
        message_sets_to_process.append(messages)
        
    else: # 1-shot 또는 2-shot
        example_files = sorted(Path(args.example_path).glob("*.csv"))
        summaries = [load_structured_summary(str(f)) for f in example_files]
        
        if args.shot_mode == 1:
            print(f"Running in One-shot mode with {len(summaries)} examples...")
            for s in summaries:
                messages = [{"role": "system", "content": prompt}, {"role": "user", "content": s}]
                message_sets_to_process.append(messages)
        
        elif args.shot_mode == 2:
            print(f"Running in Few-shot mode with {len(summaries)} examples...")
            for i, j in combinations(range(len(summaries)), 2):
                messages = [{"role": "system", "content": prompt}, {"role": "user", "content": summaries[i]}, {"role": "user", "content": summaries[j]}]
                message_sets_to_process.append(messages)

    print(f"Total generations to perform: {len(message_sets_to_process)}")
    for messages in tqdm(message_sets_to_process, desc="Generating Data"):
        generate_and_save_data(client, messages, args)

    save_setup(args)
    print(f"Finished data generation. Outputs are saved in '{args.output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment data using OpenAI API")
    
    # 경로 관련
    parser.add_argument("--example_path", type=str, default="./example/", help="Path to example data CSVs.")
    parser.add_argument("--output_path", type=str, default="./output/", help="Path to save generated data.")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the system prompt file.")
    
    # 생성 모델 관련
    parser.add_argument("--generation_model", type=str, default="gpt-4o", help="OpenAI model for generation.")
    parser.add_argument("--generation_temperature", type=float, default=1.0, help="Generation temperature.")
    parser.add_argument("--generation_top_p", type=float, default=1.0, help="Generation top_p.")
    parser.add_argument("--generation_max_tokens", type=int, default=2048, help="Generation max_tokens.")
    
    # 실행 모드 관련
    parser.add_argument(
        "--shot_mode", 
        type=int, 
        default=1, 
        choices=[0, 1, 2], 
        help="0: zero-shot, 1: one-shot, 2: few-shot"
    )
    
    args = parser.parse_args()
    main(args)