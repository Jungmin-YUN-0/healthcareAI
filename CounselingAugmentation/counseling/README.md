## Directory Layout
```bash
project/
├─ generate.bash
├─ augment_summary.py
├─ augment_dialogue.py
├─ .env
├─ data_sample/            # 예시 data sample
├─ output_summary/         # summary 저장
├─ output_dialogue/        # dialogue 저장
├─ prompt_summary.txt      # summary system prompt
└─ prompt_dialogue.txt     # dialogue system prompt
```

## Requirements
```bash
pip install openai pandas tqdm python-dotenv
```

## Environment
.env 파일에 OpenAI API 키 설정 필요

```bash
# .env
OPENAI_API_KEY=sk-********************************
```

## Usage
```bash
bash generate.sh
```

1) `aug_trg='summary'`
- 요약 데이터 생성
- output: ./output_summary/output_*.csv에 저장

2) `aug_trg='dialogue'`
- 대화 데이터 생성
- input: ./output_summary의 CSV
- output: ./output_dialogue/output_*.csv에 저장

## Arguments

### augment_summary.py

- `--example_path`: 예시 데이터 경로

- `--output_path`: 생성된 데이터 저장 경로

- `--prompt_path`: 시스템 프롬프트 파일 경로

- `--shot_mode`: 샷 모드 (default: 1)
    - 0: zero-shot
    - 1: one-shot (예시 각 파일로 별도 호출)
    - 2: few-shot (예시 2개 조합)

### augment_dialogue.py
- `--example_path`: 예시 데이터 경로

- `--input_path`: augment_summary.py 결과 디렉토리

- `--output_path`: 생성된 데이터 저장 경로

- `--prompt_path`: 시스템 프롬프트 파일 경로

- `--shot_mode`: 샷 모드 (default: 1)
    - 0: zero-shot
    - 1: one-shot (예시 각 파일로 별도 호출)
    - 2: few-shot (예시 2개 조합)
