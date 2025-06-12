# Self-Consistency Test Script 사용법

이 저장소는 LLM(대형 언어 모델) 기반의 self-consistency(자기 일관성) 평가를 위한 스크립트(`self_consistency_test.py`, `self_consistency_test.sh`)를 제공합니다.  
객관식(MCQA) 및 자유서술형(OpenQA) 문제에 대해 다수의 샘플을 생성하고, 투표 또는 확률 기반 방식으로 최종 답변을 산출할 수 있습니다.

---

## 1. 주요 파일 설명

- **self_consistency_test.py**  
  Python 기반 self-consistency 평가 스크립트 (직접 실행 가능)
- **self_consistency_test.sh**  
  실행 예시와 옵션 설명이 포함된 bash 스크립트 (예시 실행용)
- **README.md**  
  사용법 및 옵션 설명 문서

---

## 2. 주요 옵션 및 입력 설명

| 옵션명                | 필수 | 설명                                                                                 | 예시                                                         |
|----------------------|------|-------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `--model_name`       | O    | 사용할 HuggingFace 모델 이름 또는 로컬 경로                                          | `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`                    |
| `--cache_dir`        | X    | 모델 캐시 디렉토리 (모델 다운로드 경로, 로컬 모델이면 빈 문자열)                     | `/home/project/rapa/cache`                                   |
| `--input_text`       | X    | 단일 입력 문장 (단일 테스트용)                                                      | `"당뇨병 환자의 혈당 조절에 가장 적합한 약물은 무엇입니까?"` |
| `--prompt`           | X    | 프롬프트 템플릿 (`{input}`이 `--input_text`로 치환됨)                               | `"[Question]: {input}\n[Options]:\n1. ...\n[Answer]:"`       |
| `--input_json`       | X    | 여러 샘플을 한 번에 테스트할 경우 사용할 JSON 파일                                   | `sample_questions.json`                                      |
| `--output_json`      | X    | 결과를 저장할 JSON 파일명                                                            | `result.json`                                                |
| `--use_self_consistency` | X | self-consistency(다중 샘플 생성 후 투표/가중합) 사용 여부                           | (옵션, 지정 시 활성화)                                       |
| `--num_samples`      | X    | self-consistency 시 생성할 샘플 개수 (기본 5)                                        | `5`                                                          |
| `--temperature`      | X    | 샘플링 온도 (기본 0.7, 다양성 조절)                                                 | `0.7`                                                        |
| `--max_tokens`       | X    | 생성할 최대 토큰 수                                                                 | `256`                                                        |
| `--mode`             | X    | `"mcqa"`(객관식, 1~5 중 답변) 또는 `"openqa"`(자유서술형)                           | `mcqa` 또는 `openqa`                                         |

---

## 3. 실행 예시

### (1) 단일 입력 테스트 (MCQA: 객관식)

```bash
python self_consistency_test.py \
    --model_name "yanolja/EEVE-Korean-Instruct-10.8B-v1.0" \
    --cache_dir "/home/project/rapa/cache" \
    --input_text "당뇨병 환자의 혈당 조절에 가장 적합한 약물은 무엇입니까?" \
    --prompt "[Question]: {input}\n[Options]:\n1. 약A\n2. 약B\n3. 약C\n4. 약D\n5. 약E\n\n[Answer]:" \
    --use_self_consistency \
    --num_samples 5 \
    --mode mcqa \
    --output_json "./self_consistency_result.json"
```

- **MCQA 모드**: 모델이 1~5 중 하나의 숫자로 답변. 다수결(majority voting)로 최종 답변 산출.
- **프롬프트 예시**:  
  ```
  [Question]: 당뇨병 환자의 혈당 조절에 가장 적합한 약물은 무엇입니까?
  [Options]:
  1. 약A
  2. 약B
  3. 약C
  4. 약D
  5. 약E

  [Answer]:
  ```

---

### (2) 단일 입력 테스트 (OpenQA: 자유서술형)

```bash
python self_consistency_test.py \
    --model_name "yanolja/EEVE-Korean-Instruct-10.8B-v1.0" \
    --cache_dir "/home/project/rapa/cache" \
    --input_text "당뇨병 환자의 혈당 조절에 가장 적합한 약물은 무엇입니까?" \
    --prompt "질문: {input}\n답변:" \
    --use_self_consistency \
    --num_samples 5 \
    --mode openqa \
    --output_json "./self_consistency_result_openqa.json"
```

- **OpenQA 모드**: 모델이 자유롭게 텍스트로 답변. 각 샘플의 확률(logprob) 기반 가중합으로 최종 답변 산출.
- **프롬프트 예시**:  
  ```
  질문: 당뇨병 환자의 혈당 조절에 가장 적합한 약물은 무엇입니까?
  답변:
  ```

---

### (3) 여러 샘플을 한 번에 테스트 (JSON 입력)

1. **입력 JSON 파일 예시 (`sample_questions.json`)**

```json
[
  {
    "input": "고혈압 환자에게 가장 먼저 권장되는 치료법은 무엇입니까?",
    "prompt": "[Question]: {input}\n[Options]:\n1. 식이요법\n2. 운동\n3. 약물치료\n4. 수술\n5. 기타\n\n[Answer]:"
  },
  {
    "input": "감기 증상 완화에 도움이 되는 방법은?",
    "prompt": "[Question]: {input}\n[Options]:\n1. 휴식\n2. 수분 섭취\n3. 약 복용\n4. 운동\n5. 기타\n\n[Answer]:"
  }
]
```

2. **실행 예시**

```bash
python self_consistency_test.py \
    --model_name "yanolja/EEVE-Korean-Instruct-10.8B-v1.0" \
    --cache_dir "/home/project/rapa/cache" \
    --input_json "sample_questions.json" \
    --use_self_consistency \
    --num_samples 5 \
    --mode mcqa \
    --output_json "self_consistency_results.json"
```

---

## 4. 입력 방식 정리

- **단일 입력**:  
  `--input_text`와 `--prompt`를 함께 사용  
  프롬프트 내 `{input}`이 입력 문장으로 치환됨

- **여러 입력(JSON)**:  
  `--input_json`에 JSON 파일 경로 지정  
  각 샘플은 `{"input": "...", "prompt": "..."}` 형태의 객체로 구성  
  프롬프트 내 `{input}`이 해당 샘플의 입력으로 치환됨

---

## 5. 출력 결과 예시

- **MCQA 모드 (`--mode mcqa`)**  
  ```json
  [
    {
      "input": "...",
      "prompt": "...",
      "all_responses": ["1", "2", "2", "2", "3"],
      "final_response": 2
    }
  ]
  ```

- **OpenQA 모드 (`--mode openqa`)**  
  ```json
  [
    {
      "input": "...",
      "prompt": "...",
      "all_responses": ["메트포르민", "메트포르민", "설포닐우레아", "메트포르민", "메트포르민"],
      "final_response": "메트포르민"
    }
  ]
  ```

---
