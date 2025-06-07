from konlpy.tag import Okt
import torch
import re
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_noun_entities(text: str) -> List[str]:
    okt = Okt()
    nouns = okt.nouns(text)
    return list(set([n for n in nouns if len(n) > 1]))  # 1글자 제외

def filter_entities_with_passages(entities: List[str], passages: List[str]) -> List[str]:
    joined = " ".join(passages)
    return [e for e in entities if e in joined]

def mask_entities_in_query(query: str, entities: List[str]) -> Tuple[str, List[str]]:
    masked_query = query
    masked_ents = []
    for ent in entities:
        if ent in masked_query:
            masked_query = masked_query.replace(ent, "<ent>", 1)
            masked_ents.append(ent)
    return masked_query, masked_ents

def get_kcomp_prompt(query, retrieved_passages):
    logger.info("Getting kcomp prompt for summary")

    entities = extract_noun_entities(query)
    filtered_entities = filter_entities_with_passages(entities, retrieved_passages)
    masked_query, masked_order = mask_entities_in_query(query, filtered_entities)
    combined_passages = "\n".join([f"{i+1}. {p}" for i, p in enumerate(retrieved_passages)])

    prompt = f"""당신은 의학 정보를 요약하는 도우미입니다.
다음 [마스킹된 질문]과 [관련 지식]을 참고하여 아래 작업을 순서대로 수행하세요:
1단계. 질문 속 <ent>로 표시된 부분에 들어갈 의학 용어들을 순서대로 예측하세요.
2단계. 예측한 각 용어에 대해, 관련 지식을 바탕으로 간단하고 정확한 설명을 작성하세요.
3단계. 위에서 예측한 용어들과 설명을 바탕으로, 질문과 관련된 핵심 개념 중심의 요약문을 작성하세요. 질문을 반복하거나 정답을 직접 말하지 마세요. 핵심 지식 위주의 중립적인 요약만 작성하세요.

출력 형식은 다음과 같습니다:
[용어 예측]  
1. '용어1'  
2. '용어2'

[설명]  
'용어1': '설명1'
'용어2': '설명2'

[요약]  
'요약문'

---

[마스킹된 질문]  
{masked_query}

[관련 지식]  
{combined_passages}

[용어 예측]
"""
    logger.info(f"kcomp prompt: {prompt}")
    return prompt
    
def parse_kcomp_output(text: str) -> Tuple[List[str], Dict[str, str], str]:

    text = '[용어 예측]\n'+text
    entity_section = re.search(r"\[용어\s*예측\](.*?)\[설명\]", text, re.DOTALL)
    entity_lines = entity_section.group(1).strip().splitlines() if entity_section else []
    entities = [re.sub(r"^\d+\.\s*", "", line).strip() for line in entity_lines if line.strip()]

    desc_section = re.search(r"\[설명\](.*?)\[요약\]", text, re.DOTALL)
    desc_text = desc_section.group(1).strip() if desc_section else ""
    desc_chunks = re.split(r"<eod>", desc_text)
    
    descriptions = {}
    for chunk in desc_chunks:
        line = chunk.strip()
        if not line:
            continue
        if ":" in line:
            ent, desc = line.split(":", 1)
            descriptions[ent.strip()] = desc.strip()

    summary_section = re.search(r"\[요약\](.*)", text, re.DOTALL)
    summary = summary_section.group(1).strip() if summary_section else ""

    return entities, descriptions, summary



def run_kcomp(ret_results, model, tokenizer, device):
    logger.info("Starting kcomp pipeline")

    results = []

    for ret_result in tqdm(ret_results, desc="generating..."):
        query = ret_result['query']
        retrieved_passages = ret_result['ret_results']
        prompt = get_kcomp_prompt(query, retrieved_passages)

        logger.info(f"Generating summary and etc")
        
        success = False
        for attempt in range(100):
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )

                generated = outputs[0][inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

                logger.info(f"Generation (attempt {attempt+1}): {generated_text}")

                entities, descriptions, summary = parse_kcomp_output(generated_text)

                ret_result.update({
                    "masked_prompt": prompt,
                    "kcomp_llm_generation": generated_text,
                    "kcomp_entities": entities,
                    "kcomp_descriptions": descriptions,
                    "kcomp_summary": summary
                })

                success = True
                break

            except Exception as e:
                logger.warning(f"Generation failed on attempt {attempt+1}: {e}")

        if not success:
            ret_result.update({
                "masked_prompt": prompt,
                "kcomp_llm_generation": "",
                "kcomp_entities": [],
                "kcomp_descriptions": [],
                "kcomp_summary": "[KCOMP GENERATION FAILED]"
            })

        results.append(ret_result)

    logger.info(f"Completed pipeline")

    return results
