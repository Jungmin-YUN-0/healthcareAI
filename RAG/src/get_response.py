import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

def get_chatgpt_response(model, system_prompt, user_prompt, api_key): 
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model, 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    return response['choices'][0]['message']['content']

### RAG
def get_RAG_chatgpt_multiple_responses(ret_results, model, system_prompt, user_prompt, api_key):
    final_ret_results = []
    for i in tqdm(range(len(ret_results)), desc='getting ChatGPT response'):
        query = ret_results[i]['query']
        ret_text = [r for r in ret_results[i]['ret_results']]
        ret_text = '\n'.join(ret_text)
        input_prompt = '관련 의학 지식:\n' + ret_text + '\n\n' +'질문: ' + query + '\n\n' + user_prompt
        ret_results[i]['chatgpt_input'] = input_prompt
        ret_results[i]['chatgpt_response'] = get_chatgpt_response(model, system_prompt, input_prompt, api_key)
        final_ret_results.append(ret_results[i])
        
    return final_ret_results

### Using Chatgpt ONLY
def get_chatgpt_multiple_responses(queries, model, system_prompt, user_prompt, api_key):
    output = []
    for i in tqdm(range(len(queries)), desc='getting ChatGPT response'):
        query = queries[i]
        input_prompt = '질문: ' + query + '\n\n' + user_prompt
        output.append({'query':query,
                                  'chatgpt_input':input_prompt,
                                  'chatgpt_response':get_chatgpt_response(model, system_prompt, input_prompt, api_key)})
    return output


def get_openllm_generations(ret_results, model, tokenizer, generation_mode='default'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    final_ret_results = []
    
    for i in tqdm(range(len(ret_results)), desc='generating OpenLLM response'):
        print(ret_results)

        query = ret_results[i]['query']

        if generation_mode == 'default':
            ret_text = [r for r in ret_results[i]['ret_results']]
            if len(ret_text) >= 1:
                ret_text = '\n'.join(ret_text)
        else:
            query = ret_results[i]['query']
            ret_text = ret_results[i]['kcomp_summary']
        
        input_prompt = f'''[관련 의학 지식]을 참고하여 [질문]에 답하세요. [답변]은 한번만 생성하세요.
        [관련 의학 지식]
        {ret_text}

        [질문]
        {query}

        [답변]
        '''
        
        inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        result = ret_results[i].copy()
        result['final_input'] = input_prompt
        result['final_response'] = generated_text.strip()
        final_ret_results.append(result)

        print(generated_text.strip())

    return final_ret_results





    