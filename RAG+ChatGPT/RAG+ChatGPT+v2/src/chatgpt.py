import openai
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
        ret_text = [r['answer'] for r in ret_results[i]['ret_results']]
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


    