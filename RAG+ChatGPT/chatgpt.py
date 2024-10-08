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


def get_chatgpt_multiple_responses(ret_results, model, system_prompt, user_prompt, api_key):

    for i in tqdm(range(len(ret_results)), desc='getting ChatGPT response'):
        query = ret_results[i]['query']
        ret_text = ret_results[i]['ret_results']
        ret_text = '\n'.join(ret_text)
        input_prompt = '[관련 정보]\n' + ret_text + '[질문]\n' + query + '\n\n' + user_prompt
        ret_results[i]['chatgpt_input'] = input_prompt
        ret_results[i]['chatgpt_response'] = get_chatgpt_response(model, system_prompt, input_prompt, api_key)
        
    return ret_results