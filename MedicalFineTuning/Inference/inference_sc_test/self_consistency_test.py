import argparse
import json
import os
import re
from vllm import LLM, SamplingParams
from collections import Counter
import torch._dynamo

def apply_majority_voting(responses, num_samples):
    """Apply majority voting by selecting the most frequent answer (for MCQA style)"""
    if num_samples <= 1:
        if responses:
            match = re.search(r'\b[1-5]\b', responses[0])
            return int(match.group()) if match else responses[0]
        return ""
    extracted_numbers = []
    for response in responses:
        match = re.search(r'\b[1-5]\b', response)
        if match:
            extracted_numbers.append(int(match.group()))
        else:
            extracted_numbers.append(response)
    if extracted_numbers:
        counter = Counter(extracted_numbers)
        most_common = counter.most_common(1)[0][0]
        return most_common
    return responses[0] if responses else ""

def normalized_weighted_sum(responses_with_logprobs):
    """Calculate normalized weighted sum based on log probabilities (for open QA)"""
    import math
    answer_scores = {}
    for response_data in responses_with_logprobs:
        response_text = response_data['text']
        cumulative_logprob = response_data['cumulative_logprob']
        token_count = response_data['token_count']
        if token_count == 0:
            continue
        normalized_log_prob = cumulative_logprob / token_count
        normalized_prob = math.exp(normalized_log_prob)
        answer_scores[response_text] = answer_scores.get(response_text, 0) + normalized_prob
    if not answer_scores:
        return responses_with_logprobs[0]['text'] if responses_with_logprobs else ""
    return max(answer_scores.items(), key=lambda x: x[1])[0]

def run_self_consistency(
    llm, prompt, sampling_params, num_samples=5, mode="mcqa"
):
    """Generate multiple responses and apply self-consistency"""
    request_outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    if mode == "mcqa":
        responses = [output.text.strip() for output in request_outputs[0].outputs]
        final_response = apply_majority_voting(responses, num_samples)
        return {"all_responses": responses, "final_response": final_response}
    else:
        responses_with_logprobs = []
        for output in request_outputs[0].outputs:
            token_count = len(output.logprobs) if output.logprobs else 0
            responses_with_logprobs.append({
                "text": output.text.strip(),
                "cumulative_logprob": output.cumulative_logprob,
                "token_count": token_count
            })
        final_response = normalized_weighted_sum(responses_with_logprobs)
        return {
            "all_responses": [r["text"] for r in responses_with_logprobs],
            "final_response": final_response
        }

def main(args):
    torch._dynamo.config.suppress_errors = True

    if args.cache_dir:
        llm = LLM(
            model=args.model_name,
            download_dir=args.cache_dir,
            trust_remote_code=True,
            dtype='bfloat16',
        )
    else:
        llm = LLM(
            model=args.model_name,
            trust_remote_code=True,
            dtype='bfloat16',
        )

    # Decide mode: MCQA(1~5) or open QA
    mode = args.mode

    # Sampling params
    if args.use_self_consistency:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=50,
            top_p=0.9,
            max_tokens=args.max_tokens,
            n=args.num_samples,
            logprobs=1 if mode == "openqa" else None,
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
        )

    results = []
    # Single input mode
    if args.input_text and args.prompt:
        prompt = args.prompt.replace("{input}", args.input_text)
        print(f"\n[Prompt]\n{prompt}\n")
        if args.use_self_consistency:
            result = run_self_consistency(llm, prompt, sampling_params, args.num_samples, mode)
            print(f"\n[Final Self-Consistent Response]: {result['final_response']}")
            results.append({
                "input": args.input_text,
                "prompt": prompt,
                "all_responses": result["all_responses"],
                "final_response": result["final_response"]
            })
        else:
            request_outputs = llm.generate([prompt], sampling_params)
            response = request_outputs[0].outputs[0].text.strip()
            print(f"[Response]: {response}")
            results.append({
                "input": args.input_text,
                "prompt": prompt,
                "response": response
            })
        # 결과 저장 (output_json 옵션이 있을 때)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output_json}")
    # Batch mode from JSON
    elif args.input_json:
        with open(args.input_json, encoding="utf-8") as f:
            samples = json.load(f)
        for sample in samples:
            input_text = sample["input"]
            prompt = sample["prompt"].replace("{input}", input_text)
            print(f"\n[Prompt]\n{prompt}\n")
            if args.use_self_consistency:
                result = run_self_consistency(llm, prompt, sampling_params, args.num_samples, mode)
                print(f"\n[Final Self-Consistent Response]: {result['final_response']}")
                results.append({
                    "input": input_text,
                    "prompt": prompt,
                    "all_responses": result["all_responses"],
                    "final_response": result["final_response"]
                })
            else:
                request_outputs = llm.generate([prompt], sampling_params)
                response = request_outputs[0].outputs[0].text.strip()
                print(f"[Response]: {response}")
                results.append({
                    "input": input_text,
                    "prompt": prompt,
                    "response": response
                })
        # Save all results
        output_path = args.output_json or "self_consistency_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("Either --input_text/--prompt or --input_json must be provided.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-consistency test script")
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--cache_dir', type=str, default="", help='Model cache directory')
    parser.add_argument('--input_text', type=str, default="", help='Input text for single test')
    parser.add_argument('--prompt', type=str, default="", help='Prompt template (use {input} for input text)')
    parser.add_argument('--input_json', type=str, default="", help='JSON file with [{"input":..., "prompt":...}, ...]')
    parser.add_argument('--output_json', type=str, default="", help='Output JSON file for results')
    parser.add_argument('--use_self_consistency', action='store_true', help='Enable self-consistency')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples for self-consistency')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=256, help='Max tokens to generate')
    parser.add_argument('--mode', type=str, choices=["mcqa", "openqa"], default="mcqa", help='Self-consistency mode: mcqa or openqa')
    args = parser.parse_args()
    main(args)
