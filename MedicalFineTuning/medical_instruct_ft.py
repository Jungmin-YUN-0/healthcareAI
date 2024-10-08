import pandas as pd
import argparse
import torch
import datasets
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import notebook_login

tokenizer = AutoTokenizer.from_pretrained('../ChatVector/ckpt/Llama-3-8B-OpenBioLLM-Korean')

### build instruction-prompt
def generate_prompt(ds):
    prompt_lst = []
    for i in range(len(ds)):
        system = ds['system_prompt'][i]
        task = ds['task_prompt'][i]

        messages = [
            {"role": 'system', 'content': system},
            {"role": 'user', 'content': task.replace('<QUESTION>', ds['question'][i])},
            {"role": 'assistant', 'content': ds['answer'][i]}
        ]

        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_lst.append(text)

    return prompt_lst

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### model and dataset
    #parser.add_argument('--model_type', type=str, default='gemma-2b')
    parser.add_argument('--base_model', type=str, default='../ChatVector/ckpt/Llama-3-8B-OpenBioLLM-Korean')
    parser.add_argument('--cache_dir', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./KoAlpaca_v1.1_medical.jsonl')
    parser.add_argument('--test_ratio', type=float, default=0.1)

    ### PEFT
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--task_type', type=str, default='CAUSAL_LM', choices=['CAUSAL_LM','QUESTION_ANS'])
    parser.add_argument('--applyQLoRA', type=str, default='True', choices=['True','False'])

    ### instruction-tuning
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=2e-4)

    ### etc
    parser.add_argument('--hf_login', type=str, default='False', choices=['True','False'])
    parser.add_argument('--isTest', type=str, default='True', choices=['True','False'])
    args = parser.parse_args()


    if args.hf_login == 'True':
        notebook_login()

    ### construct dataset
    df = pd.read_json(args.data_path, lines=True)
    test_df = df[:int(len(df)*args.test_ratio)]
    train_df = df[int(len(df)*args.test_ratio):]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)
    ds['test'] = Dataset.from_pandas(test_df)
    print(ds['train'])

    ### set lora config
    lora_config = LoraConfig(r=args.r,
                             lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout,
                             target_modules=['q_proj','o_proj','k_proj','v_proj'],
                             task_type=args.task_type)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_compute_dtype=torch.float16)

    if args.applyQLoRA == 'True':
        model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, cache_dir=args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    tokenizer.padding_side = 'right'


    ### instruction-tuning
    trainer = SFTTrainer(model=model,
                         train_dataset=ds['train'],
                         max_seq_length=min(tokenizer.model_max_length, args.max_seq_length),
                         args=TrainingArguments(output_dir=args.output_dir,
                                                num_train_epochs=5,
                                                per_device_train_batch_size=1,
                                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                optim='paged_adamw_8bit',
                                                warmup_steps=args.warmup_steps,
                                                learning_rate=args.learning_rate,
                                                fp16=True,
                                                logging_steps=100,
                                                push_to_hub=False,
                                                report_to='none',),
                         peft_config=lora_config,
                         formatting_func=generate_prompt)
    trainer.train()



    ### save LoRA weight
    adapter_model = 'lora_adapter'
    trainer.model.save_pretrained(adapter_model)

    ### construct fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, cache_dir=args.cache_dir)
    model = PeftModel.from_pretrained(model, adapter_model, torch_dtype=torch.float16)
    model = model.merge_and_unload()
    model.save_pretrained(f'{args.base_model}-InstructFT')

    # Save tokenizer
    tokenizer.save_pretrained(f'{args.base_model}-InstructFT')
