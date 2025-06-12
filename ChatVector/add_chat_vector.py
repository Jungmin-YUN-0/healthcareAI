import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

import logging
logger = logging.getLogger(__name__)

def add_chat_vector(
    base: PreTrainedModel,
    chat_vector_path: str,
    ratio: float,
    skip_embed: bool = False,
    special_tokens_map=None,
):
    chat_vector = torch.load(f"{chat_vector_path}/pytorch_model.bin")
    cv = chat_vector["chat_vector"]

    for n, p in base.named_parameters():
        if n not in cv:
            logger.warning(f"Skip '{n}' – not found in chat_vector")
            continue

        tgt = cv[n]

        if p.data.shape != tgt.shape:
            logger.warning(
                f"Skip '{n}' – shape mismatch: base {p.data.shape} vs chat {tgt.shape}"
            )
            continue

        p.data += ratio * tgt

    return base, chat_vector["cfg"]


def main(
    base_model_path: str,
    chat_vector_path,
    output_path: str,
    cache_dir,
    ratio,
    skip_embed: bool = False,
    special_tokens_map=None,
):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype='auto', cache_dir=cache_dir)

    if special_tokens_map:
        for k, v in special_tokens_map.items():
            base_model.get_input_embeddings().weight.data[k] = torch.zeros(
                base_model.config.hidden_size
            )
            base_model.get_output_embeddings().weight.data[k] = torch.zeros(
                base_model.config.hidden_size
            )

    for cv_path, r in zip(chat_vector_path, ratio):
        base_model, cfg = add_chat_vector(
            base_model, cv_path, r, skip_embed, special_tokens_map)

    # set tokenizer
    # last chat_vector_path as chat_template.
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=cache_dir)
    chat_tokenizer = AutoTokenizer.from_pretrained(cfg['chat_model_path'])

    if chat_tokenizer.chat_template is None:
        logger.warning('chat_tokenizer.chat_template is None')
    else:
        logger.info(f'chat_template: {tokenizer.chat_template}')
        tokenizer.chat_template = chat_tokenizer.chat_template
        tokenizer.eos_token = chat_tokenizer.eos_token
        tokenizer.eos_token_id = chat_tokenizer.eos_token_id

    base_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size='8GB'
    )

    tokenizer.save_pretrained(output_path)
    

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
