# Licensed under the MIT license.

import torch
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def load_HF_model(ckpt) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def generate_with_HF_model(
    tokenizer, 
    model, 
    input=None, 
    temperature=0.8, 
    top_p=0.95, 
    top_k=40, 
    repetition_penalty=1.1,
    num_beams=1, 
    max_new_tokens=256, 
    num_return=1,
    stop=["\n",],
    **kwargs
):

    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    prompt_length = input_ids.shape[-1]
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            tokenizer=tokenizer,
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return,
            stop_strings=stop,
            repetition_penalty=repetition_penalty,
        )
    return generation_output.sequences[:, prompt_length:]

def batch_generate_with_HF_model(
    tokenizer, 
    model, 
    input=None, 
    temperature=0.8, 
    top_p=0.95, 
    top_k=40, 
    repetition_penalty=1.1,
    num_beams=1, 
    max_new_tokens=256, 
    num_return=1,
    stop=["\n",],
    **kwargs
):
    outputs = [
        generate_with_HF_model(
            tokenizer=tokenizer,
            model=model,
            input=i,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            num_return=num_return,
            stop=stop,
            repetition_penalty=repetition_penalty,
            **kwargs,
        ) for i in input
    ]

    return outputs, sum([len(o) for o in outputs])