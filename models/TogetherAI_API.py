from typing import List, Dict
from together import Together, AsyncTogether
import time, json, os, requests, asyncio


keys = []       # Add your togetherAI API keys here
key_cnt = 0


def _get_key():
    global key_cnt
    key = keys[key_cnt]
    key_cnt = (key_cnt + 1) % len(keys)
    return key


def text_completion(prompt, model_ckpt, max_tokens=256, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1, stop=None):
    while True:
        api_key = _get_key()
        client = Together(api_key=api_key)
        try:
            response = client.completions.create(
                model=model_ckpt,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            break
        except:
            print(f"Together AI API failed at key: {api_key}. Retrying...")
            
    return response.choices[0].text


def text_completion_concurrent(prompt_list: list[str], model_ckpt, max_tokens=256, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1, stop=None):
    async def async_text_completion(prompts):
        api_key = _get_key()
        async_client = AsyncTogether(api_key=api_key)
        tasks = [
            async_client.completions.create(
                model=model_ckpt,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            for prompt in prompts
        ]
        responses = await asyncio.gather(*tasks)
        text_responses = [r.choices[0].text for r in responses]
        return text_responses

    text_responses = asyncio.run(async_text_completion(prompt_list))
    return text_responses


def chat_completion(prompt, model_ckpt, system_prompt: str = "You are a helpful AI assistant.", max_tokens=256, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1, stop=None):
    while True:
        api_key = _get_key()
        client = Together(api_key=api_key)
        try:
            response = client.chat.completions.create(
                model=model_ckpt,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            break
        except:
            print(f"Together AI API failed at key: {api_key}. Retrying...")
            time.sleep(1)

    return response.choices[0].message.content


def chat_completion_concurrent(prompt_list: list[str], model_ckpt, system_prompt: str = "You are a helpful AI assistant.", max_tokens=256, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1, stop=None):
    async def async_chat_completion(messages):
        api_key = _get_key()
        async_client = AsyncTogether(api_key=api_key)
        tasks = [
            async_client.chat.completions.create(
                model=model_ckpt,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            for message in messages
        ]
        responses = await asyncio.gather(*tasks)
        text_responses = [r.choices[0].message.content for r in responses]
        return text_responses

    text_responses = asyncio.run(async_chat_completion(prompt_list))
    return text_responses


def _test01():
    prompt = ["What are some fun things to do in New York", "What is your name?"]
    model_ckpt = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_output = chat_completion_concurrent(prompt, model_ckpt)
    print(model_output)
    

def _test02():
    prompt = ["What are some fun things to do in New York", "What is your name?"]
    model_ckpt = "mistralai/Mistral-7B-v0.1"
    model_output = text_completion_concurrent(prompt, model_ckpt)
    print(model_output)


if __name__ == "__main__":
    # _test01()
    _test02()
