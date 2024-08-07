import os
import os
import time
from tqdm import tqdm
import concurrent.futures
from openai import AzureOpenAI

print("GPT35_KEY", os.environ.get('GPT35_KEY', ''))
client = AzureOpenAI()

max_threads = 32


def load_OpenAI_model(model):
    return None, model


def generate_with_OpenAI_model(
    prompt,
    model_ckpt="gpt-35-turbo",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
):
    messages = [{"role": "user", "content": prompt}]
    parameters = {
        "model": model_ckpt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "seed": 1,
    }

    ans, timeout = "", 5
    while not ans:
        try:
            time.sleep(timeout)
            completion = client.chat.completions.create(messages=messages, **parameters)
            ans = completion.choices[0].message.content

        except Exception as e:
            print(e)
        if not ans:
            timeout = timeout * 2
            if timeout > 120:
                timeout = 1
            try:
                print(f"Will retry after {timeout} seconds ...")
            except:
                pass
    return ans


def generate_n_with_OpenAI_model(
    prompt,
    n=1,
    model_ckpt="gpt-35-turbo",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
    max_threads=3,
    disable_tqdm=True,
):
    preds = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(generate_with_OpenAI_model, prompt, model_ckpt, max_tokens, temperature, top_k, top_p, stop) for _ in range(n)]
        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate', disable=disable_tqdm):
            ans = future.result()
            preds.append(ans)
    return preds


def score(
    model,
    sentences,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    n=1,
    max_tokens=512,
    logprobs=1,
    stop=["\n"],
    tokenizer=None,
):

    proba = []
    output_lst = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(generate_with_OpenAI_model, prompt) for prompt in sentences
        ]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            ans = future.result()
            output_lst.append(ans)
            if "Yes" in ans or "yes" in ans:
                proba.append(1)
            elif "No" in ans or "no" in ans:
                proba.append(0)
            else:
                proba.append(0.5)

    return proba, output_lst
