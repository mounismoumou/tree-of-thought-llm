import os
import openai
import backoff

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import backoff

completion_tokens = prompt_tokens = 0

"""---------------------------------------------------------------GPT models---------------------------------------------------------"""
api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "mistral":
        cost = 0
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

"""---------------------------------------------------------------Mistral models---------------------------------------------------------"""
mistral_model_name = "mistralai/Mistral-7B-v0.1"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_name)
mistral_model = mistral_model.to("mps")

def mistral(prompt, temperature=0.7, max_tokens=1000, n=1) -> list:
    messages = [{"role": "user", "content": prompt}]
    return lechat(messages, temperature=temperature, max_tokens=max_tokens, n=n)

def lechat(messages, temperature=0.7, max_tokens=1000, n=1) -> list:
    global completion_tokens, prompt_tokens
    inputs = mistral_tokenizer(["".join([m["content"] for m in messages])], return_tensors="pt").to("mps")
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        with torch.no_grad():
            outputs = mistral_model.generate(
                inputs["input_ids"],
                pad_token_id=mistral_tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
                num_return_sequences=cnt,
            )
            print(outputs)
        outputs = [mistral_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        # log completion tokens
        completion_tokens += len(outputs[0].split())
        prompt_tokens += len(" ".join([m["content"] for m in messages]).split())
    return outputs
