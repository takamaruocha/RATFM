import os
from loguru import logger
from openai import OpenAI
import yaml
import random


credentials = yaml.safe_load(open("credentials.yml"))


def openai_client(
    model,
    api_key=None,
    base_url="https://api.openai.com/v1"
):
    if api_key is None:
        assert model in credentials, f"Model {model} not found in credentials"
        # Randomly select an API key if multiple are provided
        if "round-robin" in credentials[model]:
            num_keys = len(credentials[model]["round-robin"])
            rand_idx = random.randint(0, num_keys - 1)
            credential = credentials[model]["round-robin"][rand_idx]
        else:
            credential = credentials[model]
        api_key = credential["api_key"]
        if "base_url" in credential:
            base_url = credential["base_url"]
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    logger.debug(
        f"API key: ****{api_key[-4:]}, endpoint: {base_url}"
    )
    
    return client


def send_openai_request(
    openai_request,
    model,
    api_key=None,
    base_url="https://api.openai.com/v1"
):
    if "gemini" in model:
        return send_gemini_request(
            convert_openai_to_gemini(openai_request),
            model,
            api_key=api_key
        )
    client = openai_client(model, api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model, **openai_request
    )

    message = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    return message, prompt_tokens, completion_tokens

    # return response.choices[0].message.content
