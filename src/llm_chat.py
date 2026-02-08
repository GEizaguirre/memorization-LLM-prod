import os
import requests

from config import (
    MAXIMUM_OUTPUT_TOKENS,
    Model,
    Provider,
    MODEL_TO_PROVIDER,
    DEFAULT_TEMPERATURE
)
from auth import load_api_keys

load_api_keys()


def call_openai_compatible(
    model: Model,
    messages: list,
    base_url: str,
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:

    """
    Calls OpenAI-compatible chat completion endpoint.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    formatted_messages = []
    for msg in messages:
        if msg["role"] not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role: {msg['role']}")
        formatted_messages.append(msg)

    if (
        model == Model.GPT_5 or
        model == Model.KIMI_K2_5
    ):
        temperature = 1


    payload = {
        "model": model.value,
        "messages": formatted_messages
    }

    is_reasoning_model = any(
        m in model.value.lower() for m in ["o1", "o3", "gpt-5", "gpt-5.2"]
    )

    if is_reasoning_model:
        payload["max_completion_tokens"] = MAXIMUM_OUTPUT_TOKENS
        payload["reasoning_effort"] = "low"
    else:
        payload["max_tokens"] = MAXIMUM_OUTPUT_TOKENS
        payload["temperature"] = temperature

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload
    )

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print("OpenAI API error:", response.text)
        raise e

    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    return content


def call_anthropic(
    model: Model,
    messages: list,
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": model.value,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAXIMUM_OUTPUT_TOKENS
    }
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    data = response.json()
    return data.get("output", {}).get("text", "")


def call_google(
    model: Model,
    messages: list,
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:

    google_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        google_contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model.value}:generateContent?key={api_key}"
    )

    config = {
        "temperature": temperature,
        "maxOutputTokens": MAXIMUM_OUTPUT_TOKENS,
        "thinkingConfig": {}
    }

    payload = {
        "contents": google_contents,
        "generationConfig": config
    }

    if "gemini-3" in model.value.lower():
        config["thinkingConfig"]["thinkingLevel"] = "low"
    elif "flash" in model.value.lower():
        config["thinkingConfig"]["thinkingBudget"] = 0
    elif model.value.lower() == "gemini-2.5-pro":
        config["thinkingConfig"]["thinkingBudget"] = 128

    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()

    # Handle various response scenarios
    if "candidates" not in result or len(result["candidates"]) == 0:
        raise ValueError(f"No candidates in response: {result}")

    candidate = result["candidates"][0]

    # Check if content exists and has parts
    if "content" not in candidate:
        raise ValueError(f"No content in candidate: {candidate}")

    content = candidate["content"]

    # Handle missing parts (happens with MAX_TOKENS, SAFETY, etc.)
    if "parts" not in content or len(content["parts"]) == 0:
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        if finish_reason == "MAX_TOKENS":
            raise ValueError("Response incomplete: hit max tokens limit. Consider increasing maxOutputTokens.")
        elif finish_reason == "SAFETY":
            raise ValueError("Response blocked due to safety filters.")
        else:
            raise ValueError(f"No parts in content. Finish reason: {finish_reason}. Response: {result}")

    return content["parts"][0]["text"]


def get_completion(model: Model, messages: list):
    provider = MODEL_TO_PROVIDER.get(model)

    if provider == Provider.OPENAI:
        return call_openai_compatible(
            model,
            messages,
            "https://api.openai.com/v1",
            os.getenv("OPENAI_API_KEY")
        )
    elif provider == Provider.MOONSHOT:
        return call_openai_compatible(
            model,
            messages,
            "https://api.moonshot.ai/v1",
            os.getenv("MOONSHOT_API_KEY")
        )
    elif provider == Provider.DEEPSEEK:
        return call_openai_compatible(
            model,
            messages,
            "https://api.deepseek.com",
            os.getenv("DEEPSEEK_API_KEY")
        )
    elif provider == Provider.CLAUDE:
        return call_anthropic(
            model,
            messages,
            os.getenv("ANTHROPIC_API_KEY")
        )
    elif provider == Provider.GOOGLE:
        return call_google(
            model,
            messages,
            os.getenv("GOOGLE_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class LLMChat:

    def __init__(
        self,
        model: Model,
        verbose: bool = False
    ):
        self.model = model
        self.prompts = []
        self.responses = []
        self.verbose = verbose

    def prompt_chat(self, message: str):
        if self.verbose:
            print(f"Prompt: \033[94m{message}\n\033[0m")
        self.prompts.append(message)
        messages = [{"role": "user", "content": msg} for msg in self.prompts]
        response = get_completion(self.model, messages)
        if self.verbose:
            print(f"Response: \033[92m{response}\n\033[0m")
        self.responses.append(response)
        return response
