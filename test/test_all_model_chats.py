import pytest
from config import Model
from llm_chat import LLMChat


def test_all_models():
    prompt = "Hello, world!"
    for model in Model:
        print(f"Testing model: {model.name}")
        chat = LLMChat(model=model, verbose=False)
        try:
            response = chat.prompt_chat(prompt)
            assert isinstance(response, str)
            print(f"Model {model.name} responded successfully.")
        except Exception as e:
            pytest.fail(f"Model {model.name} failed with error: {e}")
