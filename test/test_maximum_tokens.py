import pytest
from config import Model
from llm_chat import LLMChat


def test_maximum_tokens_argument():
    prompt = "Test max tokens."
    for model in Model:
        chat = LLMChat(model=model, verbose=False)
        try:
            # This will use MAXIMUM_OUTPUT_TOKENS as set in llm_chat.py
            response = chat.prompt_chat(prompt)
            assert isinstance(response, str) or response is None
            print(f"Model {model.name} handled max tokens without error.")

        except Exception as e:
            pytest.fail(f"Model {model.name} failed with error: {e}")


if __name__ == "__main__":
    test_maximum_tokens_argument()