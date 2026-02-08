
from typing import List
import re

from config import (
    INITIAL_TEXT_TOKENS,
    Model,
    TOKENS_PER_WORD
)


def list_models():
    print("Available models:")
    for i, model in enumerate(Model, 1):
        print(f"{i}. {model.value}")


def get_first_tokens_from_text(
    text: str,
    num_tokens: int = INITIAL_TEXT_TOKENS
) -> str:

    num_words = int(num_tokens // TOKENS_PER_WORD)

    text_words = text.strip().split()
    if len(text_words) > num_words:
        text_words = text_words[:num_words]
    return " ".join(text_words)


def text_to_num_tokens(text: str) -> int:
    words = text.strip().split()
    return int(len(words) * TOKENS_PER_WORD)


def num_tokens_to_num_words(tokens: int) -> int:
    return int(tokens // TOKENS_PER_WORD)


def text_to_words(text: str) -> List[str]:
    # text = re.sub(r'[^\w\s]', ' ', text)
    words = text.strip().split()
    return words
