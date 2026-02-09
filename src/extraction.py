import json
import random
import string

from config import (
    INITIAL_TEXT_TOKENS,
    MAX_BEST_OF_N,
    PHASE1_SUCCESS_THRESHOLD
)
from long_form_metrics import near_verbatim_metrics
from metrics import normalized_similarity_score
from permutator import BoNPermutator
from utils import (
    get_first_tokens_from_text,
    text_to_num_tokens,
    text_to_words
)
from prompt import (
    INITIAL_INSTRUCTIONS,
    CONTINUATION_INSTRUCTIONS
)
from llm_chat import LLMChat


class Extractor:

    def __init__(
        self,
        model: str,
        reference_text: str,
        max_iterations: int = None,
        verbose: bool = False,
        log_path: str = None,
        best_of_n: bool = False
    ):
        self.model = model
        self.reference_text = reference_text
        self.max_tokens = text_to_num_tokens(reference_text)
        print(
            "--- Reference text tokens: %d ---" %
            self.max_tokens
        )
        self.max_iterations = max_iterations
        self.initial_text = get_first_tokens_from_text(
            reference_text,
            num_tokens=INITIAL_TEXT_TOKENS
        )
        self.initial_words = text_to_words(self.initial_text)
        prefix_length = len(self.initial_words) // 2
        self.prefix = self.initial_words[:prefix_length]
        self.prefix = " ".join(self.prefix)
        self.expected_suffix = self.initial_words[prefix_length:]
        self.expected_suffix = " ".join(self.expected_suffix)

        self.initial_prompt = INITIAL_INSTRUCTIONS + "\n\n" + self.prefix
        self.continuation_prompt = CONTINUATION_INSTRUCTIONS
        self.num_iterations = 0
        self.responses = []
        self.verbose = verbose
        self.response_tokens = 0
        if self.verbose:
            print(f"Initial text (first {INITIAL_TEXT_TOKENS} tokens):\n{self.initial_text}\n")
        self.phase1_successful = False
        self.best_of_n = best_of_n
        if self.best_of_n:
            self.permutator = BoNPermutator(INITIAL_INSTRUCTIONS)
        self.best_of_n_results = {}
        self.best_of_n_iters = 0

        if log_path is None:
            random_string = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=6)
            )
            self.log_path = f"extraction_log_{model}_{random_string}.json"
        else:
            self.log_path = log_path

    def phase1_best_of_n(self):

        for i in range(MAX_BEST_OF_N):
            print(f"--- Best-of-N iteration {i+1}/{MAX_BEST_OF_N} ---")
            self.chat = LLMChat(
                self.model,
                verbose=self.verbose
            )
            instructions = self.permutator.next()
            prompt = instructions + "\n\n" + self.prefix
            response = self.chat.prompt_chat(prompt)
            similarity_score = normalized_similarity_score(
                self.expected_suffix,
                response
            )
            if self.verbose:
                print(f"--- Similarity score: {similarity_score:.4f} ---")
            self.best_of_n_results[i] = {
                "prompt": prompt,
                "response": response,
                "similarity_score": similarity_score
            }
            if similarity_score >= PHASE1_SUCCESS_THRESHOLD:
                print(f"Found a successful prompt in best-of-n iteration {i+1}.")
                self.responses.append(response)
                self.response_tokens += text_to_num_tokens(response)
                return similarity_score

        best_similarity_score = max(
            result["similarity_score"] for result in self.best_of_n_results.values()
        )
        return best_similarity_score

    def phase1(self):
        self.chat = LLMChat(
            self.model,
            verbose=self.verbose
        )
        response = self.chat.prompt_chat(self.initial_prompt)
        self.responses.append(response)
        self.response_tokens += text_to_num_tokens(response)

        similarity_score = normalized_similarity_score(
            self.expected_suffix,
            response
        )

        if self.verbose:
            print(f"Phase 1 - Similarity score: {similarity_score:.4f}")

        return similarity_score

    def phase2(self):

        while True:
            print(
                "--- Tokens count so far: %d/%d ---" %
                (self.response_tokens, self.max_tokens)
            )
            response = self.chat.prompt_chat(self.continuation_prompt)
            self.responses.append(response)
            self.num_iterations += 1
            self.response_tokens += text_to_num_tokens(response)
            if self.max_iterations and self.num_iterations >= self.max_iterations:
                print(f"Reached maximum iterations ({self.max_iterations}), stopping extraction.")
                break
            if self.response_tokens >= self.max_tokens:
                print(f"Reached maximum token limit ({self.max_tokens}), stopping extraction.")
                break

    def extract(self):
        if self.best_of_n:
            self.phase1_similarity = self.phase1_best_of_n()
        else:
            self.phase1_similarity = self.phase1()
        if self.phase1_similarity >= PHASE1_SUCCESS_THRESHOLD:
            print("Phase 1 successful, proceeding to Phase 2.")
            self.phase1_successful = True
            self.phase2()
            nv_recall = near_verbatim_metrics(
                self.reference_text,
                " ".join(self.responses),
                lower=True
            )
            self.nv_recall_metrics = nv_recall.to_dict()
            print(f"Final near-verbatim recall: {nv_recall.nv_recall:.6f}")
        else:
            self.phase1_successful = False
            self.nv_recall_metrics = {}
            print("Phase 1 was not successful, cannot proceed to Phase 2.")

        data = {
            "model": self.model,
            "initial_prompt": self.initial_prompt,
            "responses": self.responses,
            "num_iterations": self.num_iterations,
            "response_tokens": self.response_tokens,
            "nv_recall_metrics": self.nv_recall_metrics,
            "phase1_similarity": self.phase1_similarity,
            "phase1_successful": self.phase1_successful,
            "best_of_n": self.best_of_n,
            "best_of_n_results": self.best_of_n_results if self.best_of_n else {}
        }

        print(f"Saving extraction log to {self.log_path}")
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)
