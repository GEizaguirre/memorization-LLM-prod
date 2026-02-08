from config import MAXIMUM_PHASE1_TOKENS
from utils import num_tokens_to_num_words, text_to_words


def longest_common_substring(T, R):
    """
    Find the length of the longest common substring between two sequences of words.

    Args:
        T: List of words from baseline string
        R: List of words from LLM response string

    Returns:
        Length of the longest common substring
    """
    if not T or not R:
        return 0

    len_T = len(T)
    len_R = len(R)

    # Table to store lengths of common substrings
    dp = [[0] * (len_R + 1) for _ in range(len_T + 1)]

    max_length = 0

    # Fill the table
    for i in range(1, len_T + 1):
        for j in range(1, len_R + 1):
            if T[i - 1] == R[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0

    return max_length


def normalized_similarity_score(target, response):
    """
    Calculate the normalized similarity score between target and response strings.

    Args:
        target: Target string (t)
        response: LLM response string (r)

    Returns:
        Normalized similarity score s(T, R) in [0, 1]
    """
    # Split strings into whitespace-delimited tokens
    num_words = num_tokens_to_num_words(MAXIMUM_PHASE1_TOKENS)
    T = text_to_words(target)
    R = text_to_words(response)[:num_words]

    if not T:
        return 0.0

    longest_len = longest_common_substring(T, R)

    score = longest_len / len(T)

    return score
