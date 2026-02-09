import sys

INPUT_FILE = "data/TPC-H_v3.0.1.txt"
QUERY_HEADER = "Query Definitions"


def remove_preamble(
    text,
    preamble_word: str = QUERY_HEADER
) -> str:

    preamble_index = text.find(preamble_word)
    if preamble_index != -1:
        return text[preamble_index:]
    else:
        return text


def trim(
    text,
    query_beginning: str,
    query_ending: str
) -> str:

    query_beginning_index = text.find(query_beginning)
    query_ending_index = text.find(query_ending)

    if query_beginning_index == -1:
        raise ValueError(f"Could not find the beginning of the query: {query_beginning}")
    if query_ending_index == -1:
        raise ValueError(f"Could not find the end of the query: {query_ending}")

    return text[query_beginning_index:query_ending_index]


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError("Please provide a query_id as a command line argument.")

    query_id: str = sys.argv[1]
    output_file = f"data/q{query_id}.txt"

    with open(INPUT_FILE, "r") as f:
        text = f.read()
    text = remove_preamble(text)
    query_beginning = f"Q{query_id}"
    if int(query_id) < 22:
        query_ending = f"Q{int(query_id) + 1}"
    else:
        query_ending = "General Requirements for Refresh functions"

    trimmed_text = trim(text, query_beginning, query_ending)
    with open(output_file, "w") as f:
        f.write(trimmed_text)
