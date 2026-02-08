import re
import argparse


def remove_hyphenation(text: str) -> str:
    # remove hyphenation
    clean_text = re.sub(r'-\s+', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


def preprocess_text(text: str) -> str:
    text = remove_hyphenation(text)
    # text = remove_newlines(text)
    return text


def preprocess_file(
    input_path: str,
    output_path: str = None
):

    if output_path is None:
        file_basename = input_path.rsplit(".", 1)[0]
        output_path = f"{file_basename}_preprocessed.txt"

    with open(input_path, "r") as f:
        text = f.read()

    preprocessed_text = preprocess_text(text)

    with open(output_path, "w") as f:
        f.write(preprocessed_text)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess text file by removing hyphenation."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input text file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Path to save the preprocessed text file."
            "If not provided, saves as <input_basename>_preprocessed.txt"
        )
    )

    args = parser.parse_args()

    preprocess_file(args.input_path, args.output_path)
