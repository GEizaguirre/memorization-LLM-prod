import fitz


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def remove_preamble(
    text,
    preamble_word: str = "Preamble"
) -> str:

    preamble_index = text.find(preamble_word)
    if preamble_index != -1:
        return text[preamble_index:]
    else:
        return text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract text from a PDF file and save it to a .txt file.")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file.")
    parser.add_argument(
        "--txt_path",
        type=str,
        default=None,
        help="Path to the output .txt file."
    )
    args = parser.parse_args()

    if args.txt_path is None:
        args.txt_path = args.pdf_path.replace(".pdf", ".txt")

    extracted_text = extract_text_from_pdf(args.pdf_path)
    extracted_text = remove_preamble(extracted_text)
    with open(args.txt_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"Extracted text from {args.pdf_path} and saved it to {args.txt_path}.")
