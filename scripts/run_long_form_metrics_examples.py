import pathlib

from long_form_metrics import (
    near_verbatim_metrics,
    text_to_words
)


def summarize_case(name: str, book_text: str, gen_text: str) -> None:
    metrics = near_verbatim_metrics(book_text, gen_text)
    print(f"== {name} ==")
    print(f"matched_words: {metrics.matched}")
    print(f"nv_recall: {metrics.nv_recall:.6f}")
    print(f"missing_words: {metrics.missing}")
    print(f"additional_words: {metrics.additional}")
    print(f"num_blocks: {len(metrics.blocks)}")
    if metrics.blocks:
        print(f"first_block: {metrics.blocks[0]}")
    print()


def main() -> None:
    book_path = pathlib.Path("data/frankenstein_short.txt")
    book_text = book_path.read_text(encoding="utf-8")
    book_words = text_to_words(book_text)

    def insert_noise(words, stride, token):
        out = []
        for idx, word in enumerate(words, 1):
            out.append(word)
            if idx % stride == 0:
                out.append(token)
        return out

    def delete_stride(words, stride):
        out = []
        for idx, word in enumerate(words, 1):
            if idx % stride != 0:
                out.append(word)
        return out

    def substitute_stride(words, stride, token):
        out = []
        for idx, word in enumerate(words, 1):
            if idx % stride == 0:
                out.append(token)
            else:
                out.append(word)
        return out

    summarize_case("Exact full text", book_text, " ".join(book_words))

    inserted = insert_noise(book_words, stride=200, token="NOISE")
    summarize_case("Insert 1 word every 200", book_text, " ".join(inserted))

    deleted = delete_stride(book_words, stride=200)
    summarize_case("Delete 1 word every 200", book_text, " ".join(deleted))

    substituted = substitute_stride(book_words, stride=200, token="NOISE")
    summarize_case("Substitute 1 word every 200", book_text, " ".join(substituted))


if __name__ == "__main__":
    main()
