
from direct_extraction import Extractor
from config import Model

FILE_PATH = "data/frankenstein_very_short_preprocessed.txt"
MODEL = "gemini_2_5_pro"


def main(
    model: Model,
    text_path: str,
    verbose: bool = False
):
    with open(text_path, "r") as f:
        reference_text = f.read()

    extractor = Extractor(
        model=model,
        reference_text=reference_text,
        verbose=verbose

    )
    extractor.extract()


if __name__ == "__main__":

    model_enum: Model = Model[MODEL.upper()]

    main(
      model=model_enum,
      text_path=FILE_PATH,
      verbose=True
    )
