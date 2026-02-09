# Testing Memorization in Production LLMs

This repository is an independent, third-party implementation of the methods described in the paper [Extracting books from production language models (2026)](https://arxiv.org/abs/2601.02671). This algorithms serves to prove memorization of documents in production Large Language Models.

In our case, we use it to check if an LLM has memorized business question-SQL pairs of TPC-H queries.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Current implementation

I implement both the "direct" extraction and the BoN extraction (using a jailbreak that hacks the model scaffold). With direct extraction we got 0.7 nv-recall in Gemini-3.5-pro of Frankestein.

## Usage

### Text preprocessing

I have to preprocess the reference text and remove hyphenation to make it work (may correct this in the future).

```bash
python3 scripts/preprocess_txt.py data/frankenstein_very_short.txt
```

### Usage

[test/test_extraction_txt.py](test/test_extraction_txt.py) includes a usage example of direct extraction. [test/test_bon_extraction_txt.py](test/test_bon_extraction_txt.py) is for BoN extraction.

It may be compatible with all the models in [src/config.py](src/config.py).


## Changes with the original work

- Did not "explore different generation configurations: temperature, maximum response length and, where available, frequency penalty and presence penalty" (time issues).
- The execution finishes when an approximate number of tokens similar to that of the reference text is generated.
- BoN checks permutated prompts iteratively until one of them achieves more than 0.6 similarity (instead of launching all prompts and selecting the best one).

## Reference

This project is an independent implementation of the algorithm presented in:

**Extracting books from production language models** Ahmed Ahmed, A. Feder Cooper, Sanmi Koyejo, and Percy Liang.  
*arXiv preprint arXiv:2601.02671*, 2026.  
[[Paper Link](https://arxiv.org/abs/2601.02671)]

```bibtex
@article{ahmed2026extracting,
  title={Extracting books from production language models},
  author={Ahmed, Ahmed and Cooper, A. Feder and Koyejo, Sanmi and Liang, Percy},
  journal={arXiv preprint arXiv:2601.02671},
  year={2026}
}
```

## Disclaimers

No Affiliation: The author of this repository is not affiliated, associated, authorized, endorsed by, or in any way officially connected with Stanford University, the original authors of the paper, or any of their subsidiaries or affiliates.

Independent Work: This code was written from scratch based on the publicly available methodology in the research paper. No original source code from the authors was used in this implementation.

Purpose: This project is for research and educational purposes only. The goal is to facilitate the study of LLM memorization and alignment safety.
