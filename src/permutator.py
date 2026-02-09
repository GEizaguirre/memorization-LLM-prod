import math
import random
import re


class BoNPermutator:

    def __init__(
        self,
        reference_text: str,
        seed: int = 0
    ):
        self.reference_text = reference_text
        self.seed = seed
        self._rng = random.Random(seed)
        self._sigma = 0.6
        self._punctuation = [".", ",", "!", "?", ";", ":"]
        self._substitution_map = {
            "a": ["@", "4"],
            "b": ["8"],
            "c": ["(", "<"],
            "e": ["3"],
            "g": ["9"],
            "i": ["1", "!"],
            "l": ["1", "|"],
            "o": ["0"],
            "s": ["$", "5"],
            "t": ["7", "+"],
            "z": ["2"]
        }
        self._perturbations = self._build_perturbation_pool()

    def next(
        self
    ) -> str:
        perturbation = self._rng.choice(self._perturbations)
        local_rng = random.Random(self._rng.getrandbits(64))
        return perturbation(
            self.reference_text,
            local_rng
        )

    def _build_perturbation_pool(
        self
    ):
        pool = [self._identity]

        for p in (0.2, 0.5):
            pool.append(lambda text, rng, p=p: self._capitalization(text, rng, p))

        for padd, prm in ((0.05, 0.05), (0.1, 0.1)):
            pool.append(lambda text, rng, padd=padd, prm=prm: self._spacing(text, rng, padd, prm))

        pool.append(lambda text, rng: self._word_order_shuffle(text, rng, 0.3))

        for psub in (0.1, 0.05):
            pool.append(lambda text, rng, psub=psub: self._character_substitution(text, rng, psub))

        for padd, prm in ((0.05, 0.05), (0.1, 0.1)):
            pool.append(lambda text, rng, padd=padd, prm=prm: self._punctuation_edits(text, rng, padd, prm))

        pool.append(lambda text, rng: self._word_scrambling(text, rng, math.sqrt(self._sigma)))
        pool.append(lambda text, rng: self._capitalization(text, rng, math.sqrt(self._sigma)))
        pool.append(lambda text, rng: self._ascii_noising(text, rng, self._sigma ** 3))

        for p in (0.2, 0.5):
            for padd, prm in ((0.05, 0.05), (0.1, 0.1)):
                composite = self._composite([
                    lambda text, rng, p=p: self._capitalization(text, rng, p),
                    lambda text, rng, padd=padd, prm=prm: self._spacing(text, rng, padd, prm)
                ])
                pool.append(composite)

        pool.append(self._composite([
            lambda text, rng: self._word_scrambling(text, rng, math.sqrt(self._sigma)),
            lambda text, rng: self._capitalization(text, rng, math.sqrt(self._sigma)),
            lambda text, rng: self._ascii_noising(text, rng, self._sigma ** 3)
        ]))

        return pool

    def _composite(
        self,
        steps
    ):
        def apply(text, rng):
            for step in steps:
                step_rng = random.Random(rng.getrandbits(64))
                text = step(text, step_rng)
            return text
        return apply

    def _identity(
        self,
        text: str,
        rng
    ) -> str:
        return text

    def _capitalization(
        self,
        text: str,
        rng,
        p: float
    ) -> str:
        chars = []
        for ch in text:
            if ch.isalpha() and rng.random() < p:
                if ch.islower():
                    ch = ch.upper()
                else:
                    ch = ch.lower()
            chars.append(ch)
        return "".join(chars)

    def _spacing(
        self,
        text: str,
        rng,
        padd: float,
        prm: float
    ) -> str:
        chars = []
        text_len = len(text)
        for i, ch in enumerate(text):
            if ch == " ":
                if rng.random() < prm:
                    continue
                chars.append(ch)
                continue

            chars.append(ch)
            if rng.random() < padd:
                next_char = text[i + 1] if i + 1 < text_len else ""
                if not next_char or not next_char.isspace():
                    chars.append(" ")
        return "".join(chars)

    def _word_order_shuffle(
        self,
        text: str,
        rng,
        pshuffle: float
    ) -> str:
        parts = re.split(r"([.!?])", text)
        if len(parts) == 1:
            return self._shuffle_sentence(parts[0], rng, pshuffle)

        output = []
        i = 0
        while i < len(parts):
            sentence = parts[i]
            delim = parts[i + 1] if i + 1 < len(parts) else ""
            output.append(self._shuffle_sentence(sentence, rng, pshuffle))
            if delim:
                output.append(delim)
            i += 2
        return "".join(output)

    def _shuffle_sentence(
        self,
        sentence: str,
        rng,
        pshuffle: float
    ) -> str:
        if not sentence:
            return sentence

        leading_ws_match = re.match(r"\s*", sentence)
        leading_ws = leading_ws_match.group(0) if leading_ws_match else ""
        core = sentence[len(leading_ws):]
        tokens = core.split()
        if len(tokens) > 1 and rng.random() < pshuffle:
            rng.shuffle(tokens)
        return leading_ws + " ".join(tokens)

    def _character_substitution(
        self,
        text: str,
        rng,
        psub: float
    ) -> str:
        chars = []
        for ch in text:
            if ch.isalpha() and rng.random() < psub:
                options = self._substitution_map.get(ch.lower())
                if options:
                    replacement = rng.choice(options)
                    replacement = replacement.upper() if ch.isupper() else replacement.lower()
                    chars.append(replacement)
                else:
                    chars.append(ch)
            else:
                chars.append(ch)
        return "".join(chars)

    def _punctuation_edits(
        self,
        text: str,
        rng,
        padd: float,
        prm: float
    ) -> str:
        chars = []
        for ch in text:
            if ch in self._punctuation:
                if rng.random() < prm:
                    continue
                chars.append(ch)
                continue

            chars.append(ch)
            if ch.isalpha() and rng.random() < padd:
                chars.append(rng.choice(self._punctuation))
        return "".join(chars)

    def _word_scrambling(
        self,
        text: str,
        rng,
        p: float
    ) -> str:
        chunks = re.split(r"(\s+)", text)
        scrambled = []
        for chunk in chunks:
            if not chunk or chunk.isspace():
                scrambled.append(chunk)
                continue
            scrambled.append(self._scramble_token(chunk, rng, p))
        return "".join(scrambled)

    def _scramble_token(
        self,
        token: str,
        rng,
        p: float
    ) -> str:
        match = re.match(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$", token)
        if not match:
            return token

        prefix, core, suffix = match.groups()
        if len(core) <= 3 or rng.random() >= p:
            return token

        interior = list(core[1:-1])
        rng.shuffle(interior)
        return prefix + core[0] + "".join(interior) + core[-1] + suffix

    def _ascii_noising(
        self,
        text: str,
        rng,
        p: float
    ) -> str:
        chars = []
        for ch in text:
            code = ord(ch)
            if 32 <= code <= 126 and rng.random() < p:
                delta = rng.choice((-1, 1))
                new_code = code + delta
                if 32 <= new_code <= 126:
                    chars.append(chr(new_code))
                else:
                    chars.append(ch)
            else:
                chars.append(ch)
        return "".join(chars)
