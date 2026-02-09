from permutator import BoNPermutator


def test_permutator_sequence_is_deterministic():
    text = "Hello, World! This is a test."
    permutator = BoNPermutator(text, seed=19)

    outputs = [permutator.next() for _ in range(6)]
    expected = [
        "HelLO, World! this is a test.",
        "HELLO, WoRld! thIS IS A tesT.",
        "Hello, World! This is a +est.",
        "heLlo, w O RLD! thIS Is a tEST.",
        "Hello, World! Th is is a test.",
        "H ello, World ! This is a test." + " "
    ]
    print(outputs)
    assert outputs == expected


def test_permutator_seed_changes_sequence():
    text = "Hello, World! This is a test."
    permutator = BoNPermutator(text, seed=20)
    outputs = [permutator.next() for _ in range(6)]
    print(outputs)
    assert outputs != [
        "HelLO, World! this is a test.",
        "HELLO, WoRld! thIS IS A tesT.",
        "Hello, World! This is a +est.",
        "heLlo, w O RLD! thIS Is a tEST.",
        "Hello, World! Th is is a test.",
        "H ello, World ! This is a test." + " "
    ]
