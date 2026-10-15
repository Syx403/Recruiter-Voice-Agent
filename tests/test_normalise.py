"""Number phrasings seen in gpt-live-1 transcripts (from live calls)."""
from backend.normalise import parse_amount, words_to_number
from backend.tools.evidence import _numbers


def test_bare_hundred():
    assert _numbers("I closed about  hundred and forty percent of it") == {140.0}
    assert _numbers("about a hundred and forty percent") == {140.0}
    assert words_to_number("two hundred to two thousand employees").startswith("200")


def test_digit_point_digit():
    assert _numbers("final contract was about 1  point 2") == {1.2}
    a = parse_amount("about 1 point 2 crore rupees", "INR")
    assert a is not None and abs(a.amount - 1.2) < 1e-9


def test_homophones_before_scale_words():
    from backend.normalise import words_to_number
    assert words_to_number("for crore rupees") == "4 crore rupees"
    assert words_to_number("for the team") == "for the team"
