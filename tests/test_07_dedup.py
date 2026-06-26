from scripts.clean.dedup import normalize_query


def test_normalize_query():
    assert normalize_query(" Point to the Cup. ") == "point to the cup"
