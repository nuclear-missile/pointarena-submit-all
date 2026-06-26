from scripts.clean.rule_filter import keep_image_size, keep_point_list, keep_query


def test_keep_query():
    assert keep_query("Point to the cup")
    assert not keep_query("")
    assert not keep_query("n/a")


def test_keep_image_size():
    assert keep_image_size(224, 224)
    assert not keep_image_size(100, 100)


def test_keep_point_list():
    assert keep_point_list([{"x": 0.1, "y": 0.2}])
    assert not keep_point_list([{"x": 1.5, "y": 0.2}])
