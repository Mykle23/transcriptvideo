from webapp.transcriber import Segment, clean_segments


def test_collapses_repeated_words():
    seg = Segment(0.0, 2.0, "no no no no no no no no")
    result, removed = clean_segments([seg])
    assert len(result) == 1
    assert result[0].text == "No."
    assert removed == 1


def test_keeps_normal_segment():
    seg = Segment(0.0, 5.0, "Today we discussed the quarterly budget and revenue targets.")
    result, removed = clean_segments([seg])
    assert len(result) == 1
    assert result[0].text == seg.text
    assert removed == 0


def test_merges_consecutive_identical_short_segments():
    segs = [
        Segment(0.0, 0.5, "Si."),
        Segment(0.5, 1.0, "Si."),
        Segment(1.0, 1.5, "Si."),
        Segment(1.5, 2.0, "Si."),
        Segment(2.0, 2.5, "Si."),
    ]
    result, removed = clean_segments(segs)
    assert len(result) == 1
    assert result[0].start == 0.0
    assert result[0].end == 2.5
    assert removed == 4


def test_does_not_merge_fewer_than_four():
    segs = [
        Segment(0.0, 0.5, "Si."),
        Segment(0.5, 1.0, "Si."),
        Segment(1.0, 1.5, "Si."),
    ]
    result, removed = clean_segments(segs)
    assert len(result) == 3
    assert removed == 0


def test_format_timestamp():
    from webapp.transcriber import format_timestamp

    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(59.999) == "00:00:59,999"
