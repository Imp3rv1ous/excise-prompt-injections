"""Unit tests of the HTML span alignment in decoding. Run: python -m pytest tests -q"""

from excise.data.source import remove
from excise.model.decode import align_html


def test_leading_close_tag_is_kept():
    # The span starts at the previous paragraph's </p> and ends at the injection's </p>.
    t = "<p>text</p>\n<p>INJ</p><ul>"
    assert align_html(t, [(7, 22)]) == [(11, 22)]
    assert remove(t, align_html(t, [(7, 22)])) == "<p>text</p><ul>"


def test_leading_close_tag_then_extend_to_matching_close():
    # Span "</p><p>INJ": drop the leading </p>, then extend the open <p> to its </p>.
    t = "<p>text</p><p>INJ</p><p>next</p>"
    assert align_html(t, [(7, 17)]) == [(11, 21)]
    assert remove(t, align_html(t, [(7, 17)])) == "<p>text</p><p>next</p>"


def test_close_tag_rule_off_keeps_leading_close_tag():
    t = "<p>text</p>\n<p>INJ</p><ul>"
    assert align_html(t, [(7, 22)], close_tag_rule=False) == [(7, 22)]


def test_close_tag_with_opener_inside_span_unchanged():
    t = "<p>text</p><p>INJ</p><p>next</p>"
    assert align_html(t, [(11, 21)]) == [(11, 21)]


def test_void_tags_and_half_tag():
    # A span edge inside a tag moves out to the whole tag.
    t = '<p>a</p><div class="x"><br>INJ</div><p>b</p>'
    assert align_html(t, [(13, 30)]) == [(8, 36)]
