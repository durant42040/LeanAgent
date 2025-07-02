import pytest

from utils.constants import remove_marks


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        ("hello world", "hello world"),
        ("<a>hello</a> world", "hello world"),
        ("<a>hello</a> <a>world</a>", "hello world"),
        ("<a><a>nested</a></a> marks", "nested marks"),
        ("", ""),
        ("<a></a>", ""),
    ],
)
def test_remove_marks(input_string, expected_output):
    """
    Tests that remove_marks correctly strips <a> and </a> tags from a string.
    """
    assert remove_marks(input_string) == expected_output
