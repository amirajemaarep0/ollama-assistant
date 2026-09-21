"""Syntax errors: detection, exact positions, and the repair loop."""
import os
from unittest.mock import MagicMock, patch

import pytest


def test_check_python_syntax_reports_location():
    from rag_backend import check_python_syntax

    assert check_python_syntax("x = 1\n", "ok.py") is None

    problem = check_python_syntax("def f(name):\n    print('hi' + name\n", "bad.py")
    assert problem is not None
    assert problem["line"] == 2
    assert "never closed" in problem["msg"]

def test_try_answer_syntax_question_finds_the_error(tmp_path):
    from rag_backend import try_answer_syntax_question

    (tmp_path / "broken.py").write_text("def greet(name):\n    print('hi' + name\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["broken.py"]}

    answer = try_answer_syntax_question("What is wrong with broken.py?", str(tmp_path), meta)
    assert answer is not None
    assert "broken.py" in answer
    assert "line 2" in answer

def test_try_answer_syntax_question_falls_through_when_file_is_clean(tmp_path):
    """A clean file plus a soft trigger must reach the LLM, not answer 'no errors'."""
    from rag_backend import try_answer_syntax_question

    (tmp_path / "fine.py").write_text("x = 1\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["fine.py"]}

    assert try_answer_syntax_question("What is wrong with fine.py?", str(tmp_path), meta) is None
    # Explicit "syntax" wording still gets the all-clear.
    explicit = try_answer_syntax_question("Any syntax errors in fine.py?", str(tmp_path), meta)
    assert explicit is not None and "no syntax errors" in explicit.lower()

def test_check_python_syntax_all_finds_more_than_the_first_error(tmp_path):
    """ast.parse stops at the first error; the repair loop continues past it."""
    from rag_backend import check_python_syntax_all

    source = (
        "def one():\n"
        "    print('a'\n"
        "\n\n"
        "def two():\n"
        "    if True\n"
        "        pass\n"
        "\n\n"
        "def three():\n"
        "    for x in [1, 2\n"
        "        print(x)\n"
    )
    found = check_python_syntax_all(source, "multi.py")
    assert len(found) >= 2, "should report more than the first error"
    assert found[0]["line"] < found[-1]["line"]
    assert not found[0].get("recovered"), "first entry keeps CPython's exact message"
    assert found[-1].get("recovered")

def test_single_error_file_reports_exactly_one_problem():
    """A file with one mistake must report exactly one error, never a cascade."""
    from rag_backend import check_python_syntax_all

    source = (
        "def summarise(rows):\n"
        "    out = []\n"
        "    for key, value in rows.items()\n"
        "        out.append(key)\n"
        "        out.append(value)\n"
        "\n"
        "    return out\n"
    )
    assert len(check_python_syntax_all(source, "one.py")) == 1

def test_check_python_syntax_all_is_empty_for_valid_code():
    from rag_backend import check_python_syntax_all

    assert check_python_syntax_all("def f(x):\n    return x\n", "ok.py") == []

def test_repair_loop_finds_errors_on_adjacent_lines():
    """Regression: proximity clustering merged three real errors into one,
    because a recovering parser's cascade is indistinguishable by line distance."""
    from rag_backend import check_python_syntax_all

    source = (
        '"""Reporting."""\n'
        "from utils import format_price\n"
        "\n\n"
        "def summarise(warehouse)\n"
        "    lines = []\n"
        "    total = 0\n"
        "\n"
        "    for sku, product in warehouse.products.items()\n"
        '        lines.append(f"{sku}"\n'
        "        total += product.total_value()\n"
    )
    found = check_python_syntax_all(source, "reports.py")
    assert [f["line"] for f in found] == [5, 9, 10]
    assert all("invalid syntax" not in f["msg"] for f in found), "each keeps a precise message"

def test_repair_loop_terminates_on_unrepairable_source():
    from rag_backend import check_python_syntax_all

    found = check_python_syntax_all("@@@ !!! ???\n" * 5, "junk.py")
    assert 1 <= len(found) <= 10

def test_unclosed_bracket_line_locates_the_opener():
    from rag_backend import _unclosed_bracket_line

    lines = ["def f():", "    print('a'", "", "def g():"]
    assert _unclosed_bracket_line(lines, 4) == (2, ")")
    assert _unclosed_bracket_line(["x = (1 + 2)", "y = 3"], 2) is None
    # A bracket inside a string is not an opener.
    assert _unclosed_bracket_line(['s = "(("', "y = 1"], 2) is None
