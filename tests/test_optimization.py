"""Function rewriting and the verification of generated code."""
import os
from unittest.mock import MagicMock, patch

import pytest


def test_extract_function_source_is_exact():
    from rag_backend import extract_function_source

    source = (
        "import os\n\n\n"
        "@staticmethod\n"
        "def target(a, b):\n"
        "    return a + b\n\n\n"
        "def other():\n"
        "    pass\n"
    )
    name, code, line = extract_function_source(source, "target", "m.py")
    assert name == "target"
    assert line == 4, "decorator must be included"
    assert code.startswith("@staticmethod")
    assert "return a + b" in code
    assert "def other" not in code, "must not bleed into the next function"

def test_find_requested_function_matches_names_in_the_file():
    from rag_backend import find_requested_function

    source = "def summarise(x):\n    pass\n\n\nclass C:\n    def lookup(self):\n        pass\n"
    assert find_requested_function("speed up summarise please", source, "m.py") == "summarise"
    assert find_requested_function("optimize the lookup method", source, "m.py") == "C.lookup"
    assert find_requested_function("optimize this file", source, "m.py") is None

def test_differential_test_catches_a_behaviour_change():
    """A rewrite can parse and keep its name and still be wrong."""
    from rag_backend import differential_test

    original = "def f(items):\n    return len(items) * 2\n"
    wrong = "def f(items):\n    return len(items)\n"
    result = differential_test(original, wrong, "f")
    assert result["status"] == "ok"
    assert result["mismatches"], "should have found disagreements"

def test_differential_test_passes_an_equivalent_rewrite():
    from rag_backend import differential_test

    original = "def f(items):\n    total = 0\n    for x in items:\n        total += x\n    return total\n"
    equivalent = "def f(items):\n    return sum(items)\n"
    result = differential_test(original, equivalent, "f")
    assert result["status"] == "ok"
    assert result["ran"] > 0
    assert not result["mismatches"]

def test_verify_optimized_function_rejects_unparseable_code():
    from rag_backend import verify_optimized_function

    reply = "```python\ndef f(x:\n    return x\n```"
    note = verify_optimized_function(reply, "f")
    assert "does not parse" in note

def test_verify_optimized_function_notices_a_renamed_function():
    from rag_backend import verify_optimized_function

    reply = "```python\ndef something_else(x):\n    return x\n```"
    note = verify_optimized_function(reply, "f")
    assert "not `f`" in note
