"""Sample test file for a Python project using pytest.
Author: Angel Martinez-Tenor, 2025.
"""

from __future__ import annotations

import pytest


# Sample function to test (in a real project, this would be in a separate module)
def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b


# Fixture to provide test data
@pytest.fixture
def sample_data() -> dict[str, int]:
    """Fixture to provide test data."""
    return {"x": 5, "y": 3, "expected_sum": 8}


# Test class for grouping related tests
class TestMathOperations:
    """Tests for math operations."""

    def test_add_positive_numbers(self, sample_data: dict[str, int]) -> None:
        """Test that add() works with positive numbers."""
        result = add(sample_data["x"], sample_data["y"])
        assert result == sample_data["expected_sum"], f"Expected {sample_data['expected_sum']}, but got {result}"

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (0, 0, 0),
            (-1, 1, 0),
            (-2, -3, -5),
        ],
    )
    def test_add_various_numbers(self, a: int, b: int, expected: int) -> None:
        """Test add() with various number combinations."""
        result = add(a, b)
        assert result == expected, f"Expected {expected}, but got {result}"


# Standalone test function
@pytest.mark.slow
def test_add_large_numbers() -> None:
    """Test add() with large numbers (marked as slow)."""
    result = add(1000000, 2000000)
    assert result == 3000000, f"Expected 3000000, but got {result}"


# Example of a test that expects an exception
def test_add_invalid_input() -> None:
    """Test that add() raises TypeError with invalid input."""
    with pytest.raises(TypeError):
        add("1", 2)  # type: ignore[arg-type]
