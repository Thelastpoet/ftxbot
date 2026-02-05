"""
Basic test file to verify pre-commit hooks are working.
"""
import os
import sys


def test_basic_functionality():
    """
    Basic test to verify the testing setup works.
    """
    assert 1 + 1 == 2


if __name__ == "__main__":
    test_basic_functionality()
    print("Test passed!")