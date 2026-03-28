"""Tests for the computation engine."""

from __future__ import annotations

import math
import pytest

from klomboagi.reasoning.compute import ComputeEngine, ComputeResult


@pytest.fixture
def engine():
    return ComputeEngine()


class TestArithmetic:

    def test_addition(self, engine):
        r = engine.compute("2 + 3")
        assert r.success and r.result == 5

    def test_subtraction(self, engine):
        r = engine.compute("10 - 4")
        assert r.success and r.result == 6

    def test_multiplication(self, engine):
        r = engine.compute("7 * 8")
        assert r.success and r.result == 56

    def test_division(self, engine):
        r = engine.compute("15 / 3")
        assert r.success and r.result == 5

    def test_power(self, engine):
        r = engine.compute("2 ** 10")
        assert r.success and r.result == 1024

    def test_caret_power(self, engine):
        r = engine.compute("2^10")
        assert r.success and r.result == 1024

    def test_order_of_operations(self, engine):
        r = engine.compute("2 + 3 * 4")
        assert r.success and r.result == 14

    def test_parentheses(self, engine):
        r = engine.compute("(2 + 3) * 4")
        assert r.success and r.result == 20

    def test_negative(self, engine):
        r = engine.compute("-5 + 3")
        assert r.success and r.result == -2

    def test_decimal(self, engine):
        r = engine.compute("3.14 * 2")
        assert r.success and abs(r.result - 6.28) < 0.01

    def test_complex_expression(self, engine):
        r = engine.compute("(10 + 5) * 2 - 3")
        assert r.success and r.result == 27

    def test_division_by_zero(self, engine):
        r = engine.compute("5 / 0")
        assert not r.success


class TestFunctions:

    def test_sqrt(self, engine):
        r = engine.compute("sqrt(144)")
        assert r.success and r.result == 12

    def test_factorial(self, engine):
        r = engine.compute("factorial(5)")
        assert r.success and r.result == 120

    def test_nested(self, engine):
        r = engine.compute("sqrt(16) + sqrt(9)")
        assert r.success and r.result == 7


class TestQuestions:

    def test_is_prime_true(self, engine):
        r = engine.compute("is 17 prime?")
        assert r.success and r.result is True

    def test_is_prime_false(self, engine):
        r = engine.compute("is 15 prime?")
        assert r.success and r.result is False

    def test_percentage(self, engine):
        r = engine.compute("what is 15% of 200?")
        assert r.success and r.result == 30

    def test_what_is(self, engine):
        r = engine.compute("what is 7 * 8?")
        assert r.success and r.result == 56


class TestComparisons:

    def test_greater_than(self, engine):
        r = engine.compute("2**10 > 10**3")
        assert r.success and r.result is True

    def test_less_than(self, engine):
        r = engine.compute("5 < 10")
        assert r.success and r.result is True

    def test_equal(self, engine):
        r = engine.compute("3 * 4 == 12")
        assert r.success and r.result is True


class TestVerification:

    def test_verify_correct(self, engine):
        r = engine.verify_fact("2 + 2 = 4")
        assert r and r.result is True

    def test_verify_incorrect(self, engine):
        r = engine.verify_fact("7 * 8 = 54")
        assert r and r.result is False

    def test_verify_sqrt(self, engine):
        r = engine.verify_fact("the square root of 144 is 12")
        assert r and r.result is True

    def test_verify_unknown(self, engine):
        r = engine.verify_fact("dogs are friendly")
        assert r is None


class TestComputeResult:

    def test_to_dict(self, engine):
        r = engine.compute("2 + 2")
        d = r.to_dict()
        assert d["result"] == 4
        assert d["success"] is True
