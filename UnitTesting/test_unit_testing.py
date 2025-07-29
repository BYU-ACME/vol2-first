# test_specs.py
"""Python Essentials: Unit Testing.
<Name>
<Class>
<Date>
"""

import unit_testing
import pytest


def test_add():
    assert unit_testing.add(1, 3) == 4, "failed on positive integers"
    assert unit_testing.add(-5, -7) == -12, "failed on negative integers"
    assert unit_testing.add(-6, 14) == 8

def test_divide():
    assert unit_testing.divide(4,2) == 2, "integer division"
    assert unit_testing.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        unit_testing.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.


# Problem 2: write a unit test for specs.month_length().


# Problem 3: write a unit test for specs.operate().


# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = unit_testing.Fraction(1, 3)
    frac_1_2 = unit_testing.Fraction(1, 2)
    frac_n2_3 = unit_testing.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = unit_testing.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7

def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == unit_testing.Fraction(1, 2)
    assert frac_1_3 == unit_testing.Fraction(2, 6)
    assert frac_n2_3 == unit_testing.Fraction(8, -12)


# Problem 5: Write test cases for Set.
