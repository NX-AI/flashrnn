# Integer CSP solver file - CONSTRINT
# This script solves CSP problems tied to integer variables
# and optimized for multiplicative and divisibility constraints (e.g. tiling sizes)
# It tries to achieve arc consistency, though if ranges explode it opts for keeping
# integer ranges instead of lots of values, even though some might not be arc consistent.
#
# Parser Format looks like the following
# First and most obvious, comment lines are denoted by #
# Constraints are Lines ending with ;
# ; is equivalent to a logic and
# | has higher precendence (and should be used EOL) and denotes OR
# With a line there are three types of constraint:
# == denotes an equality constraint
# <= denotes and inequality constraint (no >= here)
# TERM_A % TERM_B == 0 denotes a divisible constraint (specialty of this library)
# As seen in the previous line, there are TERMS, which can consist of (low to high precedence)
# sum: TERM_A + TERM_B
# product: TERM_A * TERM_B
# power: TERM_A ^ INTEGER
# Note that there are no brackets and therefore by the precedence rules
# for the higher precendence level there is no TERMs consisting of a lower precendence
# brackets can be added for readability but are STRIPPED! (for now at least)
# Now again, there was another type of TERM introduced before. Basically, this is the VARIBLE.
# There are also INTEGERs and LISTs here, denoting special variables that are tied to a single or multiple integers.
# A LIST looks like [INTEGER, INTEGER, INTEGER,...] and denotes that all of the noted values are possible
# A general VARIABLE is any string consisting of alphabet and _ characters
# An INTEGER is a number in decimal format
#
# That's it. Keep in mind that there is no / or - operations, but changing you equations,
# it is always possible to formulate the proper constraint.
# Pro-Tip: Use comments for readability (i.e. write your constraint using - and / if thats easier)
# and add helper variables (connected via equalities) to simplify things.
#
# Additional note: This is no fully fledged solver, it just applies constraints to variables one way, i.e.
# it tries to achieve arc consistency (though it might not be fully enforced).
# So use it to narrow down the solution space and later check solutions by setting specific variables manually.
# Why not use an existing solver for that? Because this one is optimized for integer ranges and problems,
# i. e. problems that consist of integer variables, constraints of the forms shown above and terms of the form
# shown above.
#
# Copyright NXAI GmbH, Korbinian Poeppel


import doctest
import logging
import math
import os
import re
import sys
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, wraps
from typing import Iterable, Optional, Sequence, Union

LOGGER = logging.getLogger(__name__)

try:
    from .caching import cache_decorator
except ImportError:

    def cache_decorator(cache_dir):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator


PRIMES = set(
    (
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        91,
        97,
    )
)
EXPANSION_LIMIT = 2048
PRIME_EXPANSION_LIMIT = 8
INT_UPPER_LIMIT = 1 << 30


def gcd(a: int, b: int):
    """
    >>> gcd(25, 10)
    5
    """
    upper, lower = max(a, b), min(a, b)
    while lower != 0:
        upper, lower = lower, upper % lower
    return upper


def primes():
    yield 2
    j = 3
    while True:
        if j in PRIMES:
            yield j
        elif j > max(PRIMES):
            prime = True
            for p in PRIMES:
                if j % p == 0:
                    prime = False
                    break
                if j // p < p:
                    break
            if prime:
                PRIMES.add(j)
                yield j
        j += 2


@lru_cache(maxsize=8192)
def prime_factors(num: int):
    """
    >>> prime_factors(1024)
    (2,)
    >>> prime_factors(1021)
    (1021,)
    >>> prime_factors(1)
    ()
    """
    pfactors = set()
    if num == 0:
        return tuple()
    while num > 1:
        for prime in primes():
            if num % prime == 0:
                num //= prime
                pfactors.add(prime)
                break
            if prime * prime > num:
                pfactors.add(num)
                PRIMES.add(num)
                num = 1
                break
    return tuple(sorted(pfactors))


@dataclass
class IntegerVariable:
    upper_limit: int = INT_UPPER_LIMIT  # including limit
    lower_limit: int = 1
    values: Optional[Sequence[int]] = None  # none means all values possible
    factor: int = 1  # factors that a number must have
    sole_prime_factors: Optional[Sequence[int]] = (
        None  # sole prime factors a number can have, None means any
    )
    constant: bool = False

    def __init__(
        self,
        upper_limit=None,
        lower_limit=None,
        values=None,
        sole_prime_factors=None,
        factor=1,
        constant: bool = False,
    ):
        """
        >>> IntegerVariable(5, factor=2)
        IntegerVariable(values=(2, 4))
        """
        if isinstance(upper_limit, Sequence):
            values = upper_limit
            upper_limit = None
        if isinstance(lower_limit, Sequence):
            values = lower_limit
            lower_limit = None
        if upper_limit is None:
            upper_limit = INT_UPPER_LIMIT
        if lower_limit is None:
            lower_limit = 1
        self.factor = factor
        self.sole_prime_factors = (
            tuple(sorted(set(sole_prime_factors)))
            if sole_prime_factors is not None
            else None
        )
        self.values = tuple(sorted(set(values))) if values is not None else None
        if values is not None:
            lower_limit = min(
                values, default=lower_limit if lower_limit is not None else 1
            )
            upper_limit = max(values, default=lower_limit)
        self.upper_limit = min(upper_limit, INT_UPPER_LIMIT)
        self.lower_limit = lower_limit
        self.constant = constant

        self._apply_factor_rules()

    def _apply_factor_rules(self) -> bool:
        ret = False
        if (
            self.values
            and len(self.values) < PRIME_EXPANSION_LIMIT
            and all(val > 0 for val in self.values)
        ):
            all_primes = []
            for val in self.values:
                all_primes += list(prime_factors(val))
            sole_prime_factors = tuple(sorted(set(all_primes)))
            if sole_prime_factors != self.sole_prime_factors:
                ret = True
                # intersection of prime factors
                if self.sole_prime_factors is not None:
                    self.sole_prime_factors = tuple(
                        sorted(
                            set(
                                pr
                                for pr in self.sole_prime_factors
                                if pr in sole_prime_factors
                            )
                        )
                    )
                else:
                    self.sole_prime_factors = sole_prime_factors
        if self.values and self.factor > 1:
            self.values = tuple(val for val in self.values if val % self.factor == 0)
        if (
            self.values
            and len(self.values) < PRIME_EXPANSION_LIMIT
            and all(val > 0 for val in self.values)
        ):
            gcd_val = self.values[0]
            for val in self.values:
                gcd_val = gcd(gcd_val, val)
            if self.factor != gcd_val:
                self.factor = gcd_val
                ret = True
        # apply additional constraints by factor and primes
        if (self.sole_prime_factors is not None or self.factor != 1) and (
            self.size() < EXPANSION_LIMIT or self.values is not None
        ):
            values = self.all_values()
            if self.factor != 1:
                values = tuple(
                    sorted([val for val in values if val % self.factor == 0])
                )
            if self.sole_prime_factors is not None:
                if self.sole_prime_factors and not (
                    set(self.sole_prime_factors)
                    == set(list(PRIMES)[: len(self.sole_prime_factors)])
                    and self.sole_prime_factors[-1] >= self.upper_limit
                ):
                    values = tuple(
                        sorted(
                            [
                                val
                                for val in values
                                if (
                                    all(
                                        (fac in self.sole_prime_factors)
                                        for fac in prime_factors(val)
                                    )
                                    or val == 1
                                )
                            ]
                        )
                    )
            lower_limit = min(values, default=self.lower_limit)
            upper_limit = max(values, default=self.upper_limit)
            if self.values != values:
                ret = True
            self.values = values
            self.lower_limit = lower_limit
            self.upper_limit = upper_limit
        return ret

    def __repr__(self):
        sep = ", "
        osep = sep
        if self.upper_limit == INT_UPPER_LIMIT:
            res = "IntegerVariable("
            sep = ""
        elif self.values is None:
            res = f"IntegerVariable({self.upper_limit}, {self.lower_limit}"
        else:
            res = f"IntegerVariable(values={self.values}"
        if self.sole_prime_factors is not None and self.values is None:
            res += sep + f"sole_prime_factors={self.sole_prime_factors}"
            sep = osep
        if self.factor != 1 and self.values is None:
            res += sep + f"factor={self.factor}"
            sep = osep
        res += ")"
        return res

    def whole_range(self) -> bool:
        return self.values is None

    def copy(self):
        return IntegerVariable(
            self.upper_limit,
            self.lower_limit,
            tuple(val for val in self.values) if self.values is not None else None,
            factor=self.factor,
            sole_prime_factors=tuple(self.sole_prime_factors)
            if self.sole_prime_factors is not None
            else None,
            constant=self.constant,
        )

    def empty(self) -> bool:
        if self.whole_range():
            if self.lower_limit <= self.upper_limit:
                return False
        else:
            return not bool(self.values)

    # intersection between existing values and external ones
    def intersect(self, var: "IntegerVariable") -> bool:
        """
        Applies additional restrictions to a variable.
        Returns True if there was an additional restriction.

        >>> IntegerVariable(5, 1, values=[1, 4, 5]).intersect(IntegerVariable(upper_limit=6))
        False
        >>> IntegerVariable(5, 1, values=[1, 4, 5]).intersect(IntegerVariable(upper_limit=3))
        True
        >>> IntegerVariable(sole_prime_factors=(2, 5, 7)).intersect(IntegerVariable(sole_prime_factors=(5, 2)))
        True
        >>> a = IntegerVariable(factor=9)
        >>> a.intersect(IntegerVariable(factor=6))
        True
        >>> a
        IntegerVariable(factor=18)
        """
        if self.constant:
            return False
        if self.whole_range():
            if var.whole_range():
                upper_limit = min(self.upper_limit, var.upper_limit)
                lower_limit = max(self.lower_limit, var.lower_limit)
                values = None
            else:
                values = tuple(
                    val
                    for val in var.values
                    if val <= self.upper_limit and val >= self.lower_limit
                )
                upper_limit = max(values, default=0)
                lower_limit = min(values, default=self.lower_limit)
        else:
            if var.whole_range():
                values = tuple(
                    sorted(
                        val
                        for val in self.values
                        if val <= var.upper_limit and val >= var.lower_limit
                    )
                )
            else:
                values = tuple(sorted(val for val in self.values if val in var.values))
            upper_limit = max(values, default=0)
            lower_limit = min(values, default=self.lower_limit)
        ret = False
        if (
            (values is not None and self.values is None)
            or (
                values is not None
                and self.values is not None
                and len(values) < len(self.values)
            )
            or upper_limit < self.upper_limit
            or lower_limit > self.lower_limit
        ):
            ret = True
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.values = values
        factor = self.factor * var.factor // gcd(self.factor, var.factor)
        if self.factor != factor:
            self.factor = factor
            ret = True
        sole_prime_factors = (
            self.sole_prime_factors
            if var.sole_prime_factors is None
            else (
                var.sole_prime_factors
                if self.sole_prime_factors is None
                else tuple(
                    sorted(
                        set(
                            fac
                            for fac in self.sole_prime_factors
                            if fac in var.sole_prime_factors
                        )
                    )
                )
            )
        )
        if sole_prime_factors != self.sole_prime_factors:
            ret = True
            self.sole_prime_factors = sole_prime_factors
        ret = self._apply_factor_rules() or ret
        return ret

    def unify(self, var: "IntegerVariable") -> bool:
        """
        Build the union of two integer sets

        >>> IntegerVariable(values=[1, 4, 5]).unify(IntegerVariable(values=[3]))
        True
        >>> IntegerVariable(values=[1, 4, 5]).unify(IntegerVariable(values=[1]))
        False
        >>> a = IntegerVariable(factor=10)
        >>> a.unify(IntegerVariable(factor=15))
        True
        >>> a
        IntegerVariable(factor=5)
        >>> a = IntegerVariable(factor=10)
        >>> a.unify(IntegerVariable(factor=20))
        False
        """
        if self.constant:
            return False
        if self.whole_range():
            if var.whole_range():
                upper_limit = max(self.upper_limit, var.upper_limit)
                lower_limit = min(self.lower_limit, var.lower_limit)
                values = None
            else:
                values = tuple(
                    set(
                        list(range(self.lower_limit, self.upper_limit + 1))
                        + list(var.values)
                    )
                )
                upper_limit = max(values, default=0)
                lower_limit = min(values, default=self.lower_limit)
        else:
            if var.whole_range():
                values = tuple(
                    set(
                        list(range(var.lower_limit, var.upper_limit + 1))
                        + list(self.values)
                    )
                )
            else:
                values = tuple(set(list(self.values) + list(var.values)))
            upper_limit = max(values, default=0)
            lower_limit = min(values, default=self.lower_limit)
        ret = False

        factor = gcd(self.factor, var.factor)
        if factor != self.factor:
            ret = True
            self.factor = factor

        sole_prime_factors = (
            tuple(
                sorted(
                    set(list(self.sole_prime_factors) + list(var.sole_prime_factors))
                )
            )
            if (
                self.sole_prime_factors is not None
                and var.sole_prime_factors is not None
            )
            else None
        )
        if self.sole_prime_factors != sole_prime_factors:
            self.sole_prime_factors = sole_prime_factors
            ret = True

        if (
            (values is not None and self.values is None)
            or (
                values is not None
                and self.values is not None
                and len(values) > len(self.values)
            )
            or upper_limit > self.upper_limit
            or lower_limit < self.lower_limit
        ):
            ret = True
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.values = tuple(sorted(values)) if values is not None else None
        ret = self._apply_factor_rules() or ret
        return ret

    def all_values(self) -> tuple:
        if self.values is not None:
            return self.values
        else:
            if self.upper_limit - self.lower_limit > EXPANSION_LIMIT:
                raise ValueError(
                    f"Won't expand list with {self.upper_limit} - {self.lower_limit} elements."
                )
            return tuple(range(self.lower_limit, self.upper_limit + 1))

    def size(self) -> int:
        if self.values is not None:
            return len(self.values)
        else:
            return self.upper_limit - self.lower_limit + 1

    def whole_range_copy(self):
        if self.whole_range():
            return self.copy()
        else:
            return IntegerVariable(
                upper_limit=min(
                    self.upper_limit, max(self.values, default=self.upper_limit)
                ),
                lower_limit=max(
                    self.lower_limit, min(self.values, default=self.lower_limit)
                ),
            )

    def __eq__(self, var: "IntegerVariable"):
        """
        >>> IntegerVariable(5, 1, [1, 2, 5]) == IntegerVariable(5, 1, [1, 2, 5])
        True
        >>> IntegerVariable(5, 1, [1, 2]) == IntegerVariable(5, 1, [1, 2, 5])
        False
        >>> IntegerVariable(5) == IntegerVariable(5)
        True
        >>> IntegerVariable(5) == IntegerVariable(5, 1, [1, 2, 5])
        False
        >>> IntegerVariable(5, 1, [1, 2, 5]) == IntegerVariable(5, 0, [0, 2, 5])
        False
        >>> IntegerVariable(5, 0, [1, 2, 5]) == IntegerVariable(5, 0, [1, 2, 5])
        True
        >>> IntegerVariable(3, factor=2) == IntegerVariable([2,])
        True
        """

        return (
            self.upper_limit == var.upper_limit
            and self.lower_limit == var.lower_limit
            and (
                (
                    self.values is None
                    and var.values is None
                    and self.sole_prime_factors == var.sole_prime_factors
                    and self.factor == var.factor
                )
                or (
                    self.values is not None
                    and var.values is not None
                    and sorted(self.values) == sorted(var.values)
                )
                or (self.values is not None and sorted(self.values))
                == list(range(var.lower_limit, var.upper_limit + 1))
                or (var.values is not None and sorted(var.values))
                == list(range(self.lower_limit, self.upper_limit + 1))
            )
        )


@dataclass
class VariablesDict:
    variables: dict[str, IntegerVariable]

    def __contains__(self, obj):
        return obj in self.variables

    def __getitem__(self, obj):
        return self.variables[obj]

    def __setitem__(self, obj, val):
        self.variables[obj] = val

    def __delitem__(self, obj):
        del self.variables[obj]

    def __iter__(self) -> Iterable:
        return iter(self.variables)

    def items(self):
        return self.variables.items()

    def len(self):
        return len(self.variables)

    def copy(self):
        return VariablesDict({key: var.copy() for key, var in self.variables.items()})

    def __len__(self):
        return len(self.variables)

    def intersect(self, variables: "VariablesDict"):
        change = False
        for key, var in self.variables.items():
            if key in variables:
                change = var.intersect(variables[key]) or change
        return change

    def unify(self, variables: "VariablesDict"):
        change = False
        for key, var in self.variables.items():
            if key in variables:
                change = var.unify(variables[key]) or change
        return change

    def empty(self) -> bool:
        return any(var.empty() for _, var in self.variables.items())

    def single_solution(self) -> bool:
        return all(var.size() == 1 or var.constant for _, var in self.variables.items())


class Term:
    variables: tuple

    def _check_empty(
        self, sub_vars: VariablesDict, res_var: Optional[IntegerVariable] = None
    ):
        return sub_vars.empty() or (False if res_var is None else res_var.empty())

    def order(self, variables: VariablesDict) -> int:
        return 10000

    def topdown(self, res_var: IntegerVariable, sub_vars: VariablesDict) -> bool:
        """
        Applies a toplevel restriction of a term to all sub variables.
        Returns True if there was a change.
        """
        return False

    def bottomup(self, sub_vars: VariablesDict) -> IntegerVariable:
        """
        Returns a Variable that contains all possible Values that might result from a Term

        sub_vars is kept immutable
        """
        return IntegerVariable()


class RecursiveTerm(Term):
    terms: list["RecursiveTerm"]

    def __init__(self, terms):
        self.variables = []
        for term in terms:
            self.variables += list(term.variables)
        self.variables = tuple(self.variables)
        self.terms = terms


class VariableTerm(Term):
    def __init__(self, variable: str, constant: bool = False):
        self.variables = (variable,)
        self.constant = constant

    def order(self, variables: VariablesDict) -> int:
        if self.variables[0] in variables:
            return 0 + max(
                0, int(2 * math.log(max(1, 1 + variables[self.variables[0]].size())))
            )
        else:
            return 100000

    def topdown(self, res_var: IntegerVariable, sub_vars: VariablesDict):
        """
        >>> VariableTerm("a").topdown(IntegerVariable(5), sub_vars=VariablesDict({"a": IntegerVariable(6, 1, [1,3,6])}))
        True
        >>> VariableTerm("a").topdown(IntegerVariable(5), sub_vars=VariablesDict({"a": IntegerVariable(5, 1, [1,3,5])}))
        False
        """
        # if self._check_empty(sub_vars, res_var):
        #     return False
        if self.variables[0] in sub_vars:
            return sub_vars[self.variables[0]].intersect(res_var)
        return False

    def bottomup(self, sub_vars: VariablesDict):
        """
        >>> VariableTerm("a").bottomup(VariablesDict({"a": IntegerVariable(4)}))
        IntegerVariable(4, 1)
        >>> VariableTerm("a").bottomup(VariablesDict({"b": IntegerVariable(4)}))
        IntegerVariable()
        """
        # if self._check_empty(sub_vars):
        #     return IntegerVariable(values=())

        if self.variables[0] in sub_vars:
            return sub_vars[self.variables[0]].copy()
        else:
            return IntegerVariable()

    def __str__(self):
        """
        >>> str(VariableTerm("a"))
        'a'
        >>> str(VariableTerm("12"))
        '12'
        """
        return self.variables[0]

    def __repr__(self):
        return f"VariableTerm({self.variables[0]})"


def floor_pow(a: int, power: float):
    if int(power) == power:
        return int(a**power)
    else:
        if a >= INT_UPPER_LIMIT:
            return INT_UPPER_LIMIT
        return int(a**power)


def ceil_pow(a: int, power: float):
    if int(power) == power:
        return a**power
    else:
        res = a**power
        if int(res) == res:
            return int(res)
        else:
            return int(res) + 1


@dataclass
class PowerTerm(RecursiveTerm):
    def __init__(self, term: RecursiveTerm, power: int = 2):
        """
        >>> var = VariablesDict({"a": IntegerVariable()})
        >>> PowerTerm(VariableTerm("a"), 2).topdown(IntegerVariable(values=(9, 16)), var)
        True
        >>> var['a']
        IntegerVariable(values=(3, 4))
        """
        super().__init__([term])
        self.power = power

    def order(self, variables: VariablesDict) -> int:
        return 8 * self.terms[0].order(variables) ** self.power

    def bottomup(self, sub_vars: VariablesDict) -> IntegerVariable:
        vals = self.terms[0].bottomup(sub_vars)
        factor = int(vals.factor**self.power)
        sole_prime_factors = vals.sole_prime_factors

        lower_limit = floor_pow(vals.lower_limit, self.power)
        upper_limit = ceil_pow(vals.upper_limit, self.power)

        if vals.whole_range():
            values = None
        else:
            values = tuple((int(val**self.power) for val in vals.values))
        return IntegerVariable(
            upper_limit,
            lower_limit,
            values=values,
            factor=factor,
            sole_prime_factors=sole_prime_factors,
        )

    def topdown(self, res_var: IntegerVariable, sub_vars: VariablesDict) -> bool:
        lower_limit = (
            floor_pow(res_var.lower_limit, 1.0 / self.power)
            if res_var.lower_limit > 0
            else 1
        )
        upper_limit = (
            ceil_pow(res_var.upper_limit, 1.0 / self.power)
            if res_var.upper_limit > 0
            else 1
        )
        if res_var.whole_range():
            values = None
        else:
            values = [
                floor_pow(val, 1.0 / self.power)
                for val in res_var.values
                if val > 0
                and floor_pow(floor_pow(val, 1.0 / self.power), self.power) == val
            ]
            values += [
                ceil_pow(val, 1.0 / self.power)
                for val in res_var.values
                if val > 0
                and ceil_pow(ceil_pow(val, 1.0 / self.power), self.power) == val
            ]
            values = sorted(set(values))
        sole_prime_factors = res_var.sole_prime_factors
        factor = floor_pow(res_var.factor, 1.0 / self.power)
        root_var = IntegerVariable(
            upper_limit,
            lower_limit,
            values=values,
            factor=factor,
            sole_prime_factors=sole_prime_factors,
        )
        return self.terms[0].topdown(root_var, sub_vars)

    def __repr__(self):
        return "PowerTerm(" + repr(self.terms[0]) + f", {self.power})"


@dataclass
class ProductTerm(RecursiveTerm):
    def __init__(self, termA: RecursiveTerm, termB: RecursiveTerm):
        super().__init__([termA, termB])

    def order(self, variables: VariablesDict) -> int:
        return 10 * self.terms[0].order(variables) * self.terms[1].order(variables)

    def bottomup(self, sub_vars: VariablesDict) -> IntegerVariable:
        """
        >>> ProductTerm(VariableTerm("a"), VariableTerm("b")).bottomup(VariablesDict({"a": IntegerVariable(5), "b": IntegerVariable(3)}))
        IntegerVariable(15, 1)
        >>> var = VariablesDict({"a": IntegerVariable(factor=3), "b": IntegerVariable(factor=5)})
        >>> ProductTerm(VariableTerm("a"), VariableTerm("b")).bottomup(var)
        IntegerVariable(factor=15)
        """
        # if self._check_empty(sub_vars):
        #     return IntegerVariable(values=())

        left = self.terms[0].bottomup(sub_vars)
        right = self.terms[1].bottomup(sub_vars)

        sole_prime_factors = (
            tuple(
                sorted(
                    set(list(left.sole_prime_factors) + list(right.sole_prime_factors))
                )
            )
            if (
                left.sole_prime_factors is not None
                and right.sole_prime_factors is not None
            )
            else None
        )

        if left.whole_range() and right.whole_range():
            return IntegerVariable(
                left.upper_limit * right.upper_limit,
                left.lower_limit * right.lower_limit,
                factor=left.factor * right.factor,
                sole_prime_factors=sole_prime_factors,
            )
        elif left.whole_range():
            if right.values == ():
                return IntegerVariable(values=())
            if left.upper_limit - left.lower_limit < EXPANSION_LIMIT:
                return IntegerVariable(
                    left.upper_limit * right.upper_limit,
                    left.lower_limit * right.lower_limit,
                    tuple(
                        sorted(
                            set(
                                val * n
                                for val in right.values
                                for n in range(left.lower_limit, left.upper_limit + 1)
                            )
                        )
                    )
                    if len(right.values) * (left.upper_limit - left.lower_limit + 1)
                    < EXPANSION_LIMIT
                    else None,
                    factor=left.factor * right.factor,
                    sole_prime_factors=sole_prime_factors,
                )
            else:
                return IntegerVariable(
                    left.upper_limit * right.upper_limit,
                    left.lower_limit * right.lower_limit,
                    factor=left.factor * right.factor,
                    sole_prime_factors=sole_prime_factors,
                )
        elif right.whole_range():
            if left.values == ():
                return IntegerVariable(values=())
            if right.upper_limit - right.lower_limit < EXPANSION_LIMIT:
                return IntegerVariable(
                    left.upper_limit * right.upper_limit,
                    left.lower_limit * right.lower_limit,
                    tuple(
                        sorted(
                            set(
                                val * n
                                for val in left.values
                                for n in range(right.lower_limit, right.upper_limit + 1)
                            )
                        )
                    )
                    if len(left.values) * (right.upper_limit - right.lower_limit + 1)
                    < EXPANSION_LIMIT
                    else None,
                    factor=left.factor * right.factor,
                    sole_prime_factors=sole_prime_factors,
                )
            else:
                return IntegerVariable(
                    left.upper_limit * right.upper_limit,
                    left.lower_limit * right.lower_limit,
                    factor=left.factor * right.factor,
                    sole_prime_factors=sole_prime_factors,
                )
        else:
            return IntegerVariable(
                left.upper_limit * right.upper_limit,
                left.lower_limit * right.lower_limit,
                tuple(valL * valR for valL in left.values for valR in right.values)
                if len(left.values) * len(right.values) < EXPANSION_LIMIT
                else None,
                factor=left.factor * right.factor,
                sole_prime_factors=sole_prime_factors,
            )

    def topdown(self, res_var: IntegerVariable, sub_vars: VariablesDict):
        """
        >>> var = VariablesDict({"a": IntegerVariable(20), "b": IntegerVariable(20)})
        >>> ProductTerm(VariableTerm("a"), VariableTerm("b")).topdown(IntegerVariable(values=[15]), var)
        True
        >>> var["a"]
        IntegerVariable(values=(1, 3, 5, 15))
        >>> var["b"]
        IntegerVariable(values=(1, 3, 5, 15))
        >>> var = VariablesDict({"a": IntegerVariable(), "b": IntegerVariable()})
        >>> ProductTerm(VariableTerm("a"), VariableTerm("b")).topdown(IntegerVariable(sole_prime_factors=[2, 5]), var)
        True
        >>> var["a"]
        IntegerVariable(sole_prime_factors=(2, 5))
        >>> var["b"]
        IntegerVariable(sole_prime_factors=(2, 5))
        """
        # if self._check_empty(sub_vars, res_var):
        #     return False

        if sub_vars.empty():
            return False

        change = False
        if res_var.sole_prime_factors is not None:
            change = (
                self.terms[0].topdown(
                    IntegerVariable(sole_prime_factors=res_var.sole_prime_factors),
                    sub_vars,
                )
                or change
            )
            change = (
                self.terms[1].topdown(
                    IntegerVariable(sole_prime_factors=res_var.sole_prime_factors),
                    sub_vars,
                )
                or change
            )

        if self.terms[0].order(sub_vars) < self.terms[1].order(sub_vars):
            leftIdx = 0
        else:
            leftIdx = 1
        left = self.terms[leftIdx].bottomup(sub_vars)

        if res_var.whole_range():
            upper_limit_right = (res_var.upper_limit + left.lower_limit - 1) // max(
                1, left.lower_limit
            )  # CEIL DIV
            lower_limit_right = res_var.lower_limit // max(
                1, left.upper_limit
            )  # FLOOR DIV

            change = (
                self.terms[1 - leftIdx].topdown(
                    IntegerVariable(upper_limit_right, lower_limit_right), sub_vars
                )
                or change
            )
            right = self.terms[1 - leftIdx].bottomup(sub_vars)

            upper_limit_left = (res_var.upper_limit + right.lower_limit - 1) // max(
                1, right.lower_limit
            )  # CEIL DIV
            lower_limit_left = res_var.lower_limit // max(
                1, right.upper_limit
            )  # FLOOR DIV
            change = (
                self.terms[leftIdx].topdown(
                    IntegerVariable(upper_limit_left, lower_limit_left), sub_vars
                )
                or change
            )
        else:
            if res_var.values == ():
                change = (
                    self.terms[1 - leftIdx].topdown(
                        IntegerVariable(values=()), sub_vars
                    )
                    or change
                )
                change = (
                    self.terms[leftIdx].topdown(IntegerVariable(values=()), sub_vars)
                    or change
                )
                return change

            right_vals = []
            upper_limit_right = None
            lower_limit_right = None
            if left.whole_range():
                for val in res_var.values:
                    if left.upper_limit - left.lower_limit > EXPANSION_LIMIT:
                        llr = val // left.upper_limit
                        ulr = (val + left.lower_limit - 1) // left.lower_limit
                        lower_limit_right = (
                            llr
                            if lower_limit_right is None
                            else min(llr, lower_limit_right)
                        )
                        upper_limit_right = (
                            ulr
                            if upper_limit_right is None
                            else max(ulr, upper_limit_right)
                        )
                    else:
                        right_vals += [
                            val // lval
                            for lval in range(left.lower_limit, left.upper_limit + 1)
                            if val % lval == 0
                        ]
            else:
                for val in res_var.values:
                    right_vals += [
                        val // lval for lval in left.values if val % lval == 0
                    ]

            right_vals = (
                set(val for val in right_vals) if lower_limit_right is None else None
            )
            upper_limit_right = max(right_vals) if right_vals else upper_limit_right
            lower_limit_right = min(right_vals) if right_vals else lower_limit_right

            change = (
                self.terms[1 - leftIdx].topdown(
                    IntegerVariable(
                        upper_limit_right, lower_limit_right, values=right_vals
                    ),
                    sub_vars,
                )
                or change
            )

            right = self.terms[1 - leftIdx].bottomup(sub_vars)

            left_vals = []
            upper_limit_left = None
            lower_limit_left = None

            if right.whole_range():
                for val in res_var.values:
                    if right.upper_limit - right.lower_limit > EXPANSION_LIMIT:
                        lll = val // right.upper_limit
                        ull = (val + right.lower_limit - 1) // right.lower_limit
                        lower_limit_left = (
                            lll
                            if lower_limit_left is None
                            else min(lll, lower_limit_left)
                        )
                        upper_limit_left = (
                            ull
                            if upper_limit_left is None
                            else max(ull, upper_limit_left)
                        )
                    else:
                        left_vals += [
                            val // rval
                            for rval in range(right.lower_limit, right.upper_limit + 1)
                            if val % rval == 0
                        ]
            else:
                for val in res_var.values:
                    left_vals += [
                        val // rval for rval in right.values if val % rval == 0
                    ]

            left_vals = (
                set(val for val in left_vals) if lower_limit_left is None else None
            )
            upper_limit_left = max(left_vals) if left_vals else upper_limit_left
            lower_limit_left = min(left_vals) if left_vals else lower_limit_left

            change = (
                self.terms[leftIdx].topdown(
                    IntegerVariable(
                        upper_limit_left, lower_limit_left, values=left_vals
                    ),
                    sub_vars,
                )
                or change
            )

        return change

    def __str__(self):
        """
        >>> str(ProductTerm(VariableTerm("a"), VariableTerm("5")))
        '( a * 5 )'
        """
        return "( " + str(self.terms[0]) + " * " + str(self.terms[1]) + " )"

    def __repr__(self):
        return "ProductTerm(" + repr(self.terms[0]) + ", " + repr(self.terms[1]) + ")"


@dataclass
class SumTerm(RecursiveTerm):
    def __init__(self, termA: RecursiveTerm, termB: RecursiveTerm):
        super().__init__([termA, termB])

    def order(self, variables: VariablesDict) -> int:
        return 20 * self.terms[0].order(variables) * self.terms[1].order(variables)

    def bottomup(self, sub_vars: VariablesDict):
        """
        >>> SumTerm(VariableTerm("a"), VariableTerm("b")).bottomup(VariablesDict({"a": IntegerVariable(5), "b": IntegerVariable(3)}))
        IntegerVariable(8, 2)
        """
        # if self._check_empty(sub_vars):
        #     return IntegerVariable(values=())

        left = self.terms[0].bottomup(sub_vars)
        right = self.terms[1].bottomup(sub_vars)
        factor = gcd(left.factor, right.factor)

        if left.whole_range() and right.whole_range():
            return IntegerVariable(
                left.upper_limit + right.upper_limit,
                left.lower_limit + right.lower_limit,
                factor=factor,
            )
        elif left.whole_range():
            if right.values == ():
                return IntegerVariable(values=())
            if left.upper_limit - left.lower_limit < EXPANSION_LIMIT:
                return IntegerVariable(
                    left.upper_limit + right.upper_limit,
                    left.lower_limit + right.lower_limit,
                    tuple(
                        sorted(
                            val + n
                            for val in right.values
                            for n in range(left.lower_limit, left.upper_limit + 1)
                        )
                    ),
                    factor=factor,
                )
            else:
                return IntegerVariable(
                    left.upper_limit + right.upper_limit,
                    left.lower_limit + right.lower_limit,
                    factor=factor,
                )
        elif right.whole_range():
            if left.values == ():
                return IntegerVariable(values=())
            if right.upper_limit - right.lower_limit < EXPANSION_LIMIT:
                return IntegerVariable(
                    left.upper_limit + right.upper_limit,
                    left.lower_limit + right.lower_limit,
                    tuple(
                        sorted(
                            val + n
                            for val in left.values
                            for n in range(right.lower_limit, right.upper_limit + 1)
                        )
                    ),
                    factor=factor,
                )
            else:
                return IntegerVariable(
                    left.upper_limit + right.upper_limit,
                    left.lower_limit + right.lower_limit,
                    factor=factor,
                )
        else:
            return IntegerVariable(
                left.upper_limit + right.upper_limit,
                left.lower_limit + right.lower_limit,
                tuple(set(valL + valR for valL in left.values for valR in right.values))
                if (left.size() * right.size() < EXPANSION_LIMIT)
                else None,
                factor=factor,
            )

    def topdown(self, res_var: IntegerVariable, sub_vars: VariablesDict):
        """
        >>> var = VariablesDict({"a": IntegerVariable(20, 5), "b": IntegerVariable(20)})
        >>> SumTerm(VariableTerm("a"), VariableTerm("b")).topdown(IntegerVariable(values=[15]), var)
        True
        >>> var["a"]
        IntegerVariable(14, 5)
        >>> var["b"]
        IntegerVariable(10, 1)
        """
        # if self._check_empty(sub_vars, res_var):
        #     return False

        # choose the easier part first to track down the other part
        if self.terms[0].order(sub_vars) < self.terms[1].order(sub_vars):
            leftIdx = 0
        else:
            leftIdx = 1

        left = self.terms[leftIdx].bottomup(sub_vars)

        if res_var.whole_range():
            upper_limit_right = min(
                res_var.upper_limit - left.lower_limit, INT_UPPER_LIMIT
            )
            lower_limit_right = max(0, res_var.lower_limit - left.upper_limit)
            resR = self.terms[1 - leftIdx].topdown(
                IntegerVariable(upper_limit_right, lower_limit_right), sub_vars
            )

            right = self.terms[1 - leftIdx].bottomup(sub_vars)
            upper_limit_left = min(
                res_var.upper_limit - right.lower_limit, INT_UPPER_LIMIT
            )
            lower_limit_left = max(0, res_var.lower_limit - right.upper_limit)
            resL = self.terms[leftIdx].topdown(
                IntegerVariable(upper_limit_left, lower_limit_left), sub_vars
            )

        else:
            right_vals = []
            lower_limit_right = INT_UPPER_LIMIT  # right.upper_limit
            upper_limit_right = 0  # right.lower_limit
            for val in res_var.values:
                if left.whole_range():
                    lower_limit_right = max(
                        0, min(lower_limit_right, val - left.upper_limit)
                    )
                    upper_limit_right = min(
                        max(upper_limit_right, val - left.lower_limit), INT_UPPER_LIMIT
                    )
                    right_vals = None
                else:
                    right_vals += [val - lval for lval in left.values if val - lval > 0]

            resR = self.terms[1 - leftIdx].topdown(
                IntegerVariable(
                    upper_limit_right,
                    lower_limit_right,
                    sorted(right_vals) if right_vals is not None else None,
                ),
                sub_vars,
            )

            right = self.terms[1 - leftIdx].bottomup(sub_vars)

            left_vals = []
            lower_limit_left = INT_UPPER_LIMIT  # left.upper_limit
            upper_limit_left = 0  # left.lower_limit
            for val in res_var.values:
                if right.whole_range():
                    lower_limit_left = max(
                        0, min(lower_limit_left, val - right.upper_limit)
                    )
                    upper_limit_left = min(
                        max(upper_limit_left, val - right.lower_limit), INT_UPPER_LIMIT
                    )
                    left_vals = None
                else:
                    left_vals += [val - rval for rval in right.values if val - rval > 0]

            resL = self.terms[leftIdx].topdown(
                IntegerVariable(
                    upper_limit_left,
                    lower_limit_left,
                    sorted(left_vals) if left_vals is not None else None,
                ),
                sub_vars,
            )

        return resL or resR

    def __str__(self):
        """
        >>> str(SumTerm(VariableTerm("a"), VariableTerm("5")))
        '( a + 5 )'
        """
        return "( " + str(self.terms[0]) + " + " + str(self.terms[1]) + " )"

    def __repr__(self):
        return "SumTerm(" + repr(self.terms[0]) + ", " + repr(self.terms[1]) + ")"


@dataclass
class Constraint:
    terms: Union[Sequence[RecursiveTerm], Sequence["Constraint"]]

    def order(self, variables: VariablesDict) -> int:
        order_val = 30
        for term in self.terms:
            order_val *= term.order(variables)
        return order_val

    def _check_empty(self, variables: VariablesDict) -> bool:
        return variables.empty()

    @abstractmethod
    def apply(variables: VariablesDict) -> bool:
        """
        Enforce a Constraint on variables of a VariablesDict

        Returns True if any variable changed.
        """
        pass


@dataclass
class EqualConstraint(Constraint):
    def __init__(self, left: RecursiveTerm, right: RecursiveTerm):
        self.terms = [left, right]

    def order(self, variables: VariablesDict) -> int:
        return 2 * min(self.terms[0].order(variables), self.terms[1].order(variables))

    def apply(self, variables: VariablesDict) -> bool:
        """
        >>> var = VariablesDict({"a": IntegerVariable([10]), "b": IntegerVariable([5]), "c": IntegerVariable()})
        >>> EqualConstraint(SumTerm(VariableTerm("a"), VariableTerm("b")), VariableTerm("c")).apply(var)
        True
        >>> var["c"]
        IntegerVariable(values=(15,))

        """
        # if self._check_empty(variables):
        #     return False
        change = True
        any_change = False
        while change:
            change = False
            if self.terms[0].order(variables) < self.terms[1].order(variables):
                left_var = self.terms[0].bottomup(variables)
                change = self.terms[1].topdown(left_var, variables) or change
                right_var = self.terms[1].bottomup(variables)
                change = self.terms[0].topdown(right_var, variables) or change
            else:
                right_var = self.terms[1].bottomup(variables)
                change = self.terms[0].topdown(right_var, variables) or change
                left_var = self.terms[0].bottomup(variables)
                change = self.terms[1].topdown(left_var, variables) or change
            if change:
                any_change = True
        return any_change

    def __str__(self):
        """
        >>> str(EqualConstraint(VariableTerm("a"), VariableTerm("5")))
        '( a == 5 )'
        """

        return "( " + str(self.terms[0]) + " == " + str(self.terms[1]) + " )"

    def __repr__(self):
        return (
            "EqualConstraint(" + repr(self.terms[0]) + ", " + repr(self.terms[1]) + ")"
        )


@dataclass
class AssignmentConstraint(EqualConstraint):
    def __init__(self, left: VariableTerm, right: VariableTerm):
        super().__init__(left, right)


@dataclass
class ConstantAssignmentConstraint(EqualConstraint):
    def __init__(self, left: VariableTerm, right: VariableTerm):
        if not (left.constant or right.constant):
            raise ValueError(f"Bad constant assignment constraint {left} == {right}")
        super().__init__(left, right)

    def order(self, variables: VariablesDict) -> int:
        return 0


@dataclass
class LessEqualConstraint(Constraint):
    def __init__(self, left: RecursiveTerm, right: RecursiveTerm):
        self.terms = [left, right]

    def order(self, variables: VariablesDict) -> int:
        return min(2 * self.terms[0].order(variables), self.terms[1].order(variables))

    def apply(self, variables: VariablesDict) -> bool:
        """
        >>> var = VariablesDict({"a": IntegerVariable([10]), "b": IntegerVariable(8), "c": IntegerVariable([15])})
        >>> LessEqualConstraint(SumTerm(VariableTerm("a"), VariableTerm("b")), VariableTerm("c")).apply(var)
        True
        >>> var["b"]
        IntegerVariable(5, 1)
        >>> var = VariablesDict({"a": IntegerVariable([10]), "b": IntegerVariable(200), "c": IntegerVariable([30])})
        >>> LessEqualConstraint(ProductTerm(VariableTerm("a"), VariableTerm("b")), VariableTerm("c")).apply(var)
        True
        >>> var["b"]
        IntegerVariable(3, 1)
        """
        # if self._check_empty(variables):
        #     return False
        change = True
        any_change = False
        while change:
            change = False
            # choose order by order of terms
            if self.terms[0].order(variables) < self.terms[1].order(variables):
                left_var = self.terms[0].bottomup(variables)
                left_var = IntegerVariable(lower_limit=left_var.lower_limit)
                change = self.terms[1].topdown(left_var, variables) or change
                right_var = self.terms[1].bottomup(variables)
                right_var = IntegerVariable(upper_limit=right_var.upper_limit)
                change = self.terms[0].topdown(right_var, variables) or change
            else:
                right_var = self.terms[1].bottomup(variables)
                right_var = IntegerVariable(upper_limit=right_var.upper_limit)
                change = self.terms[0].topdown(right_var, variables) or change
                left_var = self.terms[0].bottomup(variables)
                left_var = IntegerVariable(lower_limit=left_var.lower_limit)
                change = self.terms[1].topdown(left_var, variables) or change

            if change:
                any_change = True
        return any_change

    def __str__(self):
        """
        >>> str(LessEqualConstraint(VariableTerm("a"), VariableTerm("5")))
        '( a <= 5 )'
        """
        return "( " + str(self.terms[0]) + " <= " + str(self.terms[1]) + " )"

    def __repr__(self):
        return (
            "LessEqualConstraint("
            + repr(self.terms[0])
            + ", "
            + repr(self.terms[1])
            + ")"
        )


@dataclass
class DivisibleConstraint(Constraint):
    def __init__(self, dividend: RecursiveTerm, divisor: RecursiveTerm):
        self.terms = [dividend, divisor]

    def order(self, variables: VariablesDict) -> int:
        return min(self.terms[0].order(variables), 2 * self.terms[1].order(variables))

    def apply(self, variables: VariablesDict) -> bool:
        """
        >>> var = VariablesDict({"a": IntegerVariable(5), "b": IntegerVariable([2])})
        >>> DivisibleConstraint(VariableTerm("a"), VariableTerm("b")).apply(var)
        True
        >>> var["a"]
        IntegerVariable(values=(2, 4))
        >>> var = VariablesDict({"a": IntegerVariable([512]), "b": IntegerVariable()})
        >>> DivisibleConstraint(VariableTerm("a"), VariableTerm("b")).apply(var)
        True
        >>> var["b"]
        IntegerVariable(values=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512))
        """
        # if self._check_empty(variables):
        #     return False

        change = True
        any_change = False
        help_var = "_" + str(uuid.uuid1()).replace("-", "") + "_"
        while change:
            change = False
            dividend_var = self.terms[0].bottomup(variables)
            change = (
                self.terms[1].topdown(
                    IntegerVariable(dividend_var.upper_limit, 1), variables
                )
                or change
            )
            if dividend_var.sole_prime_factors is not None:
                change = (
                    self.terms[1].topdown(
                        IntegerVariable(
                            sole_prime_factors=dividend_var.sole_prime_factors
                        ),
                        variables,
                    )
                    or change
                )
            divisor_var = self.terms[1].bottomup(variables)
            if divisor_var.factor != 1:
                change = (
                    self.terms[0].topdown(
                        IntegerVariable(factor=divisor_var.factor), variables
                    )
                    or change
                )
            if (
                divisor_var.size() > EXPANSION_LIMIT
                or dividend_var.size() > EXPANSION_LIMIT
            ):
                if change:
                    self.terms[1].topdown(
                        IntegerVariable(dividend_var.upper_limit, 1), variables
                    )
                    any_change = True
                break
            divisor_values = divisor_var.all_values()
            first_val = True
            variables_copy = variables.copy()

            # skip if 1 in divisor values as it won't change anything
            if 1 in divisor_values:
                return change

            # constraint satisfied anyways, so skip
            if not dividend_var.whole_range() and all(
                all((divd % divs == 0) for divd in dividend_var.values)
                for divs in divisor_values
            ):
                return any_change or change

            for val in divisor_values:
                for n in range(1, dividend_var.upper_limit // val + 1):
                    if dividend_var.values is None or n * val in dividend_var.values:
                        eq = EqualConstraint(
                            ProductTerm(VariableTerm(help_var), self.terms[1]),
                            self.terms[0],
                        )
                        if first_val:
                            variables_copy.variables[help_var] = IntegerVariable([n])
                            eq.apply(variables_copy)
                        else:
                            variables_copy2 = variables.copy()
                            variables_copy2.variables[help_var] = IntegerVariable([n])
                            eq.apply(variables_copy2)
                            variables_copy.unify(variables_copy2)
                        first_val = False
            change = variables.intersect(variables_copy) or change
            if change:
                any_change = True
        return any_change

    def __str__(self):
        """
        >>> str(DivisibleConstraint(VariableTerm("a"), VariableTerm("5")))
        '( a % 5 == 0 )'
        """
        return "( " + str(self.terms[0]) + " % " + str(self.terms[1]) + " == 0 )"

    def __repr__(self):
        return (
            "DivisibleConstraint("
            + repr(self.terms[0])
            + ", "
            + repr(self.terms[1])
            + ")"
        )


@dataclass
class AndConstraint(Constraint):
    def __init__(self, constraints: Sequence[Constraint]):
        self.terms = tuple(constraints)

    def order(self, variables: VariablesDict) -> int:
        return min(term.order(variables) for term in self.terms)

    def apply(self, variables: VariablesDict) -> bool:
        """
        >>> var = VariablesDict({"a": IntegerVariable(5), "b": IntegerVariable(5), "5": IntegerVariable([5]), "2": IntegerVariable([2])})
        >>> constr = AndConstraint((
        ...     LessEqualConstraint(SumTerm(VariableTerm("a"), VariableTerm("b")), VariableTerm("5")),
        ...     LessEqualConstraint(SumTerm(VariableTerm("a"), VariableTerm("b")), VariableTerm("2"))))
        >>> constr.apply(var)
        True
        >>> var["a"]
        IntegerVariable(1, 1)
        >>> var = VariablesDict({"a": IntegerVariable(), "3": IntegerVariable([3]), "5": IntegerVariable([5])})
        >>> AndConstraint((
        ...     EqualConstraint(VariableTerm("a"), VariableTerm("5")),
        ...     EqualConstraint(VariableTerm("a"), VariableTerm("3")))).apply(var)
        True
        >>> var["a"]
        IntegerVariable(values=())
        """
        # if self._check_empty(variables):
        #     return False

        change = True
        any_change = False
        while change:
            change = False
            terms = sorted(self.terms, key=lambda term: term.order(variables))
            for n, constr in enumerate(terms):
                # skip constant assignments after first round
                if not (
                    any_change and isinstance(constr, ConstantAssignmentConstraint)
                ):
                    change = constr.apply(variables) or change
                    if change and not isinstance(constr, ConstantAssignmentConstraint):
                        any_change = True
                        break
            if change:
                any_change = True
        return any_change

    def __str__(self):
        """
        >>> str(AndConstraint((
        ...     EqualConstraint(VariableTerm("a"), VariableTerm("5")),
        ...     EqualConstraint(VariableTerm("a"), VariableTerm("3")))))
        '(\\n\\t( a == 5 );\\n\\t( a == 3 )\\n)'
        """
        return "(\n\t" + ";\n\t".join(map(str, self.terms)) + "\n)"

    def __repr__(self):
        return "AndConstraint(" + ", ".join(repr(term) for term in self.terms) + ")"


@dataclass
class OrConstraint(Constraint):
    _order: int = 999

    def __init__(self, constraints: Sequence[Constraint]):
        self.terms = tuple(constraints)

    def order(self, variables: VariablesDict) -> int:
        return sum(term.order(variables) for term in self.terms)

    def apply(self, variables: VariablesDict) -> bool:
        """
        >>> var = VariablesDict({"a": IntegerVariable(5), "b": IntegerVariable(5), "5": IntegerVariable([5]), "2": IntegerVariable([2])})
        >>> constr = OrConstraint((
        ...     LessEqualConstraint(SumTerm(VariableTerm("a"), VariableTerm("b")), VariableTerm("5")),
        ...     LessEqualConstraint(SumTerm(VariableTerm("a"), VariableTerm("b")), VariableTerm("2"))))
        >>> constr.apply(var)
        True
        >>> var["a"]
        IntegerVariable(4, 1)
        >>> var = VariablesDict({"a": IntegerVariable(), "3": IntegerVariable([3]), "5": IntegerVariable([5])})
        >>> OrConstraint((
        ...     EqualConstraint(VariableTerm("a"), VariableTerm("5")),
        ...     EqualConstraint(VariableTerm("a"), VariableTerm("3")))).apply(var)
        True
        >>> var["a"]
        IntegerVariable(values=(3, 5))
        """
        # if self._check_empty(variables):
        #     return False

        change = True
        any_change = False
        while change:
            change = False
            first_appl = True
            variables_copy = variables.copy()
            for constr in self.terms:
                if first_appl:
                    constr.apply(variables_copy)
                else:
                    variables_copy2 = variables.copy()
                    constr.apply(variables_copy2)
                    variables_copy.unify(variables_copy2)
                first_appl = False
            change = variables.intersect(variables_copy)
            if change:
                any_change = True
        return any_change

    def __str__(self):
        """
        >>> str(OrConstraint((
        ...     EqualConstraint(VariableTerm("a"), VariableTerm("5")),
        ...     EqualConstraint(VariableTerm("a"), VariableTerm("3")))))
        '(\\n\\t( a == 5 ) ||\\n\\t( a == 3 )\\n)'
        """
        return "(\n\t" + " ||\n\t".join(map(str, self.terms)) + "\n)"

    def __repr__(self):
        return "OrConstraint(" + ", ".join(repr(term) for term in self.terms) + ")"


def parse_expression(expression: str) -> tuple[Union[Constraint, Term], VariablesDict]:
    """
    This does NOT support brackets yet, they can be added for a nice look, but it will always be parsed in the
    following order:
    - lines with comma: AND constraints
    - lines with ||: OR constraints
    - less equal <= line
    - mod == 0 line
    - equal == line
    - sum +
    - product *
    - number_list like [1, 2, 3]
    - variables / numbers

    >>> constr, var = parse_expression("5")
    >>> str(constr)
    '5'
    >>> "5" in var
    True
    >>> var['5']
    IntegerVariable(values=(5,))
    >>> constr, var = parse_expression("5 * x == 10")
    >>> str(constr)
    '( ( 5 * x ) == 10 )'
    >>> "x" in var
    True
    >>> var["x"]
    IntegerVariable()
    >>> constr.apply(var)
    True
    >>> var["x"]
    IntegerVariable(values=(2,))
    >>> constr, var = parse_expression("x * 5 == 10")
    >>> str(constr)
    '( ( x * 5 ) == 10 )'
    >>> "x" in var
    True
    >>> var["x"]
    IntegerVariable()
    >>> constr.apply(var)
    True
    >>> var["x"]
    IntegerVariable(values=(2,))
    >>> constr, var = parse_expression("5 * x <= 15; x + 1 <= 3")
    >>> str(constr) is not None
    True
    >>> constr.apply(var)
    True
    >>> var["x"] == IntegerVariable(values=(1, 2))
    True
    >>> constr, var = parse_expression("x == [1, 2, 3]")
    >>> constr.apply(var)
    True
    >>> var["x"]
    IntegerVariable(values=(1, 2, 3))
    >>> constr, var = parse_expression("x == [1, 2, 3] # This is a comment")
    >>> constr.apply(var)
    True
    >>> var["x"]
    IntegerVariable(values=(1, 2, 3))
    >>> constr, var = parse_expression("x ^ 2 == 9  # Another comment")
    >>> constr.apply(var)
    True
    >>> var["x"]
    IntegerVariable(values=(3,))
    """
    variables_dict = VariablesDict({})
    constraint = parse_expression_(expression, variables_dict)
    return constraint, variables_dict


def parse_expression_(
    expression: str, variables_dict: VariablesDict, recursion_limit=20
) -> Union[Constraint, Term]:
    lines = expression.split("\n")
    expression = "\n".join(
        line.strip().split("#")[0] for line in lines if line.strip().split("#")[0]
    )

    rl = recursion_limit - 1

    if rl < 0:
        raise ValueError(f"Recursion Limit reached at {expression}")

    and_constr_str = expression.split(";")
    if len(and_constr_str) > 1:
        and_constr_str = [
            st.split("#")[0]
            for st in and_constr_str
            if st.strip(" \t()\n").split("#")[0]
        ]
        return AndConstraint(
            [parse_expression_(constr, variables_dict, rl) for constr in and_constr_str]
        )

    or_constr_str = expression.split("|")
    if len(or_constr_str) > 1:
        or_constr_str = [
            st.split("#")[0]
            for st in or_constr_str
            if st.strip(" \t()\n").split("#")[0]
        ]
        return OrConstraint(
            [parse_expression_(constr, variables_dict, rl) for constr in or_constr_str]
        )

    expression = expression.split("#")[0]

    lessequal_str = expression.split("<=")
    if len(lessequal_str) > 1:
        lessequal_str = [
            st.split("#")[0] for st in lessequal_str if st.strip(" \t()\n")
        ]
        if len(lessequal_str) != 2:
            return ValueError(f"Parsing Error at LessEqual of {lessequal_str}")
        left = parse_expression_(lessequal_str[0], variables_dict, rl)
        right = parse_expression_(lessequal_str[1], variables_dict, rl)
        if isinstance(left, Term) and isinstance(right, Term):
            return LessEqualConstraint(left, right)
        else:
            raise ValueError(f"Parsing Error at LessEqual of {lessequal_str}")

    divisible = "== 0" in expression and "%" in expression
    if divisible:
        divisible_str = expression.replace("== 0", "").split("%")
        divisible_str = [
            st.split("#")[0] for st in divisible_str if st.strip(" \t()\n")
        ]
        if len(divisible_str) != 2:
            return ValueError(f"Parsing Error at Equal of {divisible_str}")
        left = parse_expression_(divisible_str[0], variables_dict, rl)
        right = parse_expression_(divisible_str[1], variables_dict, rl)
        if isinstance(left, Term) and isinstance(right, Term):
            return DivisibleConstraint(left, right)
        else:
            raise ValueError(f"Parsing Error at Equal of {divisible_str}")

    equal_str = expression.split("==")
    if len(equal_str) > 1:
        equal_str = [st.split("#")[0] for st in equal_str if st.strip(" \t()\n")]
        if len(equal_str) != 2:
            return ValueError(f"Parsing Error at Equal of {equal_str}")
        left = parse_expression_(equal_str[0], variables_dict, rl)
        right = parse_expression_(equal_str[1], variables_dict, rl)
        if isinstance(left, VariableTerm) and isinstance(right, VariableTerm):
            if left.constant or right.constant:
                return ConstantAssignmentConstraint(left, right)
            else:
                return AssignmentConstraint(left, right)
        elif isinstance(left, Term) and isinstance(right, Term):
            return EqualConstraint(left, right)
        else:
            raise ValueError(f"Parsing Error at Equal of {equal_str}")

    sum_str = expression.split("+")
    if len(sum_str) > 1:
        sum_str = [st.split("#")[0] for st in sum_str if st.strip(" \t()\n")]
        terms = [parse_expression_(st, variables_dict, rl) for st in sum_str]
        if len(terms) < 2 or not all(map(lambda t: isinstance(t, Term), terms)):
            raise ValueError(f"Parsing Error at Sum of {sum_str}")
        res = SumTerm(terms[0], terms[1])
        for term in terms[2:]:
            res = SumTerm(res, term)
        return res

    prod_str = expression.split("*")
    if len(prod_str) > 1:
        sum_str = [st.split("#")[0] for st in sum_str if st.strip(" \t()\n")]
        terms = [parse_expression_(st, variables_dict, rl) for st in prod_str]
        if len(terms) < 2 or not all(map(lambda t: isinstance(t, Term), terms)):
            raise ValueError(f"Parsing Error at Prod of {prod_str}")
        res = ProductTerm(terms[0], terms[1])
        for term in terms[2:]:
            res = ProductTerm(res, term)
        return res

    power_str = expression.split("^")
    if len(power_str) > 1:
        power_str = [
            st.split("#")[0].strip(" \t()\n") for st in power_str if st.strip(" \t()\n")
        ]
        if len(power_str) != 2 or not re.match(r"\d+", power_str[1]):
            raise ValueError(f"Parsing Error at Power of {power_str}")
        res = PowerTerm(
            parse_expression_(power_str[0], variables_dict, rl), int(power_str[1])
        )
        return res

    expression = expression.strip(" \t()\n")

    if re.match(r"\[\s*(\d+\s*)(,(\s*\d+\s*))*\]", expression):
        if expression not in variables_dict:
            variables_dict[expression] = IntegerVariable(
                [int(exp.strip("[] ")) for exp in expression.split(",")], constant=True
            )
        return VariableTerm(expression)

    if re.match(r"\d+", expression):
        if expression not in variables_dict:
            variables_dict[expression] = IntegerVariable(
                [int(expression)], constant=True
            )
        return VariableTerm(expression, constant=True)

    if re.match(r"[a-zA-Z_]+", expression):
        if expression not in variables_dict:
            variables_dict[expression] = IntegerVariable()
        return VariableTerm(expression)

    raise ValueError(f"Parsing error: {expression}")


class ValueHeuristic(Enum):
    SMALLEST_FIRST = 0
    LARGEST_FIRST = 1


@dataclass
class ValueRefinement:
    name: str
    heuristic: ValueHeuristic = field(
        default_factory=lambda: ValueHeuristic.SMALLEST_FIRST
    )


@cache_decorator(
    os.getenv("CONSTRINT_CACHE_DIR", os.getenv("HOME") + "/.cache/constrint")
)
def solve_constrint(
    constr: Constraint,
    var_base: VariablesDict,
    resulting_vars: Sequence[ValueRefinement],
    num_solutions: int = 1,
    exclude_solutions: Sequence[dict[str, int]] = (),
) -> Sequence[dict]:
    """
    >>> vr = (ValueRefinement("A"), ValueRefinement("B"))
    >>> constr, var = parse_expression("A == 1; B == 2;")
    >>> solve_constrint(constr, var, resulting_vars=vr)
    ({'A': 1, 'B': 2},)
    >>> constr, var = parse_expression("A == 1; B == 2;")
    >>> solve_constrint(constr, var, resulting_vars=vr, num_solutions=2)
    ({'A': 1, 'B': 2},)
    >>> constr, var = parse_expression("A == [1, 2]; B == [3, 4];")
    >>> solve_constrint(constr, var, resulting_vars=vr)
    ({'A': 1, 'B': 3},)
    >>> constr, var = parse_expression("A == [1, 2]; B == [3, 4];")
    >>> solve_constrint(constr, var, resulting_vars=vr, num_solutions=4)
    ({'A': 1, 'B': 3}, {'A': 1, 'B': 4}, {'A': 2, 'B': 3}, {'A': 2, 'B': 4})
    >>> constr, var = parse_expression("A == [1, 2]; B == [3, 4];")
    >>> solve_constrint(constr, var, resulting_vars=vr, num_solutions=3)
    ({'A': 1, 'B': 3}, {'A': 1, 'B': 4}, {'A': 2, 'B': 3})
    >>> constr, var = parse_expression("A == [1, 2]; B == [3, 4];")
    >>> solve_constrint(constr, var, resulting_vars=vr, num_solutions=5)
    ({'A': 1, 'B': 3}, {'A': 1, 'B': 4}, {'A': 2, 'B': 3}, {'A': 2, 'B': 4})
    >>> constr, var = parse_expression("A == [1, 2]; B == [3, 4];")
    >>> solve_constrint(constr, var, resulting_vars=vr, exclude_solutions=({'A': 1, 'B': 3},))
    ({'A': 1, 'B': 4},)
    """
    constr.apply(var_base)
    LOGGER.debug("Constrint: Arc consistency satisfied")
    previous_choices = [0 for _ in range(len(resulting_vars) - 1)] + [-1]
    previous_choice_level = len(resulting_vars) - 1
    current_choices = [0 for _ in range(len(resulting_vars))]
    solutions = []

    if var_base.empty():
        return ()

    def variable_subset(var: VariablesDict, pars: Sequence[ValueRefinement]):
        return {par.name: var[par.name].all_values()[0] for par in pars}

    if var_base.single_solution():
        return (variable_subset(var_base, resulting_vars),)

    # an array that contains constrained solutions up to n-th variable excluded (n array index)
    var = [var_base.copy()]

    resulting_vars = [par for par in resulting_vars if par.name in var_base]

    while previous_choice_level >= 0:
        all_chosen = True
        current_choices = [0 for _ in range(len(resulting_vars))]

        for var_num, par in enumerate(resulting_vars):
            choice = previous_choices[var_num]

            if var_num + 1 == len(var):
                var.append(var[var_num].copy())
            elif var_num > len(var):
                raise ValueError
            else:
                current_choices[var_num] = choice
                continue

            var_choices = sorted(var[var_num + 1][par.name].all_values())
            if par.heuristic == ValueHeuristic.LARGEST_FIRST:
                var_choices = list(reversed(var_choices))
                # if current variable is a last value
            if (
                previous_choice_level == var_num and choice + 1 >= len(var_choices)
            ) or choice >= len(var_choices):
                # go back to upper variable (larger precedence for value setting)
                previous_choice_level = var_num - 1
                var = var[: previous_choice_level + 1]
                # reset this and lower variables to a first value
                for i in range(var_num, len(resulting_vars)):
                    previous_choices[i] = 0
                previous_choices[-1] = -1
                all_chosen = False
                break
            if previous_choice_level == var_num:
                choice = choice + 1
                if previous_choice_level != len(previous_choices) - 1:
                    for i in range(var_num + 1, len(previous_choices)):
                        previous_choices[i] = 0
                    previous_choices[-1] = -1
                    previous_choice_level = len(previous_choices) - 1
            current_choices[var_num] = choice
            previous_choice_level = max(previous_choice_level, var_num)

            var[var_num + 1][par.name].intersect(
                IntegerVariable(values=(var_choices[choice],))
            )
            constr.apply(var[var_num + 1])

            if var[var_num + 1].empty():
                for i in range(var_num + 1):
                    previous_choices[i] = current_choices[i]
                for i in range(var_num + 1, len(previous_choices)):
                    previous_choices[i] = 0
                previous_choice_level = var_num
                var.pop()
                all_chosen = False
                break
        if all_chosen:
            if not var[-1].single_solution():
                LOGGER.debug("Some background variables remain arbitrary")
            if not var[-1].empty():
                sol = variable_subset(var[-1], resulting_vars)
                if sol not in exclude_solutions:
                    solutions.append(sol)
            if len(solutions) >= num_solutions:
                break
            previous_choices = current_choices
    LOGGER.debug("Constrint solutions ready")
    return tuple(solutions)


if __name__ == "__main__":
    # # VariableTerm("a").topdown(IntegerVariable(5), sub_vars=VariablesDict({"a": IntegerVariable(5, 1, [1,3,5])}))
    # var = VariablesDict({"a": IntegerVariable(20), "b": IntegerVariable(20)})
    # ProductTerm(VariableTerm("a"), VariableTerm("b")).topdown(IntegerVariable(values=[15]), var)

    # var = VariablesDict({"a": IntegerVariable([10]), "b": IntegerVariable(200), "c": IntegerVariable([30])})
    # LessEqualConstraint(ProductTerm(VariableTerm("a"), VariableTerm("b")), VariableTerm("c")).apply(var)

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as fp:
            file = fp.read()

        constr, var = parse_expression(file)

        print(str(constr))
        print(var)

        constr.apply(var)

        print("Possible Values")
        print(var)

        if var.empty():
            print("NO PROPER CONFIGURATION FOUND!")

        if var.single_solution():
            print("It already is a Single Solution")

        sol = solve_constrint(
            constr, var, [ValueRefinement(st) for st in var], num_solutions=1
        )

        print(f"Solution: {sol}")
    else:
        print("Doctests for CONSTRINT\n\n")
        doctest.testmod(verbose=True)
