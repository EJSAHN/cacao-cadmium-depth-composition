from fractions import Fraction
from itertools import product
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from code5_stats import (combine, linear_box, endpoint_values, interval_sign,
                         run_code5_analysis, rounding_cells)
from depth_core import read_source


class RoundingAlgebraTests(unittest.TestCase):
    def test_combines_shared_cells_before_propagating_selection(self):
        kept = {1: Fraction(1, 2), 2: Fraction(1, 2)}
        all_sites = {1: Fraction(1, 3), 2: Fraction(1, 3), 3: Fraction(1, 3)}
        combined = combine((1, kept), (-1, all_sites))
        self.assertEqual(combined, {1: Fraction(1, 6), 2: Fraction(1, 6), 3: Fraction(-1, 3)})
        box = linear_box(combined, {1: 1., 2: 2., 3: 4.}, {1: .05, 2: .05, 3: .05})
        self.assertAlmostEqual(box['radius'], .05 * 2 / 3)
        self.assertNotAlmostEqual(box['radius'], .1)
        canceled = combine((1, all_sites), (-1, all_sites))
        self.assertEqual(canceled, {})
        self.assertEqual(linear_box(canceled, {}, {})['radius'], 0)

    def test_bounds_equal_independent_enumeration_of_every_rounding_corner(self):
        coefficients = {1: Fraction(2, 3), 2: Fraction(-3, 7), 3: Fraction(1, 6), 4: Fraction(-5, 8)}
        values = {1: 5., 2: 3., 3: 1., 4: 9.}
        half_steps = {1: .05, 2: .005, 3: .05, 4: .005}
        answers = []
        for directions in product((-1, 1), repeat=4):
            corner = {i: values[i] + directions[i-1] * half_steps[i] for i in range(1, 5)}
            answers.append(sum(float(coefficients[i]) * corner[i] for i in range(1, 5)))
        box = linear_box(coefficients, values, half_steps)
        self.assertAlmostEqual(box['lower'], min(answers))
        self.assertAlmostEqual(box['upper'], max(answers))
        for endpoint in ('lower', 'upper'):
            corner = endpoint_values(coefficients, values, half_steps, endpoint)
            self.assertAlmostEqual(sum(float(c) * corner[i] for i, c in coefficients.items()), box[endpoint])

    def test_zero_contact_is_not_strictly_robust(self):
        self.assertEqual(interval_sign(-1, 0), 'includes_zero')
        self.assertEqual(interval_sign(0, 1), 'includes_zero')
        self.assertEqual(interval_sign(0, 0), 'zero')
        self.assertEqual(interval_sign(.1, 1), 'positive')
        self.assertEqual(interval_sign(-1, -.1), 'negative')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            linear_box({1: Fraction(1)}, {}, {})
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            linear_box({1: Fraction(1)}, {1: 1}, {1: -.1})



if __name__ == '__main__': unittest.main()
