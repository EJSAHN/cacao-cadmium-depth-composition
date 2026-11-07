import csv
from decimal import Decimal
from fractions import Fraction
from itertools import product
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from external_stats import (SOURCE_FILE, box, combine, expressions_for,
                            literal_half_step, load_source, run_external_analysis)


class ExternalStatisticsTests(unittest.TestCase):
    def test_shared_cell_cancellation_and_extrema_against_enumerated_corners(self):
        terms = expressions_for(('A', 'B'), ('A',))
        self.assertFalse(combine((1,terms['naive']),(-1,terms['paired']),(-1,terms['selection'])))
        self.assertEqual(terms['selection'], {('A',5): Fraction(1,2), ('B',5): Fraction(-1,2)})
        values = {('A',5):Fraction(1), ('A',20):Fraction(2), ('B',5):Fraction(3), ('B',20):Fraction(4)}
        h = {key: Fraction(i+1,100) for i,key in enumerate(sorted(values))}
        keys = sorted(values)
        for name, expression in terms.items():
            calculated = box(expression, values, h)
            corners = []
            for directions in product((-1,1), repeat=len(keys)):
                corner = {key: values[key]+direction*h[key] for key,direction in zip(keys,directions)}
                corners.append(sum((coefficient*corner[key] for key,coefficient in expression.items()),Fraction(0)))
            self.assertEqual(calculated['lower'], min(corners), name)
            self.assertEqual(calculated['upper'], max(corners), name)

    def test_literal_precision_keeps_ambiguous_short_tokens(self):
        self.assertEqual(literal_half_step('1'), Fraction(1,2))
        self.assertEqual(literal_half_step('1.00'), Fraction(1,200))
        self.assertEqual(literal_half_step('0.5'), Fraction(1,20))
        self.assertEqual(literal_half_step('0.50'), Fraction(1,200))
        with self.assertRaises(ValueError):literal_half_step('NaN')

    def test_real_tables_against_independent_decimal_reference_and_witness(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_external_analysis(PACKAGE,tmp)
            self.assertEqual(summary['masks_including_baselines'],62)
            self.assertEqual([(r['assay'],r['all_opposite_sign_masks_reported']) for r in summary['scope_results']], [('EPA3050B',6),('Mehlich3',7)])
            self.assertFalse(summary['pooled_with_main_analysis'])
            self.assertFalse(summary['same_depth_estimand_as_main'])
            witness = summary['epa_hide_f2_f4_witness']
            self.assertTrue(witness['all_farms_decline_at_reported_values'])
            self.assertTrue(witness['flip_while_paired_retains_baseline_reported'])
            self.assertAlmostEqual(witness['naive_delta_mg_kg'], .0153333333333333, places=14)
            self.assertAlmostEqual(witness['paired_visible_delta_mg_kg'], -.0966666666666667, places=14)
            self.assertAlmostEqual(witness['selection_term_mg_kg'], .112, places=14)
            self.assertFalse(witness['robust_flip_under_literal_token_scenario'])
            self.assertLess(witness['naive_literal_lower_mg_kg'],0)
            self.assertGreater(witness['naive_literal_upper_mg_kg'],0)
            with (Path(tmp)/'code5_external_mask_contrasts.csv').open(newline='') as handle:
                calculated = list(csv.DictReader(handle))
            with (PACKAGE/'data/external/guarin2023_all_masks_decimal_reference.csv').open(newline='') as handle:
                reference = {(r['assay'],r['hidden_farms']):r for r in csv.DictReader(handle)}
            for row in calculated:
                ref = reference[(row['assay'],row['hidden_farms'])]
                for result_key,reference_key in [('naive_delta_mg_kg','naive'),
                    ('paired_visible_delta_mg_kg','paired'),('selection_term_mg_kg','selection')]:
                    self.assertAlmostEqual(float(row[result_key]),float(Decimal(ref[reference_key])),places=14)
                if row['assay']=='EPA3050B':self.assertLess(float(row['paired_visible_delta_mg_kg']),0)

    def test_external_transcription_tampering_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/SOURCE_FILE
            target.parent.mkdir(parents=True)
            shutil.copyfile(PACKAGE/SOURCE_FILE,target)
            self.assertEqual(len(load_source(tmp)),20)
            target.write_bytes(target.read_bytes().replace(b'0.69',b'0.70',1))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):load_source(tmp)

    def test_outputs_repeat_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            first=run_external_analysis(PACKAGE,a);second=run_external_analysis(PACKAGE,b)
            self.assertEqual(first,second)
            for name in first['outputs']:
                self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())

if __name__=='__main__':unittest.main()
