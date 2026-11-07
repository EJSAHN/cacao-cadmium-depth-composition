"""Tests of units, support, missingness, and prespecified depth contrasts."""

import csv
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_core import read_source, profile_table, support_table, contrasts


HEADER = ["Sample.ID", "Study.ID", "SiteName", "Block.Rep", "Country",
          "SampleDepth.min", "SampleDepth.max", "Soil.TotalCd"]


class DepthCoreTests(unittest.TestCase):
    def source(self, rows, header=HEADER):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "source.csv"
        with path.open("w", encoding="cp1252", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return read_source(path)

    def test_duplicate_headers_are_preserved_and_audited(self):
        raw, clean, audit = self.source(
            [["1", "1", "site", "", "X", "0", "10", "1", "a", "b"]],
            HEADER + ["Unused", "Unused"])
        self.assertEqual(list(raw.columns).count("Unused"), 2)
        self.assertEqual(raw.iloc[0, -2:].tolist(), ["a", "b"])
        self.assertEqual(audit["source_header_count"], 10)
        self.assertEqual(audit["source_unique_header_count"], 9)
        self.assertEqual(audit["duplicate_headers"][0]["positions_1based"], [9, 10])

    def test_missing_numeric_rows_excluded_zeros_retained_without_pseudocount(self):
        _, clean, audit = self.source([
            ["1", "1", "A", "", "X", "0", "10", "0"],
            ["2", "1", "A", "", "X", "10", "20", "2"],
            ["3", "1", "B", "", "X", "0", "10", ""],
            ["4", "1", "C", "", "X", "", "10", "3"],
        ])
        self.assertEqual(len(clean), 2)
        self.assertEqual(audit["zero_total_cd_rows_retained"], 1)
        result = contrasts(clean).iloc[0]
        self.assertEqual(result["raw_delta_mg_kg"], 2)
        self.assertTrue(np.isnan(result["log_deep_surface_ratio"]))

    def test_malformed_numbers_and_invalid_intervals_fail(self):
        base = ["1", "1", "A", "", "X", "0", "10", "1"]
        for index, value in [(7, "below detection"), (7, "-1"), (7, "inf"), (5, "10")]:
            row = base.copy()
            row[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                self.source([row])

    def test_study_site_block_keys_do_not_collide(self):
        rows = []
        groups = [("1", "A", ""), ("2", "A", ""), ("1", "A", "1"),
                  ("1", "A", "@MISSING"), ("1|A", "B", ""), ("1", "A|B", "")]
        for index, (study, site, block) in enumerate(groups):
            for layer in range(2):
                rows.append([str(index * 2 + layer), study, site, block, "X", str(layer * 10), str((layer + 1) * 10), "1"])
        _, clean, _ = self.source(rows)
        profile = profile_table(clean)
        self.assertEqual(len(profile), len(groups))
        self.assertEqual(int(profile["eligible_primary"].sum()), len(groups))

    def test_duplicate_or_overlapping_intervals_exclude_primary_not_pooled_rows(self):
        _, clean, audit = self.source([
            ["1", "1", "A", "", "X", "0", "10", "1"],
            ["2", "1", "A", "", "X", "0", "10", "2"],
            ["3", "1", "A", "", "X", "10", "20", "3"],
            ["4", "1", "B", "", "X", "0", "20", "1"],
            ["5", "1", "B", "", "X", "10", "30", "2"],
        ])
        self.assertEqual(len(clean), 5)
        self.assertTrue(contrasts(clean).empty)
        self.assertEqual(audit["series_with_duplicate_intervals"], 1)
        self.assertEqual(audit["series_with_overlapping_intervals"], 1)

    def test_depth_support_conserves_rows_and_counts_boundary_coordinates(self):
        _, clean, _ = self.source([
            ["1", "1", "A", "", "X", "0", "10", "1"],
            ["2", "2", "B", "", "Y", "0", "10", "2"],
            ["3", "2", "B", "", "Y", "10", "20", "3"],
        ])
        support = support_table(clean)
        self.assertEqual(support.groupby("axis")["n_rows"].sum().to_dict(),
                         {"lower": 3, "midpoint": 3, "upper": 3})
        first = support[(support.axis == "upper") & (support.bin_index == 0)].iloc[0]
        self.assertEqual(first.n_rows, 2)
        self.assertAlmostEqual(first.effective_studies, 2)
        self.assertAlmostEqual(first.max_study_fraction, 0.5)
        self.assertEqual(support[(support.axis == "lower") & (support.bin_index == 0)].iloc[0].n_rows, 3)

    def test_geographic_composition_can_raise_pooled_means_despite_within_series_declines(self):
        # A low-background study contributes only shallow observations.
        # The one multi-depth study decreases within its site, while the
        # pooled deep mean exceeds the pooled shallow mean.
        _, clean, _ = self.source([
            ["1", "L", "a", "", "Low", "0", "10", "1"],
            ["2", "L", "b", "", "Low", "0", "10", "1"],
            ["3", "H", "c", "", "High", "0", "10", "10"],
            ["4", "H", "c", "", "High", "10", "20", "9"],
        ])
        means = clean.groupby("depth_max_cm")["total_cd_mg_kg"].mean()
        self.assertGreater(means.loc[20], means.loc[10])
        within = contrasts(clean)
        self.assertEqual(len(within), 1)
        self.assertLess(within.iloc[0].raw_delta_mg_kg, 0)

    def test_interior_anchor_is_selected_by_depth_not_lowest_outcome(self):
        _, clean, _ = self.source([
            ["1", "1", "A", "", "X", "0", "10", "5"],
            ["2", "1", "A", "", "X", "10", "20", "1"],
            ["3", "1", "A", "", "X", "20", "30", "8"],
            ["4", "1", "A", "", "X", "30", "50", "7"],
        ])
        result = contrasts(clean).iloc[0]
        self.assertEqual(result.interior_anchor_upper_cm, 30)
        self.assertEqual(result.interior_anchor_cd_mg_kg, 8)
        self.assertFalse(result.u_pattern_at_depth_anchor)


if __name__ == "__main__":
    unittest.main()
