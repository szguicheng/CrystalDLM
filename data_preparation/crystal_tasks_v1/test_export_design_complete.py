import unittest

from data_preparation.crystal_tasks_v1.export_design_complete import summarize_lengths


class LengthSummaryTests(unittest.TestCase):
    def test_reports_exact_distribution_and_cutoff_exceedance(self):
        summary = summarize_lengths(range(1, 101), cutoff=99)

        self.assertEqual(summary, {
            "count": 100,
            "min": 1,
            "median": 50.5,
            "p95": 95,
            "p99": 99,
            "max": 100,
            "above_cutoff": 1,
        })

    def test_rejects_empty_distribution(self):
        with self.assertRaisesRegex(ValueError, "at least one length"):
            summarize_lengths([], cutoff=2048)


if __name__ == "__main__":
    unittest.main()
