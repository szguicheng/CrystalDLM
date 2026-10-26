import unittest
from summarize import summarize_groups


class SummaryContract(unittest.TestCase):
    def test_budget_failures_remain_in_denominator_and_first_cost_is_charged_to_both(self):
        # Dropping exhausted groups or counting only the last retry must fail this test.
        first = dict(success=False, round=0, seed=7, failure_categories=['formula_header_mismatch'],
                     timing_seconds={'total': 2.0})
        good = dict(success=True, round=1, seed=8, failure_categories=[], timing_seconds={'total': 3.0})
        bad = dict(success=False, round=1, seed=8, failure_categories=['formula_header_mismatch'],
                   timing_seconds={'total': 4.0})
        last = dict(success=False, round=2, seed=9, failure_categories=['wrong_requested_sg'],
                    timing_seconds={'total': 5.0})
        group = dict(request_id='sg-001', group_index=0, status='complete', arms={
            'independent': dict(success=False, attempts_used=3, attempts=[first, bad, last]),
            'feedback': dict(success=True, attempts_used=2, attempts=[first, good])})
        result = summarize_groups([group], scheduled=2)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['arms']['feedback']['success_within_attempts'], {'1': 0, '2': 1, '3': 1})
        self.assertEqual(result['arms']['feedback']['success_rate_within_3'], .5)
        self.assertEqual(result['arms']['independent']['mean_attempts_completed_groups'], 3)
        self.assertEqual(result['arms']['feedback']['mean_seconds_completed_groups'], 5)
        self.assertEqual(result['paired']['feedback_only'], 1)
        self.assertEqual(result['actual_candidate_generations'], 4)
        self.assertEqual(result['arms']['independent']['repeated_error_transitions'], 1)


if __name__ == '__main__':
    unittest.main()
