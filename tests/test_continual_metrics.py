import unittest


class TestContinualMetrics(unittest.TestCase):
    def test_summarize_matrix(self):
        from yolozu.continual_metrics import summarize_continual_matrix

        matrix = [
            [0.8, 0.1, 0.05],
            [0.6, 0.75, 0.1],
            [0.55, 0.7, 0.72],
        ]
        summary = summarize_continual_matrix(matrix, baseline=[0.0, 0.05, 0.02])

        self.assertAlmostEqual(summary.avg_acc or 0.0, (0.55 + 0.7 + 0.72) / 3.0, places=7)
        self.assertAlmostEqual(summary.forgetting or 0.0, (0.25 + 0.05) / 2.0, places=7)
        self.assertAlmostEqual(summary.bwt or 0.0, (-0.25 - 0.05) / 2.0, places=7)
        self.assertAlmostEqual(summary.fwt or 0.0, ((0.1 - 0.05) + (0.1 - 0.02)) / 2.0, places=7)
        self.assertTrue(summary.details["fwt_defined"])

    def test_fwt_is_undefined_without_initial_baseline(self):
        from yolozu.continual_metrics import summarize_continual_matrix

        summary = summarize_continual_matrix([[0.8, 0.1], [0.7, 0.6]])

        self.assertIsNone(summary.fwt)
        self.assertFalse(summary.details["fwt_defined"])
        self.assertEqual(summary.details["fwt_pretrain"], [0.1])

    def test_empty_matrix(self):
        from yolozu.continual_metrics import summarize_continual_matrix

        summary = summarize_continual_matrix([])
        self.assertIsNone(summary.avg_acc)
        self.assertIsNone(summary.forgetting)


if __name__ == "__main__":
    unittest.main()
