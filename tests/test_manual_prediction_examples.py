import json
from pathlib import Path
import re
import unittest

from yolozu.predictions.predictions import validate_predictions_payload


class TestManualPredictionExamples(unittest.TestCase):
    def test_base_prediction_examples_pass_strict_validation(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "manual/chapters/03_concepts_contracts.tex").read_text(encoding="utf-8")
        for heading in ("Detection", "Keypoints", "6DoF pose"):
            with self.subTest(heading=heading):
                section = text.split(r"\paragraph{" + heading + "}", 1)[1]
                match = re.search(r"\\begin\{lstlisting\}[^\n]*\n(.*?)\\end\{lstlisting\}", section, re.DOTALL)
                self.assertIsNotNone(match)
                payload = json.loads(match.group(1))
                result = validate_predictions_payload(payload, strict=True)
                self.assertEqual(result.warnings, [])
                self.assertEqual(payload["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
