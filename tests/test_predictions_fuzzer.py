import unittest

from fuzz.predictions_canonicalize_fuzzer import consume_input


class PredictionsFuzzerHarnessTests(unittest.TestCase):
    def test_accepts_wrapped_predictions_payload(self) -> None:
        consume_input(
            b'{"predictions":[{"image":"demo.jpg","detections":[{"class_id":1,"score":0.9,"bbox":[0.5,0.5,0.2,0.2]}]}]}'
        )

    def test_ignores_non_json_bytes(self) -> None:
        consume_input(b"\x00\xffnot-json")


if __name__ == "__main__":
    unittest.main()
