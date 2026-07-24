import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start_marker = f"## {heading}"
    try:
        start = lines.index(start_marker) + 1
    except ValueError as exc:
        raise AssertionError(f"missing README section: {start_marker}") from exc
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


class TestProductPositioning(unittest.TestCase):
    def test_readme_translations_share_the_stable_product_wedge(self) -> None:
        expectations = {
            "README.md": {
                "intro": "commercial product developed by ToppyMicroServices OÜ and provided free of charge",
                "fit_heading": "Best Fit",
                "not_fit_heading": "Not The Best Fit",
                "not_fit_marker": "managed training platform",
                "secondary_marker": "secondary qualified lanes",
            },
            "Readme_jp.md": {
                "intro": "ToppyMicroServices OÜ が開発する商用プロダクトで、無料で提供",
                "fit_heading": "特に向いている3つのケース",
                "not_fit_heading": "あまり向いていないケース",
                "not_fit_marker": "managed training platform",
                "secondary_marker": "secondary lane",
            },
            "Readme_zh.md": {
                "intro": "ToppyMicroServices OÜ 开发、免费提供的商业产品",
                "fit_heading": "特别适合的三个场景",
                "not_fit_heading": "不太适合的场景",
                "not_fit_marker": "managed training platform",
                "secondary_marker": "secondary lane",
            },
        }

        for relative_path, expected in expectations.items():
            with self.subTest(relative_path=relative_path):
                text = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(expected["intro"], text)
                self.assertIn("stable predictions interface contract", text)
                self.assertIn(
                    "[ToppyMicroServices OÜ](https://www.toppymicros.com/)",
                    text,
                )
                self.assertIn(
                    "Official page: <https://www.toppymicros.com/yolozu/>",
                    text,
                )
                self.assertIn("PyPI: <https://pypi.org/project/yolozu/>", text)

                fit = _section(text, expected["fit_heading"])
                use_cases = [line for line in fit.splitlines() if line.startswith("- ")]
                self.assertEqual(
                    len(use_cases),
                    3,
                    f"{relative_path} must list exactly three best-fit use cases",
                )
                for marker in ("dataset", "third-party", "CI", "drift"):
                    self.assertIn(marker, fit)

                not_fit = _section(text, expected["not_fit_heading"])
                self.assertIn(expected["not_fit_marker"], not_fit)
                self.assertIn("hosted inference service", not_fit)
                support_boundary = (
                    "support or SLA"
                    if relative_path == "README.md"
                    else "support / SLA"
                )
                self.assertIn(support_boundary, not_fit)
                self.assertIn("one-click production deployment", not_fit)
                self.assertIn("native evaluator", not_fit)
                self.assertIn("training", not_fit)
                self.assertIn("research", not_fit)
                self.assertIn(expected["secondary_marker"], not_fit)

    def test_package_metadata_uses_the_same_value_proposition_and_links(self) -> None:
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            'description = "Commercial ToppyMicroServices product, provided free of '
            "charge, for validating and fairly evaluating existing vision predictions "
            "through a stable predictions "
            'interface contract."',
            metadata,
        )
        self.assertIn(
            'Homepage = "https://www.toppymicros.com/yolozu/"',
            metadata,
        )
        self.assertIn(
            'Company = "https://www.toppymicros.com/"',
            metadata,
        )
        self.assertIn(
            'Repository = "https://github.com/ToppyMicroServices/YOLOZU"',
            metadata,
        )
        self.assertIn('"model-evaluation"', metadata)
        self.assertIn('"predictions"', metadata)


if __name__ == "__main__":
    unittest.main()
