from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _bash_block_containing(text: str, marker: str) -> str:
    blocks = re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL)
    matches = [block for block in blocks if marker in block]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one bash block containing {marker!r}, found {len(matches)}"
        )
    return matches[0]


def _section_between(text: str, start: str, end: str | None = None) -> str:
    if start not in text:
        raise AssertionError(f"missing section heading {start!r}")
    section = text.split(start, 1)[1]
    if end is None:
        return section
    if end not in section:
        raise AssertionError(f"missing section heading {end!r} after {start!r}")
    return section.split(end, 1)[0]


class AdoptionDocsTests(unittest.TestCase):
    def test_baseline_preserves_privacy_and_automation_boundaries(self) -> None:
        guide = (ROOT / "docs/adoption/README.md").read_text(encoding="utf-8")
        baseline = (ROOT / "docs/adoption/2026-07-23-baseline.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("does not include product telemetry", guide)
        self.assertIn("automation-sensitive", guide)
        self.assertIn("Human adoption from", baseline)
        self.assertIn("`unknown`, not 72", baseline)
        self.assertIn("## 30-day targets", baseline)
        self.assertIn("## 90-day targets", baseline)

    def test_docs_index_links_the_measurement_guide(self) -> None:
        docs_index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
        self.assertIn("[`adoption/README.md`](adoption/README.md)", docs_index)
        self.assertIn(
            "[`adoption/design_partner_observation_kit.md`]"
            "(adoption/design_partner_observation_kit.md)",
            docs_index,
        )

    def test_observation_kit_preserves_consent_and_privacy_boundaries(self) -> None:
        kit = (
            ROOT / "docs/adoption/design_partner_observation_kit.md"
        ).read_text(encoding="utf-8")

        for heading in (
            "## Invitation",
            "## Consent and recording boundary",
            "## Pre-session checklist",
            "## Checked observation task",
            "## Moderator script",
            "## Privacy-safe worksheet",
            "## Anonymization and retention",
            "## Blocker-to-Beads triage",
            "## Post-session follow-up",
        ):
            self.assertIn(heading, kit)

        self.assertIn("Audio, video, and screen recording are off", kit)
        self.assertIn("requires separate explicit written permission", kit)
        self.assertIn("[SECURITY.md](../../SECURITY.md)", kit)
        self.assertIn("security@toppymicros.com", kit)
        self.assertIn("at least two consented external sessions", kit)
        self.assertIn("Do not put names, organizations, logos", kit)
        self.assertIn("Do not commit an individual", kit)
        self.assertIn("not count it as a real external evaluation", kit)
        self.assertIn("14 calendar days", kit)
        self.assertIn("90 days after the", kit)
        self.assertIn("session correspondence from the active", kit)
        self.assertIn(
            "earlier correspondence body, attachments, or session summary",
            kit,
        )
        self.assertIn(
            "delete new follow-up correspondence and attachments",
            kit,
        )
        self.assertIn("private aggregate blocker tally", kit)

    def test_observation_task_matches_the_checked_stable_quickstart(self) -> None:
        docs_index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
        kit = (
            ROOT / "docs/adoption/design_partner_observation_kit.md"
        ).read_text(encoding="utf-8")
        marker = "--output reports/smoke_coco_eval_dry_run.json"
        docs_block = _bash_block_containing(docs_index, marker)
        kit_block = _bash_block_containing(kit, marker)

        self.assertEqual(kit_block, docs_block)
        for required in (
            "--dataset data/smoke",
            "--split val",
            "--predictions data/smoke/predictions/predictions_dummy.json",
            "--dry-run",
            marker,
        ):
            self.assertIn(required, kit_block)
        self.assertNotIn("validate dataset", kit_block)
        self.assertNotIn("validate predictions", kit_block)

    def test_real_evaluation_preflight_is_explicit(self) -> None:
        kit = (
            ROOT / "docs/adoption/design_partner_observation_kit.md"
        ).read_text(encoding="utf-8")

        for required in (
            "compatible object-detection predictions",
            "permitted YOLO-format ground truth",
            "python3 -c 'import pycocotools'",
            "`image` join keys",
            "canonical normalized `cxcywh`",
            "contiguous zero-based class IDs",
            "outside this observation",
            "contains non-null metrics",
        ):
            self.assertIn(required, kit)

        real_block = _bash_block_containing(
            kit,
            "--output /absolute/path/to/coco_eval.json",
        )
        self.assertIn(
            "validate dataset \\\n"
            "\t/absolute/path/to/yolo-dataset \\\n"
            "\t--split val \\\n"
            "\t--strict",
            real_block,
        )
        self.assertIn("--bbox-format cxcywh_norm", real_block)

    def test_consent_boundary_discloses_retention_before_observation(self) -> None:
        kit = (
            ROOT / "docs/adoption/design_partner_observation_kit.md"
        ).read_text(encoding="utf-8")
        boundary = _section_between(
            kit,
            "## Consent and recording boundary",
            "## Pre-session checklist",
        )

        self.assertIn("14-day correction or deletion window", boundary)
        self.assertIn("at most 90 days", boundary)
        self.assertIn("Provider-managed backups", boundary)
        self.assertIn("individual record cannot be located", boundary)

    def test_copyable_messages_use_the_absolute_private_security_route(self) -> None:
        kit = (
            ROOT / "docs/adoption/design_partner_observation_kit.md"
        ).read_text(encoding="utf-8")
        invitation = _section_between(
            kit,
            "## Invitation",
            "## Consent and recording boundary",
        )
        follow_up = _section_between(kit, "## Post-session follow-up")

        for message in (invitation, follow_up):
            self.assertIn("security@toppymicros.com", message)
            self.assertIn(
                "https://toppymicros.com/security-policy.html",
                message,
            )
            self.assertNotIn("[SECURITY.md](../../SECURITY.md)", message)

    def test_rehearsal_not_applicable_values_match_the_worksheet(self) -> None:
        kit = (
            ROOT / "docs/adoption/design_partner_observation_kit.md"
        ).read_text(encoding="utf-8")
        dry_run = (
            ROOT / "docs/adoption/2026-07-23-maintainer-kit-dry-run.md"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "`not applicable` is allowed only for a maintainer rehearsal",
            kit,
        )
        self.assertIn(
            "`not applicable` only for a maintainer rehearsal",
            kit,
        )
        self.assertIn("no external participant", dry_run)
        self.assertIn("Follow-up intent | not applicable", dry_run)

    def test_maintainer_dry_run_is_not_external_adoption_evidence(self) -> None:
        dry_run = (
            ROOT / "docs/adoption/2026-07-23-maintainer-kit-dry-run.md"
        ).read_text(encoding="utf-8")

        self.assertIn("repository-owned smoke data", dry_run)
        self.assertIn("not a design-partner session", dry_run)
        self.assertIn("does not count toward", dry_run)
        self.assertIn("Time to first comparable report", dry_run)
        self.assertIn("`not reached`; the smoke run was dry-run only", dry_run)
        self.assertIn("optional dependency was", dry_run)
        self.assertIn("not an external blocker observation", dry_run)
        self.assertIn("Real design-partner outcomes remain unknown", dry_run)


if __name__ == "__main__":
    unittest.main()
