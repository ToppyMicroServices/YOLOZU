"""Repository wrapper for the packaged prediction export CLI."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from yolozu.inference import export_predictions_cli


def main(argv=None):
    """Run the packaged CLI with repository-relative paths anchored to this checkout."""
    previous_root = export_predictions_cli.repo_root
    export_predictions_cli.repo_root = REPO_ROOT
    try:
        return export_predictions_cli.main(argv)
    finally:
        export_predictions_cli.repo_root = previous_root


if __name__ == "__main__":
    main()
