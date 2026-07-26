"""Repository wrapper for the packaged prediction export CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.inference.export_predictions_cli import main


if __name__ == "__main__":
    main()
