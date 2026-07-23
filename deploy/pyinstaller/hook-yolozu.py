"""PyInstaller hook for modules loaded through YOLOZU's legacy aliases."""

from yolozu import _LEGACY_SUBMODULE_ALIASES


hiddenimports = sorted(set(_LEGACY_SUBMODULE_ALIASES.values()))
