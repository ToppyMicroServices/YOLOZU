"""Image identity lookup for evaluation, with unambiguous fallback aliases."""

from __future__ import annotations

from pathlib import Path

from yolozu.core.image_keys import image_basename


class ImageIdIndex:
    """Build once in O(images); resolve each image in constant time."""

    def __init__(self) -> None:
        self._exact: dict[str, int] = {}
        self._basenames: dict[str, int | None] = {}
        self._stems: dict[str, int | None] = {}

    @staticmethod
    def _add_alias(index: dict[str, int | None], alias: str, image_id: int) -> None:
        if not alias:
            return
        if alias not in index:
            index[alias] = image_id
        elif index[alias] != image_id:
            index[alias] = None

    def add(self, image: str, image_id: int) -> None:
        exact_keys = (image, image.replace("\\", "/"))
        for key in exact_keys:
            previous = self._exact.get(key)
            if previous is not None and previous != image_id:
                raise ValueError(f"duplicate dataset image key: {image}")
            self._exact[key] = image_id
        basename = image_basename(image)
        self._add_alias(self._basenames, basename, image_id)
        self._add_alias(self._stems, Path(basename).stem, image_id)

    def lookup(self, image: str) -> int | None:
        for key in (image, image.replace("\\", "/")):
            image_id = self._exact.get(key)
            if image_id is not None:
                return image_id
        basename = image_basename(image)
        for index, alias in (
            (self._basenames, basename),
            (self._stems, basename),
            (self._stems, Path(basename).stem),
        ):
            if alias not in index:
                continue
            image_id = index[alias]
            if image_id is None:
                raise ValueError(
                    f"ambiguous prediction image key: {image}; "
                    "use an exact dataset image path"
                )
            return image_id
        return None

    def as_dict(self) -> dict[str, int]:
        """Keep the existing public COCO index view without ambiguous aliases."""
        result = {key: value for key, value in self._stems.items() if value is not None}
        for key, image_id in self._basenames.items():
            if image_id is None:
                result.pop(key, None)
            else:
                result[key] = image_id
        result.update(self._exact)
        return result
