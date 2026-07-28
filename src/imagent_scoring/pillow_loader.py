from __future__ import annotations

from pathlib import Path

from .imaging import ImageData


# The only backend with a hard dependency that ships in this step. Pillow is
# imported at module import time on purpose: a missing image decoder must fail
# immediately and loudly rather than at the moment a candidate is being graded.

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "PillowImageLoader requires Pillow. Install the scoring extra: "
        'pip install "imagent-bench[scoring]"'
    ) from exc


_BOX = getattr(getattr(Image, "Resampling", Image), "BOX")


class PillowImageLoader:
    """Decode images with Pillow, downsampling by area average when asked."""

    def load(self, path: Path, *, max_edge: int | None = None) -> ImageData:
        with Image.open(path) as handle:
            handle.load()
            image = handle.convert("RGB")
            if max_edge is not None and max(image.size) > max_edge:
                scale = max_edge / max(image.size)
                width = max(1, int(image.width * scale))
                height = max(1, int(image.height * scale))
                image = image.resize((width, height), _BOX)
            return ImageData(
                width=image.width,
                height=image.height,
                pixels=tuple(image.getdata()),
            )

    def size(self, path: Path) -> tuple[int, int]:
        # Pillow reads only the header here, so this stays cheap on large files.
        with Image.open(path) as handle:
            return handle.size
