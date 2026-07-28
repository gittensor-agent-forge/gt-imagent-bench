from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .imaging import ImageData
from .models import Detection, TextSpan


# Every heavy dependency sits behind one of these protocols. The scoring rules
# themselves are pure Python and have no dependencies, so they can be tested
# exhaustively with stubs and audited without a GPU. Real implementations
# (Pillow, an OCR engine, a detector, a VQA model) are wired in separately.


@runtime_checkable
class ImageLoader(Protocol):
    def load(self, path: Path, *, max_edge: int | None = None) -> ImageData:
        """Decode an image, optionally downsampling so the longest edge is max_edge."""

    def size(self, path: Path) -> tuple[int, int]:
        """Return (width, height) without decoding the pixels.

        Spatial relations are checked against detection boxes in original pixel
        coordinates, so the true size is needed even when the pixels are not.
        """


@runtime_checkable
class OcrEngine(Protocol):
    def read(self, path: Path) -> list[TextSpan]:
        """Return every text span found in the image."""


@runtime_checkable
class ObjectDetector(Protocol):
    def detect(self, path: Path) -> list[Detection]:
        """Return every detected object, with colour filled in where available."""


@runtime_checkable
class VqaEngine(Protocol):
    def answer(self, path: Path, question: str) -> str:
        """Answer a yes/no question about the image. Returns 'yes' or 'no'."""


@runtime_checkable
class ObjectVerifier(Protocol):
    """Answers object questions directly, for deployments with no local detector.

    A detector is strictly better — it is deterministic and its answer can be
    re-derived by anyone. This exists because a detector needs a GPU, and a
    competition that cannot grade at all is worse than one that grades with a
    vision model. Results from this path are marked non-deterministic in the
    report so nobody mistakes one for the other.
    """

    def count(self, path: Path, name: str) -> int:
        """How many of `name` are visible."""

    def colour(self, path: Path, name: str) -> str:
        """The dominant colour of `name`, or "" if it is not present."""

    def relation(self, path: Path, subject: str, relation: str, obj: str) -> bool:
        """Whether `subject` is positioned `relation` `obj`."""
