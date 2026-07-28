from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .backends import ImageLoader
from .imaging import hamming_distance, phash, stddev, to_grayscale
from .models import ValidityReport


# S1. Runs before anything expensive, so a broken or replayed submission costs
# nothing to reject.

MIN_EDGE = 64
# Below this luma standard deviation the image carries no structure: a solid
# colour, or near enough that no benchmark answer could be read from it.
MIN_STDDEV = 2.0
# Two perceptual hashes this close are the same picture. 5/64 bits is the
# conventional threshold and tolerates re-encoding without matching unrelated
# images.
DUPLICATE_DISTANCE = 5
# Validity only needs coarse statistics, so decode small and keep it fast.
ANALYSIS_EDGE = 256


def check_validity(
    path: Path,
    *,
    loader: ImageLoader,
    known_hashes: Iterable[str] = (),
    min_edge: int = MIN_EDGE,
    min_stddev: float = MIN_STDDEV,
    duplicate_distance: int = DUPLICATE_DISTANCE,
) -> ValidityReport:
    """Decide whether an image is usable at all.

    `known_hashes` is every previously archived output. A match means the agent
    replayed an existing answer rather than generating one.
    """
    try:
        # True dimensions come from the header. The pixels are only needed for
        # coarse statistics, so they are decoded small - but the size check and
        # the published report must describe the real image, not the analysis copy.
        width, height = loader.size(path)
        image = loader.load(path, max_edge=ANALYSIS_EDGE)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same verdict
        return ValidityReport(
            valid=False,
            reasons=(f"image could not be decoded: {exc}",),
            width=0,
            height=0,
            phash="",
            stddev=0.0,
        )

    reasons: list[str] = []
    if min(width, height) < min_edge:
        reasons.append(f"image is smaller than {min_edge}px on its shortest edge")

    spread = stddev(to_grayscale(image))
    if spread < min_stddev:
        reasons.append("image is blank or a solid colour")

    digest = phash(image)
    for known in known_hashes:
        if not known or len(known) != len(digest):
            continue
        if hamming_distance(digest, known) <= duplicate_distance:
            reasons.append(f"image duplicates a previously archived output ({known})")
            break

    return ValidityReport(
        valid=not reasons,
        reasons=tuple(reasons),
        width=width,
        height=height,
        phash=digest,
        stddev=spread,
    )
