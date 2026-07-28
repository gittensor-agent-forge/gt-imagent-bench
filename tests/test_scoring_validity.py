from __future__ import annotations

from pathlib import Path

import pytest

from imagent_scoring import check_validity, hamming_distance, phash
from imagent_scoring.imaging import ImageData
from imagent_scoring.pillow_loader import PillowImageLoader

Image = pytest.importorskip("PIL.Image")


def _gradient(path: Path, size: int = 128) -> Path:
    image = Image.new("RGB", (size, size))
    image.putdata(
        [
            (int(255 * x / size), int(255 * y / size), (x * y) % 256)
            for y in range(size)
            for x in range(size)
        ]
    )
    image.save(path)
    return path


def _solid(path: Path, size: int = 128) -> Path:
    Image.new("RGB", (size, size), (17, 17, 17)).save(path)
    return path


def test_a_normal_image_is_valid(tmp_path: Path) -> None:
    report = check_validity(_gradient(tmp_path / "ok.png"), loader=PillowImageLoader())

    assert report.valid
    assert report.reasons == ()
    assert report.width == 128
    assert len(report.phash) == 16


def test_a_blank_image_is_rejected(tmp_path: Path) -> None:
    report = check_validity(_solid(tmp_path / "blank.png"), loader=PillowImageLoader())

    assert not report.valid
    assert any("blank" in reason for reason in report.reasons)


def test_the_report_describes_the_real_image_not_the_analysis_copy(tmp_path: Path) -> None:
    # Validity decodes a downscaled copy for speed. If the size check ran against
    # that copy, a large but non-square image would be rejected for being small.
    wide = tmp_path / "wide.png"
    image = Image.new("RGB", (1024, 128))
    image.putdata([(x % 256, y % 256, 0) for y in range(128) for x in range(1024)])
    image.save(wide)

    report = check_validity(wide, loader=PillowImageLoader())

    assert report.valid
    assert (report.width, report.height) == (1024, 128)


def test_a_tiny_image_is_rejected(tmp_path: Path) -> None:
    report = check_validity(_gradient(tmp_path / "tiny.png", size=32), loader=PillowImageLoader())

    assert not report.valid
    assert any("shortest edge" in reason for reason in report.reasons)


def test_an_undecodable_file_is_rejected_without_raising(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")

    report = check_validity(broken, loader=PillowImageLoader())

    assert not report.valid
    assert any("decoded" in reason for reason in report.reasons)


def test_a_replayed_image_is_rejected(tmp_path: Path) -> None:
    original = _gradient(tmp_path / "original.png")
    loader = PillowImageLoader()
    archived = check_validity(original, loader=loader).phash

    # Same picture, re-encoded and rescaled - exactly what a replay looks like.
    with Image.open(original) as handle:
        handle.resize((96, 96)).save(tmp_path / "replay.jpg", quality=85)

    report = check_validity(tmp_path / "replay.jpg", loader=loader, known_hashes=[archived])

    assert not report.valid
    assert any("duplicates" in reason for reason in report.reasons)


def test_a_different_image_is_not_flagged_as_a_duplicate(tmp_path: Path) -> None:
    loader = PillowImageLoader()
    archived = check_validity(_gradient(tmp_path / "a.png"), loader=loader).phash

    checkerboard = Image.new("RGB", (128, 128))
    checkerboard.putdata(
        [
            (255, 255, 255) if ((x // 8) + (y // 8)) % 2 else (0, 0, 0)
            for y in range(128)
            for x in range(128)
        ]
    )
    checkerboard.save(tmp_path / "b.png")

    report = check_validity(tmp_path / "b.png", loader=loader, known_hashes=[archived])

    assert report.valid


def test_phash_is_stable_and_comparable() -> None:
    pixels = tuple((x, y, 0) for y in range(64) for x in range(64))
    image = ImageData(width=64, height=64, pixels=pixels)

    first = phash(image)
    second = phash(image)

    assert first == second
    assert hamming_distance(first, second) == 0
