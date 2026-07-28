from __future__ import annotations

import math
from dataclasses import dataclass


# Pure-Python image maths. No third-party imports: a miner's score must be
# recomputable by anyone with a stock interpreter, and these are the numbers the
# validity check publishes.


@dataclass(frozen=True)
class ImageData:
    """A decoded image as row-major RGB triples."""

    width: int
    height: int
    pixels: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if len(self.pixels) != self.width * self.height:
            raise ValueError(
                f"pixel count {len(self.pixels)} does not match {self.width}x{self.height}"
            )


def to_grayscale(image: ImageData) -> list[float]:
    """ITU-R BT.601 luma, the same weighting Pillow's "L" conversion uses."""
    return [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in image.pixels]


def stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def resize_gray(
    values: list[float], width: int, height: int, target_width: int, target_height: int
) -> list[float]:
    """Box-average downscale (nearest-neighbour when upscaling).

    Averaging rather than sampling matters: a single-pixel sample of a dithered
    or noisy image produces an unstable hash, and the hash has to be stable to be
    usable as a duplicate check.
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target dimensions must be positive")

    result: list[float] = []
    for target_y in range(target_height):
        y0 = int(target_y * height / target_height)
        y1 = max(y0 + 1, int((target_y + 1) * height / target_height))
        for target_x in range(target_width):
            x0 = int(target_x * width / target_width)
            x1 = max(x0 + 1, int((target_x + 1) * width / target_width))
            total = 0.0
            count = 0
            for y in range(y0, min(y1, height)):
                row = y * width
                for x in range(x0, min(x1, width)):
                    total += values[row + x]
                    count += 1
            result.append(total / count if count else 0.0)
    return result


def _dct_1d(values: list[float]) -> list[float]:
    """Unnormalised DCT-II. Only the coefficient ordering matters for a hash."""
    size = len(values)
    factor = math.pi / (2.0 * size)
    return [
        sum(value * math.cos((2 * index + 1) * k * factor) for index, value in enumerate(values))
        for k in range(size)
    ]


def _dct_2d(values: list[float], size: int) -> list[float]:
    rows = [_dct_1d(values[y * size : (y + 1) * size]) for y in range(size)]
    columns = [_dct_1d([rows[y][x] for y in range(size)]) for x in range(size)]
    # columns[x][y] -> back to row-major
    return [columns[x][y] for y in range(size) for x in range(size)]


def phash(image: ImageData, *, hash_size: int = 8, dct_size: int = 32) -> str:
    """Perceptual hash, the standard DCT construction.

    Used to catch a candidate replaying a previous winner's image: two images
    within a small Hamming distance are the same picture, even if re-encoded,
    rescaled, or lightly recompressed.
    """
    if hash_size <= 0 or dct_size < hash_size:
        raise ValueError("hash_size must be positive and no larger than dct_size")

    gray = resize_gray(to_grayscale(image), image.width, image.height, dct_size, dct_size)
    coefficients = _dct_2d(gray, dct_size)

    # Top-left block holds the low frequencies. Drop the DC term: it only encodes
    # overall brightness, which we do not want the hash to depend on.
    block = [coefficients[y * dct_size + x] for y in range(hash_size) for x in range(hash_size)]
    ranked = sorted(block[1:])
    median = ranked[len(ranked) // 2] if ranked else 0.0

    bits = "".join("1" if value > median else "0" for value in block)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("hashes must be the same length")
    return bin(int(left, 16) ^ int(right, 16)).count("1")
