from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

from imagent_bench.scoring import evaluate_openrouter_vision


def test_openrouter_vision_judge_parses_dimension_scores(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")

    class FakeResponse:
        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "judge/model",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "scores": {
                                            "prompt_alignment": 90,
                                            "visual_quality": 80,
                                        },
                                        "rationale": "matches prompt",
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"cost": 0.002},
                }
            ).encode("utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())

    result = evaluate_openrouter_vision(
        image_path,
        prompt="Create a clean product image.",
        config={
            "provider": "openrouter_vision",
            "model": "judge/model",
            "dimensions": {"prompt_alignment": 0.75, "visual_quality": 0.25},
        },
    )

    assert result["overall_score"] == 87.5
    assert result["dimensions"] == {"prompt_alignment": 90.0, "visual_quality": 80.0}
    assert result["cost_usd"] == 0.002
    assert result["judge"]["model"] == "judge/model"


def test_openrouter_vision_judge_retries_on_transient_429(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")

    class FakeResponse:
        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "judge/model",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"scores": {"prompt_alignment": 90}, "rationale": "ok"}
                                )
                            }
                        }
                    ],
                    "usage": {"cost": 0.001},
                }
            ).encode("utf-8")

    calls = {"count": 0}

    def flaky_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                url="https://openrouter.ai",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=io.BytesIO(b"rate limited"),
            )
        return FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", flaky_urlopen)
    monkeypatch.setattr("imagent_bench.scoring.time.sleep", lambda *_: None)

    result = evaluate_openrouter_vision(
        image_path,
        prompt="Create a clean product image.",
        config={
            "provider": "openrouter_vision",
            "model": "judge/model",
            "dimensions": {"prompt_alignment": 1.0},
        },
    )

    assert calls["count"] == 2
    assert result["overall_score"] == 90.0
