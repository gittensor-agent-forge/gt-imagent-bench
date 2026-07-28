from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from imagent_scoring.checklist import check_questions
from imagent_scoring.judge import judge_problem
from imagent_scoring.models import ChecklistQuestion
from imagent_scoring.openrouter import (
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_VQA_TEMPERATURE,
    OpenRouterClient,
    OpenRouterError,
    OpenRouterPairwiseJudge,
    OpenRouterVqa,
    image_data_url,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}], "usage": {"cost": 0.0001}}


def _client(*replies, capture: list | None = None, raiser=None, slept: list | None = None):
    queue = list(replies)

    def opener(request, timeout):
        if capture is not None:
            capture.append(json.loads(request.data))
        if raiser is not None:
            value = raiser() if callable(raiser) else raiser
            if value is not None:
                raise value
        return _Response(queue.pop(0) if queue else _reply("yes"))

    return OpenRouterClient(
        opener=opener, sleep=(slept.append if slept is not None else lambda _: None)
    )


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return path


# --- the client -------------------------------------------------------------


def test_a_missing_api_key_fails_loudly(monkeypatch, image: Path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(OpenRouterError, match="missing API key"):
        OpenRouterVqa(client=_client()).answer(image, "Is there a cup?")


def test_the_api_key_never_appears_in_an_error(image: Path) -> None:
    error = urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
    error.read = lambda: b'{"error": "bad model"}'  # type: ignore[method-assign]

    with pytest.raises(OpenRouterError) as raised:
        OpenRouterVqa(client=_client(raiser=error)).answer(image, "Is there a cup?")

    assert "test-key" not in str(raised.value)
    assert "bad model" in str(raised.value)


def test_transient_failures_are_retried_with_backoff(image: Path) -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def raiser():
        calls["n"] += 1
        if calls["n"] <= 2:
            error = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
            error.read = lambda: b"rate limited"  # type: ignore[method-assign]
            return error
        return None

    answer = OpenRouterVqa(client=_client(_reply("yes"), raiser=raiser, slept=slept)).answer(
        image, "Is there a cup?"
    )

    assert answer == "yes"
    assert slept == [1.0, 2.0]  # exponential


def test_a_permanent_failure_is_not_retried(image: Path) -> None:
    slept: list[float] = []
    error = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
    error.read = lambda: b"invalid key"  # type: ignore[method-assign]

    with pytest.raises(OpenRouterError, match="HTTP 401"):
        OpenRouterVqa(client=_client(raiser=error, slept=slept)).answer(image, "q")

    assert slept == []


def test_a_malformed_response_is_an_error(image: Path) -> None:
    with pytest.raises(OpenRouterError, match="no choices"):
        OpenRouterVqa(client=_client({"choices": []})).answer(image, "q")


# --- the checklist answerer -------------------------------------------------


def test_the_question_and_image_both_reach_the_model(image: Path) -> None:
    captured: list = []
    OpenRouterVqa(client=_client(capture=captured)).answer(image, "Is the cup blue?")

    content = captured[0]["messages"][0]["content"]
    assert "Is the cup blue?" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured[0]["temperature"] == DEFAULT_VQA_TEMPERATURE


def test_an_image_is_encoded_once_no_matter_how_many_questions(image: Path, monkeypatch) -> None:
    # A checklist asks one image several questions; re-reading and re-encoding a
    # megabyte each time is pure waste.
    reads = {"n": 0}
    original = Path.read_bytes

    def counting(self):
        if self.name == "candidate.png":
            reads["n"] += 1
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    vqa = OpenRouterVqa(client=_client(_reply("yes"), _reply("no"), _reply("yes")))
    for question in ("q1", "q2", "q3"):
        vqa.answer(image, question)

    assert reads["n"] == 1


def test_the_answer_flows_through_the_checklist(image: Path) -> None:
    vqa = OpenRouterVqa(client=_client(_reply("Yes."), _reply("no")))
    questions = [
        ChecklistQuestion(id="q1", text="Is there a hat?"),
        ChecklistQuestion(id="q2", text="Is the hat red?", depends_on="q1"),
    ]

    results = check_questions(questions, image, vqa=vqa)

    assert results[0].passed
    assert not results[1].passed


def test_a_hedging_answer_never_earns_a_point(image: Path) -> None:
    vqa = OpenRouterVqa(client=_client(_reply("It is difficult to say for certain")))

    results = check_questions([ChecklistQuestion(id="q1", text="Is there a hat?")], image, vqa=vqa)

    assert not results[0].passed


# --- the pairwise judge -----------------------------------------------------


def _judge(*contents, capture=None):
    replies = [_reply(content) for content in contents] or [_reply('{"winner":"tie"}')]
    return OpenRouterPairwiseJudge(client=_client(*replies, capture=capture))


def test_the_judge_samples_rather_than_running_greedy(image: Path) -> None:
    # Three votes at temperature 0 would be three copies of one answer, so the
    # majority rule would buy nothing while paying triple.
    captured: list = []
    _judge('{"winner":"A","reason":"cleaner"}', capture=captured).compare(
        prompt="a red cup", image_a=image, image_b=image
    )

    assert captured[0]["temperature"] == DEFAULT_JUDGE_TEMPERATURE
    assert DEFAULT_JUDGE_TEMPERATURE > 0


def test_both_images_are_sent_and_labelled(image: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.png"
    other.write_bytes(b"\x89PNG\r\n\x1a\nsecond")
    captured: list = []

    _judge('{"winner":"B","reason":"sharper"}', capture=captured).compare(
        prompt="a red cup", image_a=image, image_b=other
    )

    content = captured[0]["messages"][0]["content"]
    labels = [part.get("text", "") for part in content if part["type"] == "text"]
    images = [part for part in content if part["type"] == "image_url"]
    assert "Image A:" in labels and "Image B:" in labels
    assert len(images) == 2
    assert images[0]["image_url"]["url"] != images[1]["image_url"]["url"]


def test_the_judge_is_never_told_who_made_which_image(image: Path) -> None:
    captured: list = []
    _judge('{"winner":"A","reason":"x"}', capture=captured).compare(
        prompt="a red cup", image_a=image, image_b=image, notes_a="7/8 checks", notes_b="8/8 checks"
    )

    payload = json.dumps(captured[0])
    for word in ("king", "challenger", "incumbent", "reigning"):
        assert word not in payload.casefold()


def test_fact_check_results_are_passed_through(image: Path) -> None:
    captured: list = []
    _judge('{"winner":"A","reason":"x"}', capture=captured).compare(
        prompt="a red cup", image_a=image, image_b=image, notes_a="7/8 checks", notes_b="8/8 checks"
    )

    text = captured[0]["messages"][0]["content"][0]["text"]
    assert "7/8 checks" in text and "8/8 checks" in text
    assert "do not re-count" in text.casefold()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"winner":"A","reason":"x"}', "A"),
        ('{"winner":"b","reason":"x"}', "B"),
        ('{"winner":"Image A"}', "A"),
        ('{"winner":"tie"}', "tie"),
        ('```json\n{"winner":"B"}\n```', "B"),
        ('{"winner":"neither"}', "tie"),
        ("{}", "tie"),
    ],
)
def test_verdicts_are_parsed(image: Path, content: str, expected: str) -> None:
    assert _judge(content).compare(prompt="p", image_a=image, image_b=image) == expected


def test_a_non_json_verdict_is_an_error_not_a_silent_win(image: Path) -> None:
    with pytest.raises(OpenRouterError, match="expected a JSON object"):
        _judge("Image A is clearly better.").compare(prompt="p", image_a=image, image_b=image)


def test_rationales_are_kept_for_publication(image: Path) -> None:
    judge = _judge('{"winner":"A","reason":"crisper typography"}')
    judge.compare(prompt="p", image_a=image, image_b=image)

    assert judge.rationales == ["crisper typography"]


def test_the_judge_plugs_into_the_blind_voting_layer(image: Path) -> None:
    judge = _judge(
        '{"winner":"A","reason":"1"}', '{"winner":"A","reason":"2"}', '{"winner":"B","reason":"3"}'
    )

    verdict = judge_problem(
        judge=judge,
        challenge_id="c1",
        problem_id="p1",
        prompt="a red cup",
        king_image=image,
        challenger_image=image,
        slot="A",
    )

    assert verdict.winner == "challenger"  # 2 of 3 votes for slot A
    assert len(verdict.votes) == 3


def test_a_judge_outage_abstains_rather_than_deciding(image: Path) -> None:
    error = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
    error.read = lambda: b"invalid key"  # type: ignore[method-assign]
    judge = OpenRouterPairwiseJudge(client=_client(raiser=error))

    verdict = judge_problem(
        judge=judge,
        challenge_id="c1",
        problem_id="p1",
        prompt="a red cup",
        king_image=image,
        challenger_image=image,
    )

    assert verdict.winner == "tie"
    assert all(vote.error for vote in verdict.votes)


# --- encoding ---------------------------------------------------------------


def test_the_data_url_carries_the_real_media_type(tmp_path: Path) -> None:
    jpeg = tmp_path / "shot.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff")

    assert image_data_url(jpeg).startswith("data:image/jpeg;base64,")
