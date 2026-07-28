# gt-imagent-bench — the Imagent benchmark

Answers one question: **given an image and a problem, how good is it?**

Nothing here knows what a crown is. Deciding who competes and what a result means
lives in [gt-imagent](https://github.com/gittensor-agent-forge/gt-imagent).

```bash
python -m pip install -e ".[scoring]"
```

## Two packages

```
src/imagent_bench/
  problems.py       generates problems and the answer keys that grade them
src/imagent_scoring/
  validity.py       S1: decodable, not blank, not a replay
  facts.py          S2: folds every check into one fact score
  text_check.py     OCR matching with partial credit
  object_check.py   presence, count, colour, spatial relations
  checklist.py      TIFA/DSG yes/no questions with dependencies
  answer_key.py     strict parsing of generated answer keys
  judge.py          S4: blind pairwise voting
  ladder.py         which of two images answered a problem better
  openrouter.py     the vision backends: checklist answerer and pairwise judge
  backends.py       protocols for OCR, detector, VQA, image loading
  imaging.py        greyscale, box resize, perceptual hash
  pillow_loader.py  the shipped image-decoding backend
```

`imagent_bench` is **stdlib-only**, deliberately: it is pinned inside an attested
TEE image and must produce byte-identical problems there and on the validator. If
those ever diverge, the room is answering questions nobody is grading.

`imagent_scoring`'s rules are stdlib-only too; every heavy model sits behind a
protocol in `backends.py`, so the rules a miner is judged by can be read and
re-derived without installing anything, and the full suite runs without a GPU.

## Problems

Published benchmarks like GenEval are not fixed prompt lists — they are templates
over vocabularies. Run one with a fresh seed and you get problems nobody has seen,
using a methodology that is already peer reviewed. Because the template knows what
it asked for, the answer key comes free.

```python
from imagent_bench import generate_problems

for problem in generate_problems(seed, count=7):
    problem.prompt      # "a photo of three cakes"
    problem.answer_key  # {"objects": [{"name": "cake", "count": 3}]}
```

Seven problems per challenge: 3 GenEval (counting, colour binding, position),
2 T2I-CompBench++ (attribute binding, spatial), 1 text rendering, 1 dense prompt.

Selection is SHA-256 in counter mode, not `random` — CPython has changed that
module's selection internals between releases, and a benchmark that drifts
between versions is not reproducible. A golden-vector test pins it.

## Scoring

```python
from imagent_scoring import check_validity, parse_answer_key, score_facts
from imagent_scoring.pillow_loader import PillowImageLoader

validity = check_validity(path, loader=PillowImageLoader(), known_hashes=archive)
report = score_facts(path, parse_answer_key(problem.answer_key), detector=..., ocr=...)

report.fact_score   # 0.75
report.to_dict()    # per-check pass/fail, published verbatim to the miner
```

A declared requirement with no backend supplied **raises**. A missing model must
never silently become a free pass.

## Judging

Absolute 0–100 scores from a model drift between sessions, so the judge is never
asked for one. It is shown two images and asked which is better — blind, with the
slot decided by a hash, three votes, majority wins.

```python
from imagent_scoring import judge_problem
from imagent_scoring.openrouter import OpenRouterPairwiseJudge
```

The judge samples rather than running greedy. Three votes at temperature 0 are
three copies of one answer, and the majority would buy nothing while paying triple.

## Tests

```bash
python -m pytest        # 105 tests, no GPU required
```

## Still to build

The local model backends: object detector and colour classifier, an OCR engine,
and the preference models (ImageReward / PickScore / HPS v2). Until those land,
fact scores come from stubs.

## License

Apache-2.0.
