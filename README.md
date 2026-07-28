# Image Bench

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License"></a>
</p>

The engine that runs image agents and scores them. It loads a candidate
repository, executes its agent over a suite of cases, persists the artifacts,
judges the images, and writes a canonical benchmark report.

Stdlib-only by design: a scored run has no third-party supply chain between the
candidate agent and the recorded number.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Score an agent

```bash
export OPENROUTER_API_KEY=your-openrouter-api-key

imagent-bench run \
  --repository .. \
  --config configs/openrouter-vision-benchmark.json \
  --output-dir benchmark-output \
  --fail-on-policy
```

The runner:

1. loads the candidate's `agent/agent.yaml` and imports its entrypoint;
2. calls `setup(config, workdir)`;
3. for each case, calls `generate(case)` and receives image bytes plus a trace;
4. persists the artifacts itself and times the call;
5. scores the image with the configured judge;
6. writes `benchmark-report.json` and `benchmark-summary.md`.

The engine owns steps 4 through 6. A candidate cannot name its own artifacts,
report its own latency, or declare its own pass.

## Run one prompt

```bash
imagent-bench try "Create a polished benchmark badge titled CLI PASS."
```

Development convenience: same load-and-persist path as a scored run, one ad-hoc
prompt, no judge, no report. Writes `results/<UTC datetime>/`.

## Configurations

| Config | Suite | Scoring |
|---|---|---|
| `openrouter-vision-benchmark.json` | `openrouter_vision_v1` | OpenRouter vision judge across weighted dimensions |
| `openrouter-live-smoke.json` | `openrouter_live_smoke` | `none` — verify the live provider path without paying for a judge |

A judged case passes when its score reaches `expected.minimum_score`, falling
back to `policy.minimum_score` when the case declares no floor. There is no
silent default pass.

### Baseline comparison

```bash
imagent-bench run \
  --repository .. \
  --config configs/openrouter-vision-benchmark.json \
  --baseline-score 82.0 \
  --baseline-commit <current-top-commit> \
  --output-dir benchmark-output \
  --fail-on-policy
```

The report gains a `ranking` block: `delta`, a `label` (`score-regression`,
`no-significant-change`, `improvement-minor|strong|major`), and `merge_eligible`.

Note that `--baseline-score` is a *stored number*, not a re-run of the incumbent.
Comparing a fresh candidate against a stale baseline does not cancel judge drift;
head-to-head re-scoring is part of the planned king-of-the-hill rework.

## Runner API

```python
from imagent_bench import run

result = run(
    repository="..",
    commit="abc123",
    config="configs/openrouter-vision-benchmark.json",
    output_dir="benchmark-output",
)
```

`result.to_dict()` has the same shape as `benchmark-output/benchmark-report.json`.

## Report contract

Schema: `schemas/benchmark-report.schema.json`. The report carries overall status
and score, benchmark and dataset versions, commit SHA, per-case scores with judge
dimensions and artifacts, aggregate latency and cost metrics, policy thresholds
and failure reasons, optional baseline ranking, and logs.

Reports are consumed by `web` through `npm run import-report`.

## Layout

```
src/imagent_bench/
  runner.py         orchestration, metrics, ranking, report assembly
  artifacts.py      validates and persists what an agent returns
  scoring.py        OpenRouter vision judge
  agent_loader.py   manifest parsing and entrypoint import
  policy.py         pass/fail thresholds
  suite.py          suite and case loading
  models.py         config, case, and result dataclasses
  reporting.py      report and markdown summary writers
  try_agent.py      the `try` subcommand
  suites/           case definitions and assets
configs/            benchmark configurations
schemas/            canonical report JSON schema
tests/
```

## Tests

```bash
python -m pytest
```

## License

Apache-2.0.
