# RAG Evaluation Summary

- Generated at: 2026-03-09T00:15:53.418802+00:00
- Mode: `direct`
- Questions file: `eval/golden_questions_50.jsonl`
- Total questions: **50**
- Auto pass rate: **92.0%** (46/50)
- Groundedness pass rate: **100.0%** (50/50)
- Fallback rate: **2.0%** (1/50)
- p50 latency: **136.8 ms**
- p95 latency: **225.7 ms**

## Failure Buckets
- `fallback_inappropriate`: 4
- `behavior_expectation_miss`: 4

## Per-Category Hotspots
- `Other` -> fallback_inappropriate:4, behavior_expectation_miss:4

## Acceptance Targets
- Grounded answers with valid citations: at least 90%
- Correct/acceptable final answers (human-reviewed sample): at least 80%
- Low-evidence queries should prefer safe fallback over hallucination
- Stable p95 latency under your chosen runtime budget
