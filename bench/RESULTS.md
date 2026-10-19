# Benchmark results

Retrieval quality over time (SPEC.md M7.6), appended by `uv run python -m bench <dataset> --record`. Session recall: share of the answer's sessions among the top k results; any-hit: at least one; NDCG over the distinct sessions found; turn recall: share of the marked answer turns. Datasets are LongMemEval (MIT) and LoCoMo (non-commercial terms), downloaded to ~/.cache/openmcp-bench, never committed.


**Baselines (2026-10-19):** the two entries below measured main at 36471d3, before M7.7 (with the 30-day recency half-life), plus search's new `now` argument so ranking uses each question's date. That's what "+uncommitted" marks.
## 2026-10-19 locomo at 36471d3+uncommitted

Embedder: BAAI/bge-small-en-v1.5. 10 histories, 1986 questions. Ingest 4.593 s per history; query p50 6.4 ms, p95 8.0 ms.

| type | n | session_recall@5 | any_hit@5 | ndcg@5 | turn_recall@5 | session_recall@10 | any_hit@10 | ndcg@10 | turn_recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1986 | 0.833 | 0.888 | 0.734 | 0.560 | 0.902 | 0.947 | 0.762 | 0.653 |
| adversarial | 446 | 0.933 | 0.933 | 0.813 | 0.595 | 0.971 | 0.971 | 0.828 | 0.719 |
| multi-hop | 282 | 0.524 | 0.840 | 0.494 | 0.268 | 0.684 | 0.933 | 0.568 | 0.370 |
| open-domain | 96 | 0.505 | 0.652 | 0.446 | 0.266 | 0.640 | 0.783 | 0.495 | 0.332 |
| single-hop | 841 | 0.916 | 0.916 | 0.805 | 0.641 | 0.970 | 0.970 | 0.825 | 0.726 |
| temporal | 321 | 0.844 | 0.863 | 0.732 | 0.639 | 0.895 | 0.910 | 0.752 | 0.709 |

## 2026-10-19 longmemeval-s (limit 100) at 36471d3+uncommitted (originally logged as bf5d2be: the harness took the commit when the run ended; fixed)

Embedder: BAAI/bge-small-en-v1.5. 100 histories, 100 questions. Ingest 22.621 s per history; query p50 8.1 ms, p95 14.1 ms.

| type | n | session_recall@5 | any_hit@5 | ndcg@5 | turn_recall@5 | session_recall@10 | any_hit@10 | ndcg@10 | turn_recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 100 | 0.900 | 0.971 | 0.885 | 0.614 | 0.974 | 1.000 | 0.927 | 0.721 |
| knowledge-update | 12 | 0.917 | 1.000 | 0.917 | 0.458 | 1.000 | 1.000 | 0.975 | 0.667 |
| knowledge-update-abstention | 6 | – | – | – | – | – | – | – | – |
| multi-session | 12 | 0.792 | 1.000 | 0.837 | 0.569 | 0.903 | 1.000 | 0.920 | 0.792 |
| multi-session-abstention | 12 | – | – | – | – | – | – | – | – |
| single-session-assistant | 12 | 1.000 | 1.000 | 0.969 | 0.750 | 1.000 | 1.000 | 0.969 | 0.833 |
| single-session-preference | 12 | 0.833 | 0.833 | 0.710 | 0.403 | 1.000 | 1.000 | 0.775 | 0.403 |
| single-session-user | 11 | 1.000 | 1.000 | 1.000 | 0.909 | 1.000 | 1.000 | 1.000 | 0.909 |
| single-session-user-abstention | 6 | – | – | – | – | – | – | – | – |
| temporal-reasoning | 11 | 0.864 | 1.000 | 0.890 | 0.621 | 0.939 | 1.000 | 0.930 | 0.742 |
| temporal-reasoning-abstention | 6 | – | – | – | – | – | – | – | – |

## 2026-10-19 activation sweep (M7.7 part 7) at 0baa405

Age-only activation (benchmark runs present nothing), decay 0.5, weight varied; one row per weight. Session recall@5 and NDCG@5 overall, temporal questions' recall@5, LongMemEval knowledge-update recall@5:

| weight | LME recall@5 | LME ndcg@5 | LME temporal@5 | LME knowledge-update@5 | LoCoMo recall@5 | LoCoMo ndcg@5 | LoCoMo temporal@5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.907 | 0.885 | 0.864 | 0.958 | 0.839 | 0.746 | 0.835 |
| 0.0005 | 0.895 | 0.882 | 0.864 | 0.917 | 0.836 | 0.736 | 0.844 |
| 0.002 | 0.892 | 0.876 | 0.864 | 0.917 | 0.828 | 0.728 | 0.833 |
| 0.005 | 0.906 | 0.881 | 0.864 | 0.917 | 0.813 | 0.711 | 0.816 |
| 0.01 | 0.892 | 0.865 | 0.818 | 0.833 | 0.790 | 0.693 | 0.808 |
| 0.02 | 0.870 | 0.849 | 0.773 | 0.833 | 0.768 | 0.674 | 0.788 |

Any age weighting lowered evidence recall, more so the stronger it was: questions often ask about older conversations. Shipped: no age weighting for conversation messages; memories keep 0.0005 as a tie-breaker by use and confirmation (not measured here).

## 2026-10-19 longmemeval-s (limit 100) at 063455d

Embedder: BAAI/bge-small-en-v1.5. 100 histories, 100 questions. Ingest 0.252 s per history; query p50 3.6 ms, p95 6.4 ms.

| type | n | session_recall@5 | any_hit@5 | ndcg@5 | turn_recall@5 | session_recall@10 | any_hit@10 | ndcg@10 | turn_recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 100 | 0.907 | 0.971 | 0.885 | 0.610 | 0.979 | 1.000 | 0.924 | 0.733 |
| knowledge-update | 12 | 0.958 | 1.000 | 0.942 | 0.458 | 1.000 | 1.000 | 0.975 | 0.708 |
| knowledge-update-abstention | 6 | – | – | – | – | – | – | – | – |
| multi-session | 12 | 0.792 | 1.000 | 0.837 | 0.569 | 0.931 | 1.000 | 0.939 | 0.819 |
| multi-session-abstention | 12 | – | – | – | – | – | – | – | – |
| single-session-assistant | 12 | 1.000 | 1.000 | 0.969 | 0.750 | 1.000 | 1.000 | 0.969 | 0.833 |
| single-session-preference | 12 | 0.833 | 0.833 | 0.680 | 0.375 | 1.000 | 1.000 | 0.742 | 0.403 |
| single-session-user | 11 | 1.000 | 1.000 | 1.000 | 0.909 | 1.000 | 1.000 | 1.000 | 0.909 |
| single-session-user-abstention | 6 | – | – | – | – | – | – | – | – |
| temporal-reasoning | 11 | 0.864 | 1.000 | 0.890 | 0.621 | 0.939 | 1.000 | 0.930 | 0.742 |
| temporal-reasoning-abstention | 6 | – | – | – | – | – | – | – | – |

## 2026-10-19 locomo at 54d1ed5

Embedder: BAAI/bge-small-en-v1.5. 10 histories, 1986 questions. Ingest 0.12 s per history; query p50 2.6 ms, p95 3.4 ms.

| type | n | session_recall@5 | any_hit@5 | ndcg@5 | turn_recall@5 | session_recall@10 | any_hit@10 | ndcg@10 | turn_recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1986 | 0.839 | 0.892 | 0.746 | 0.567 | 0.905 | 0.949 | 0.772 | 0.655 |
| adversarial | 446 | 0.946 | 0.946 | 0.833 | 0.603 | 0.975 | 0.975 | 0.843 | 0.712 |
| multi-hop | 282 | 0.533 | 0.837 | 0.501 | 0.281 | 0.693 | 0.940 | 0.575 | 0.376 |
| open-domain | 96 | 0.511 | 0.663 | 0.461 | 0.277 | 0.640 | 0.772 | 0.509 | 0.335 |
| single-hop | 841 | 0.921 | 0.921 | 0.816 | 0.645 | 0.970 | 0.970 | 0.834 | 0.731 |
| temporal | 321 | 0.835 | 0.854 | 0.737 | 0.645 | 0.901 | 0.916 | 0.763 | 0.713 |
