PROJECT FRAMING PLAN
Team: Akul [surname] and Yohan [surname]
Date: July 14, 2026
Track: AI, Data Science and Computing

RESEARCH QUESTION
Do large language models exhibit systematic confidence-accuracy
miscalibration that scales with task difficulty in a domain-dependent
manner, and does this pattern resemble human metacognitive failure
patterns documented in cognitive psychology?

MOTIVATION
As LLMs are deployed in education and decision-support settings,
whether they express calibrated uncertainty affects user trust.
Human research documents a hard-easy effect (Lichtenstein, Fischhoff &
Phillips, 1982) and related metacognitive failures (Kruger & Dunning,
1999). LLM calibration work exists (Kadavath et al., 2022; Xiong et al.,
2024), but compact multi-model audits that jointly use verbal confidence,
pre-registered difficulty tiers, domain splits, and human-baseline
comparison remain limited for student-led reproducible studies.

METHOD
We evaluate three LLMs (Claude Haiku, Llama 3.1 8B, Mistral 7B) on a
frozen stratified MMLU subset (~180 items; STEM / Social Science /
Humanities; easy/medium/hard tiers; temperature 0.3; 2 trials). Models
provide ANSWER + CONFIDENCE (0–100). We compute ECE, overconfidence gap,
and difficulty-scaling of overconfidence, compared directionally to
published human hard-easy patterns.

ANTICIPATED SOURCES
1. Kruger & Dunning (1999)
2. Lichtenstein, Fischhoff & Phillips (1982)
3. Kadavath et al. (2022)
4. Hendrycks et al. (2021) — MMLU
5. Xiong et al. (2024)

ANTICIPATED LIMITATIONS
1. Verbal confidence ≠ token probabilities
2. MMLU is multiple-choice only
3. Fixed temperature
4. Human baselines use different paradigms — compare directionally
5. Model-version specific (July 2026)
6. Subject-tier difficulty is a proxy, not item-level IRT

EXPECTED DELIVERABLE
A 2–5 page Research Brief with reliability diagrams, overconfidence-by-
difficulty figure, optional domain ECE heatmap, and open GitHub reproduction
pipeline: https://github.com/yoyoyohan/llm-calibration-audit
