TEAM: Akul [last name] and Yohan [last name]
TRACK: AI, Data Science and Computing
DATE: July 13, 2026

RESEARCH QUESTION:
Do large language models exhibit systematic confidence-accuracy
miscalibration that scales with task difficulty in a domain-dependent
manner, and does this pattern resemble the hard-easy effect and
Dunning-Kruger pattern documented in human cognitive psychology
literature?

WHY THIS MATTERS:
LLMs are used in tutoring, advice, and decision support. A model that
is confidently wrong can be more harmful than one that is merely wrong,
because users defer to confident answers. We need to know whether
overconfidence gets worse as questions get harder, and whether that
depends on knowledge domain.

INITIAL METHOD PLAN (SCOPED — matches our GitHub pipeline):
1. Question bank: Stratified sample from MMLU (Hendrycks et al., 2021),
   difficulty tiers from pre-registered subject maps tied to published
   MMLU difficulty patterns. Domains: STEM, Social Science, Humanities.
   Target ~20 questions per domain × difficulty (~180 total).

2. Models: Claude Haiku (Anthropic API), Llama 3.1 (Ollama), Mistral (Ollama).
   (Gemini was explored early but not retained as a primary arm due to API quota.)

3. Protocol: For each question, each model returns ANSWER (A–D) and
   CONFIDENCE (0–100). Temperature 0.3. Two trials per question×model.

4. Analysis: ECE, overconfidence gap, difficulty→overconfidence slope,
   domain ECE comparison; qualitative comparison to human hard-easy
   baselines (Lichtenstein et al., 1982; Kruger & Dunning, 1999).

5. Code/data: https://github.com/yoyoyohan/llm-calibration-audit

SOLO OR TEAM: Team of 2
