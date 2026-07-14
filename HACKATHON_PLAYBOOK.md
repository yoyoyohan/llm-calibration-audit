# NSRI Hackathon 2026 — Merged Playbook
## Confidence Calibration Project (Akul + Yohan)

**This is the single guide to follow.** It merges:
- The complete Akul/Yohan organizational playbook (Drive, literature, Day 1–6 writing, Finale)
- The technical pipeline already in this repo and on GitHub

**Repo (already live):** https://github.com/yoyoyohan/llm-calibration-audit  
**Local folder:** `/Users/yohandeshpande/NSRIHACK/llm-calibration-audit`

Do **not** create a second project called `nsri-hackathon-2026` or rewrite everything under `scripts/`. Use **this** repo’s `src/` pipeline.

---

## Who does what

| Person | Owns |
|---|---|
| **Akul** | Google Drive, Master Tracker, literature, Day 1/2 written outputs, Research Brief writing, citations, AI transparency |
| **Yohan** | Mac setup, Ollama, `.env`, all `python src/...` runs, GitHub pushes, figures, sending numbers to Akul |

**Comm rule:** Keep WhatsApp/Discord open. Message the other when a step finishes.

---

## Critical decisions (merged — don’t argue these mid-week)

| Topic | External mega-guide | **What we actually run** | Why |
|---|---|---|---|
| Domains | 5 (incl. Medicine, Professional) | **3:** STEM, Social_Science, Humanities | Fits time; still domain analysis |
| Models | 4 (incl. Gemma2) | **3:** Claude Haiku, Llama 3.1, Mistral | Faster; Gemini was exploratory only (quota) |
| Qs / cell | 40 | **20** (~180 total) | Finishable |
| Trials | 3 | **2** | Enough for light consistency |
| API key | Pasted into `.py` / Tracker | **`.env` only** (never GitHub) | Security |
| Folders | `scripts/`, `results/` | **`src/`, `data/raw/`, `data/processed/`, `figures/`** | Already built + pushed |
| Overnight | Required for 9k prompts | Recommended for full collect (~1k calls) | Still leave `collect.py` overnight |

**Stretch (only if ahead):** raise `questions_per_cell` / add `gemma2` / set `trials_per_question: 3` in `config/experiment.yaml`.

**Honesty rule for the paper:** say models **resemble** hard–easy / overconfidence *patterns*. Do **not** claim you “proved Dunning–Kruger in LLMs.”

---

## Exact prompt (what models see)

Defined permanently in `src/prompts.py`:

```text
Answer the following multiple-choice question. Then rate your confidence.

Question: {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Respond in EXACTLY this format with no other text:
ANSWER: [A/B/C/D]
CONFIDENCE: [integer 0-100]
```

---

## Method (one paragraph — use in Day 1 / Day 2 docs)

We evaluate Claude Haiku (Anthropic API), Llama 3.1 (Ollama), and Mistral (Ollama) on a frozen, difficulty-stratified MMLU subset (~180 questions; 3 domains × 3 difficulty tiers; 2 trials; temperature 0.3). Difficulty tiers are **pre-registered** in `config/difficulty_map.yaml` from MMLU subject difficulty patterns (Hendrycks et al., 2021). Each model returns ANSWER + CONFIDENCE (0–100). We compute accuracy, overconfidence gap, ECE, and how overconfidence scales with difficulty, and compare **directionally** to human hard–easy baselines (Lichtenstein et al., 1982; Kruger & Dunning, 1999).

**Metrics:** accuracy · mean confidence · overconfidence gap (conf − acc) · ECE · difficulty→overconfidence slope  
**Hero figure:** `figures/figure2_overconfidence_by_difficulty.png`

---

# PART 0 — TONIGHT / ASAP (before deep Day 1 research)

## 0A — AKUL

### A1. Shared Google Drive
1. drive.google.com → **New → Folder** → `NSRI Hackathon 2026 — Akul and Yohan`
2. Inside, create:
   - `0 — Admin`
   - `1 — Literature`
   - `2 — Data`
   - `3 — Figures`
   - `4 — Drafts`
   - `5 — Final Submission`
3. Share folder with Yohan (**Editor**)

### A2. Master Tracker
In `0 — Admin`, create Google Doc: **`Master Tracker — NSRI 2026`**

Paste (edit links):

```text
NSRI HACKATHON 2026 — MASTER TRACKER
Team: Akul and Yohan
Track: AI, Data Science and Computing
Project: Confidence Calibration in LLMs Across Difficulty and Domain

=== DAILY LOG ===
DAY 1 (July 13):
- [ ] Akul done:
- [ ] Yohan done:
- [ ] Pending:

DAY 2 (July 14):
- [ ] Akul done:
- [ ] Yohan done:
- [ ] Pending:

DAY 3 (July 15):
- [ ] Akul done:
- [ ] Yohan done:
- [ ] Pending:

DAY 4 (July 16):
- [ ] Akul done:
- [ ] Yohan done:
- [ ] Pending:

DAY 5 (July 17):
- [ ] Akul done:
- [ ] Yohan done:
- [ ] Pending:

DAY 6 (July 18):
- [ ] Submitted:
- [ ] Confirmation received:

=== RESEARCH QUESTION ===
Do large language models exhibit systematic confidence-accuracy
miscalibration that varies with task difficulty in a domain-dependent
manner, and does this miscalibration resemble patterns of human
metacognitive failure documented in cognitive psychology?

=== KEY LINKS ===
GitHub: https://github.com/yoyoyohan/llm-calibration-audit
Results CSV (after collect): data/raw/final_results.csv in the repo
Anthropic API key: DO NOT PASTE HERE — send to Yohan on WhatsApp only

=== PAPERS ===
1. Kruger & Dunning 1999
2. Lichtenstein, Fischhoff & Phillips 1982
3. Kadavath et al. 2022
4. Hendrycks et al. 2021 (MMLU)
5. Xiong et al. 2024
```

### A3. Anthropic API key (Claude Haiku)
1. https://console.anthropic.com → API keys → Create  
2. Send key to Yohan on **WhatsApp/Discord only** (not Drive, not GitHub)

### A4. Literature start
In `1 — Literature`, Doc: **`Literature Notes`**

For each paper (abstract + conclusion tonight is enough):

```text
PAPER N: Author (Year)
Full title:
What they studied:
What they found:
Key numbers:
What OUR study does that this paper does NOT:
URL:
```

Papers: Kruger & Dunning 1999 · Lichtenstein et al. 1982 · Kadavath 2022 · Hendrycks 2021 · Xiong 2024  
MMLU PDF: https://arxiv.org/abs/2009.03300 (note Table 1 human accuracies)

---

## 0B — YOHAN (map to THIS repo — already partly done)

### Done already
- [x] Code scaffolded in `llm-calibration-audit/`
- [x] Pushed to https://github.com/yoyoyohan/llm-calibration-audit

### B1. Specs
Apple menu → About This Mac → RAM (≥8 GB needed for Ollama; else Claude-API-only fallback)

### B2. One-time Mac setup (copy exactly)

```bash
cd /Users/yohandeshpande/NSRIHACK/llm-calibration-audit

python3 --version   # need 3.9+

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
open -e .env
# set: ANTHROPIC_API_KEY=...   (from Akul via WhatsApp)
```

### B3. Ollama

```bash
# Install from https://ollama.com if needed, then:
ollama pull llama3.1
ollama pull mistral
# OPTIONAL stretch:
# ollama pull gemma2

ollama list
```

### B4. Verify stack

```bash
cd /Users/yohandeshpande/NSRIHACK/llm-calibration-audit
source .venv/bin/activate
python src/setup_verify.py
# or:
python src/smoke_test.py --both
```

Everything should OK. Fix errors before collecting data.

### B5. Do **not** hardcode the API key into Python files
Use `.env` only. `.gitignore` already blocks `.env`.

---

# PART 1 — DAY 1 (Jul 13)

**NSRI required output:** track, team, research question, initial method plan.

## Zoom (12:00–1:30 PM ET)
Both join. Note registration / team formalization instructions in Master Tracker.

## AKUL — Day 1 (after Zoom)
1. Confirm team registration as organizers instruct. Track: **AI, Data Science and Computing**
2. Deepen lit notes (15–20 min/paper): KD99, Lichtenstein, MMLU Table 1, Kadavath, Xiong  
   For each: “What we do that they don’t”
3. Create Drive doc `0 — Admin` / **`Day 1 Output`** using `docs/day1_output_template.md` in this repo (copy into Docs). **Align method with scoped 3 domains / 3 models / 2 trials** (not the mega 5×4×3 plan unless you stretch).
4. Message Yohan when Day 1 Output is done.

## YOHAN — Day 1 (after Zoom) — technical

```bash
cd /Users/yohandeshpande/NSRIHACK/llm-calibration-audit
source .venv/bin/activate

# 1) Freeze question bank (commit this file once)
python src/build_question_bank.py

# 2) Tiny pipeline test
python src/collect.py --smoke
python src/analyze.py

# 3) Start full collection (leave running / overnight)
python src/collect.py
# if interrupted later:
# python src/collect.py --resume
```

```bash
git add data/question_bank.csv
git commit -m "Day 1: freeze stratified MMLU question bank"
git push origin main
```

Message Akul: “bank frozen; collect started; GitHub updated.”

**End-of-day gate:** Day 1 Output written · bank CSV exists · smoke OK · full collect running.

---

# PART 2 — DAY 2 (Jul 14)

**NSRI required output:** 1-page framing plan.

## AKUL
1. Zoom notes → Master Tracker  
2. Message Yohan: progress on `data/raw/final_results.csv`  
3. Write Drive doc **`Day 2 Framing Plan`** using `docs/day2_framing_template.md` (scoped method)  
4. Continue lit: one sentence gap per paper  

## YOHAN

```bash
source .venv/bin/activate
cd /Users/yohandeshpande/NSRIHACK/llm-calibration-audit

python -c "
import pandas as pd
df = pd.read_csv('data/raw/final_results.csv')
print('rows', len(df))
print(df.groupby('model_id').size())
print('parse_ok', df['parse_ok'].mean())
print(df.groupby('difficulty')['is_correct'].mean())
"
```

Send numbers to Akul. Keep/resume collect. If parse_ok ≪ 80%, inspect:

```bash
python -c "
import pandas as pd
df=pd.read_csv('data/raw/final_results.csv')
print(df.loc[~df['parse_ok'], 'raw_response'].head(10).tolist())
"
```

Optional: run `python src/analyze.py` on partial data for early figures.

---

# PART 3 — DAY 3 (Jul 15) — push hard

## AKUL (writing day)
1. Finish gap statements for all 5 papers  
2. Open / expand `paper/research_brief.md` (or Drive `4 — Drafts/Research Brief Draft v1`)  
3. Write Intro, Related Work, Method with **placeholders** `[INSERT N]` for results  
4. Limitations can be drafted early (templates already in `paper/research_brief.md`)  
5. Use GitHub URL: https://github.com/yoyoyohan/llm-calibration-audit  

Keep body length compatible with **2–5 page** NSRI brief (intro shorter than full conference draft).

## YOHAN (data + figures)

```bash
source .venv/bin/activate
python src/collect.py --resume   # finish if needed
python src/analyze.py            # writes data/processed/* and figures/*
```

Send Akul:
- parse rate, n rows  
- accuracy / mean confidence by model  
- contents of `data/processed/overall_by_model.csv`  
- `dk_hard_easy_slopes.csv`  
- PNGs in `figures/` (upload to Drive `3 — Figures`)

```bash
git add data/raw/final_results.csv data/processed figures
git commit -m "Day 3: results tables and figures"
git push origin main
```

**Kill list if behind:** skip domain heatmap story · don’t add Gemma2 · don’t expand to 5 domains.

---

# PART 4 — DAY 4 (Jul 16)

## AKUL
Fill every `[INSERT]` with Yohan’s numbers. Write Results + Discussion:
- Does overconfidence rise easy→hard?  
- Domain differences (if figure 3 ready)  
- Deployment implication (don’t trust confident wrong answers on hard items)  
Upload figures into the brief/appendix.

## YOHAN
Re-run analysis once on final CSV. Confirm figures. Update README “Key findings” with real numbers. Push.

```bash
python src/analyze.py
git add -A
git commit -m "Day 4: final figures and processed metrics"
git push origin main
```

---

# PART 5 — DAY 5 (Jul 17)

## AKUL
1. Write abstract **last** (≤250 words; include 1–2 real numbers)  
2. Full read-aloud edit  
3. Verify every citation on Scholar (APA in References)  
4. Finalize AI Use Transparency (template in `paper/research_brief.md`)  

## YOHAN
```bash
python src/final_verification.py
```
Send full output to Akul. Ensure no secrets in repo. Final push:

```bash
git add -A
git commit -m "Day 5: verification + docs for submission"
git push origin main
```

---

# PART 6 — DAY 6 (Jul 18) — SUBMIT ~5 PM ET

1. Zoom: note exact submission form rules  
2. Joint read-through of brief + figures  
3. Export PDF (2–5 pages body; refs/appendix extra)  
4. Package checklist:

```text
□ Title
□ Track: AI, Data Science and Computing
□ Authors
□ Abstract ≤250 words
□ Research Brief PDF
□ References verified
□ AI Use Transparency Statement
□ GitHub public: https://github.com/yoyoyohan/llm-calibration-audit
□ Figures in appendix / Drive
□ Raw CSV in repo: data/raw/final_results.csv
```

5. Submit → screenshot confirmation → Drive `5 — Final Submission`  
6. Update Master Tracker  

---

# PART 7 — Grand Finale (Jul 25) IF selected

Use your external guide’s 3-minute script + 6 slides, but **update numbers** to scoped experiment (3 models / ~180 Qs / 2 trials). Hero slide = Figure 2. Practice judge Qs from that guide (verbal vs token probs; MMLU contamination; Kadavath difference).

---

# Command cheat sheet (Yohan)

```bash
cd /Users/yohandeshpande/NSRIHACK/llm-calibration-audit
source .venv/bin/activate

python src/setup_verify.py
python src/smoke_test.py --both
python src/build_question_bank.py
python src/collect.py --smoke
python src/collect.py
python src/collect.py --resume
python src/analyze.py
python src/final_verification.py
```

| Script | Purpose |
|---|---|
| `src/setup_verify.py` | Imports + Ollama + Claude connectivity |
| `src/smoke_test.py` | Ollama parse smoke test |
| `src/smoke_claude.py` | Claude Haiku parse smoke test |
| `src/build_question_bank.py` | Freeze MMLU sample |
| `src/collect.py` | Run all model calls → `data/raw/final_results.csv` |
| `src/analyze.py` | Tables + figures 1–3 |
| `src/final_verification.py` | Numbers dump for Results section |

---

# What wins (shared)

1. Real data + reproducible GitHub  
2. Clear Figure 2 story  
3. Honest limitations  
4. Submit early (5 PM, not 11:59)

**Start now:** Akul → Drive + Tracker + lit + API key to Yohan. Yohan → `.env` + Ollama + `setup_verify` + `build_question_bank` + `collect --smoke` → overnight `collect.py`.
