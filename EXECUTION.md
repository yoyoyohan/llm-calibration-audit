# EXECUTION GUIDE — Short technical checklist

**Prefer the full merged guide:** [`HACKATHON_PLAYBOOK.md`](./HACKATHON_PLAYBOOK.md)  
(Drive, literature, Day 1–6 writing, Finale + exact `src/` commands.)

**Deadline:** Sat Jul 18, 2026 · 11:59 PM ET (submit by ~5 PM ET)  
**Project folder:** `llm-calibration-audit/`  
**Split:** Yohan = code/runs · Akul = literature + Research Brief  

Below is the condensed Yohan tech path.

---

## Before anything else (30–45 min) — BOTH of you tonight

### 1. Create GitHub repo
1. Create a **public** repo on GitHub named e.g. `llm-calibration-audit`
2. From this folder:

```bash
cd /Users/yohandeshpande/NSRIHACK/llm-calibration-audit
git init
git add .
git commit -m "Initial calibration audit pipeline for NSRI 2026"
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/llm-calibration-audit.git
git push -u origin main
```

### 2. Python + keys

```bash
cd /Users/yohandeshpande/NSRIHACK/llm-calibration-audit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

- Get an Anthropic API key: https://console.anthropic.com/  
- Put it in `.env` as `ANTHROPIC_API_KEY=...`  
- **Never commit `.env`**

### 3. Ollama

```bash
# install from https://ollama.com if needed
ollama pull llama3.1
ollama pull mistral
# confirm daemon is up
curl http://localhost:11434/api/tags
```

### 4. Smoke test (must pass before claiming Day 1 done)

```bash
source .venv/bin/activate
python src/smoke_test.py --ollama
python src/smoke_claude.py
```

You want parsed output like `('B', 80, True)`.

### 5. Lock claims language
In the paper you will say:
- “resembles hard–easy / overconfidence patterns”
- NOT “LLMs have the Dunning–Kruger effect”

---

## Method you are running (memorize this)

| Piece | Choice |
|---|---|
| Dataset | MMLU test split (`cais/mmlu`) |
| Domains | STEM, Social_Science, Humanities |
| Difficulty | Pre-registered subject tiers in `config/difficulty_map.yaml` |
| Qs / cell | 20 per domain×difficulty → ~180 total |
| Models | Claude Haiku, Llama 3.1, Mistral |
| Trials | 2 |
| Temperature | 0.3 |
| Metrics | accuracy, confidence, overconfidence gap, ECE, slope over difficulty |
| Prompt | see `src/prompts.py` / README |

**Primary finding you hope to show:** overconfidence gap rises from easy → hard (positive slope), with ECE by model.

---

## Day 1 — Mon Jul 13 (2–3h)

### Yohan
- [ ] Setup + smoke tests done  
- [ ] Build bank:

```bash
source .venv/bin/activate
python src/build_question_bank.py
# optional tiny bank for pipeline debug:
# python src/build_question_bank.py --per-cell 5 --out data/question_bank_smoke.csv
```

- [ ] Commit `data/question_bank.csv` once (freeze seed)  
- [ ] Start tiny collect:

```bash
python src/collect.py --smoke
python src/analyze.py   # should work on smoke data
```

- [ ] If smoke OK, start full collect in background:

```bash
python src/collect.py
# if interrupted later:
python src/collect.py --resume
```

Claude Haiku uses ~1s sleep between calls. Ollama is local CPU — overnight OK.

### Akul
- [ ] Create Google Doc / edit `paper/research_brief.md` with all section headings  
- [ ] Read abstracts / note 3 bullets each:
  - Kruger & Dunning (1999)
  - Lichtenstein, Fischhoff & Phillips (1982) hard–easy
  - Kadavath et al. (2022)
  - Hendrycks et al. (2021) MMLU
  - Xiong et al. (2024) confidence elicitation  
- [ ] Draft **Research question + Motivation** (½ page)

### End-of-day gate
Must have: repo online, question bank frozen, ≥ smoke results CSV, brief skeleton.

---

## Day 2 — Tue Jul 14 (2–3h)

### NSRI expectation
1-page framing: question, method, sources, limitations.

### Akul
- [ ] Write that 1-pager into the Doc (this becomes Method/Related Work seed)

### Yohan
- [ ] Finish / monitor collection  
- [ ] Print sanity checks in Python or glance at CSV:
  - parse_ok rate > ~80% ideally  
  - easy accuracy > hard accuracy **on average**  
- [ ] If easy ≯ hard: do **not** reshuffle difficulty after looking at model scores for your narrative; instead discuss map limitations. (Difficulty was pre-registered.)

### End-of-day gate
Most of `data/raw/final_results.csv` present (or clear overnight finish plan).

---

## Day 3 — Wed Jul 15 (2–3h+)

### Yohan
```bash
python src/analyze.py
```
- [ ] Open `figures/figure2_overconfidence_by_difficulty.png` — this is your hero figure  
- [ ] Open `data/processed/dk_hard_easy_slopes.csv` and `overall_by_model.csv`  
- [ ] Push figures + processed tables to GitHub  

### Akul
- [ ] Draft **Method** exactly matching what you ran (n questions, models, trials, temp, metrics)  
- [ ] Draft **Results** bullets tied to real numbers from processed CSVs  

### Kill list if behind
1. Drop figure 3 (domain heatmap)  
2. Ignore consistency analysis (not required by code default)  
3. Drop one model only as last resort  

---

## Day 4 — Thu Jul 16 (2–3h)

### Yohan
- [ ] README “Key findings” filled with real numbers  
- [ ] Ensure reproduction steps work on a clean shell  

### Akul
- [ ] Full rough Research Brief (2–5 pages body)  
- [ ] Limitations drafted (copy from `paper/research_brief.md` checklist)  
- [ ] AI Use Transparency Statement draft  

---

## Day 5 — Fri Jul 17 (2–3h)

- [ ] Abstract last (≤250 words) — include **one real number**  
- [ ] Verify every reference exists (Scholar)  
- [ ] Export PDF  
- [ ] Optional: record 3-minute walkthrough (screen + voice)  
- [ ] Both: practice explaining figure 2 aloud  

---

## Day 6 — Sat Jul 18 (6–7h) — SUBMIT

### Morning
- [ ] Freeze code (`git tag submission` optional)  
- [ ] Re-run `python src/analyze.py` once for final figures  

### Afternoon (target **5:00 PM ET**)
Submit Google Form with:
- [ ] Title + track  
- [ ] Authors  
- [ ] Abstract ≤250 words  
- [ ] Research Brief PDF (2–5 pages excl. refs/appendix)  
- [ ] References + AI transparency  
- [ ] Public GitHub URL  
- [ ] (Optional) video link  

---

## Commands cheat sheet

```bash
source .venv/bin/activate

# connectivity
python src/smoke_test.py --both

# data
python src/build_question_bank.py
python src/build_question_bank.py --per-cell 5   # smaller

# collect
python src/collect.py --smoke
python src/collect.py
python src/collect.py --resume
python src/collect.py --models llama3.1 mistral   # local models only
python src/collect.py --models claude_haiku        # API arm

# analyze
python src/analyze.py
```

### Config knobs (`config/experiment.yaml`)
- `questions_per_cell` — default 20  
- `trials_per_question` — default 2  
- `temperature` — default 0.3  
- `anthropic_sleep_seconds` — raise if Anthropic rate-limits  

---

## Paper metrics glossary (for writing)

| Metric | Meaning |
|---|---|
| **Accuracy** | Fraction of correct multiple-choice answers |
| **Mean confidence** | Average stated confidence / 100 |
| **Overconfidence gap** | mean confidence − accuracy (>0 = overconfident) |
| **ECE** | Expected Calibration Error; 0 = perfect calibration |
| **DK / hard–easy slope** | Regression slope of overconfidence on difficulty (1/2/3). Positive ≈ more overconfident on harder items |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ANTHROPIC_API_KEY missing` | Create `.env` from `.env.example` |
| Ollama connection error | Start Ollama app; `ollama list` |
| MMLU download slow/fails | Re-run builder; need network; HF cache will reuse |
| Empty domain×difficulty cell | Warning is OK; fewer than 180 Qs still fine — report actual n |
| Low parse_ok | Models not following format; keep prompt strict; report failure rate in limitations |
| collect interrupted | `python src/collect.py --resume` |
| Import errors running scripts | Always `cd` repo root and use `python src/...` with venv on |

---

## What “done” looks like

You can submit when all are true:

1. Frozen question bank in repo  
2. Full (or clearly documented partial) results CSV  
3. Three figures (or at least figures 1–2)  
4. Processed summary tables with slopes/ECE  
5. Research Brief PDF with honest limitations  
6. Public GitHub README someone else could follow  
7. AI transparency statement  

**Start tonight:** smoke test → build bank → `--smoke` collect → push to GitHub.
