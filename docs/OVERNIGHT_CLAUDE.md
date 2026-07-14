# Overnight Claude Haiku collect (merges into existing Llama/Mistral results)

## 1. Put your key in `.env` (never commit this file)

```bash
cd /Users/yohandeshpande/NSRIHACK/llm-calibration-audit
open -e .env
```

Add this line:

```bash
ANTHROPIC_API_KEY=paste_your_real_key_here
```

Save.

## 2. Activate venv + smoke-test Claude only

```bash
cd /Users/yohandeshpande/NSRIHACK/llm-calibration-audit
source .venv/bin/activate
python src/smoke_claude.py
```

You want something like `PARSED: ('B', 90, True)`.

Tiny collect test (3 questions, Claude only — **safe**, will not wipe Llama/Mistral):

```bash
python src/collect.py --smoke --models claude_haiku
```

Check:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/raw/final_results.csv'); print(df['model_id'].value_counts())"
```

You should still see `llama3.1` and `mistral` plus a few `claude_haiku` rows.

## 3. Start full Claude overnight (~360 calls)

```bash
cd /Users/yohandeshpande/NSRIHACK/llm-calibration-audit
source .venv/bin/activate
python src/collect.py --models claude_haiku
```

- Leaves Llama/Mistral rows intact  
- Appends Claude for all 180 Q × 2 trials  
- ~1s sleep between calls → often **~1–2 hours** (can be longer)  
- Cost: well under $1 of your $4.81  

If it stops mid-night:

```bash
python src/collect.py --models claude_haiku
```

(safe to re-run; skips cells already done)

**Do not use `--overwrite`.**

## 4. In the morning

```bash
source .venv/bin/activate
python -c "import pandas as pd; df=pd.read_csv('data/raw/final_results.csv'); print(df['model_id'].value_counts()); print(df[df.model_id=='claude_haiku']['parse_ok'].mean())"
python src/analyze.py
python src/final_verification.py
```

Primary paper models: **Claude Haiku + Llama 3.1 + Mistral**  
An early Gemini Flash attempt was blocked by free-tier quota and is not part of the primary analysis.

## 5. Optional push

```bash
git add config/experiment.yaml src/clients.py src/collect.py src/smoke_claude.py .env.example
git add data/raw/final_results.csv data/processed figures
git commit -m "Add Claude Haiku as third model and merge into calibration results"
git push origin main
```
