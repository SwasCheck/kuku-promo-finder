# Promo Finder

Reads timestamped microdrama screenplays and returns ranked 20-60s promo clips
(episode + start/end of every segment, hook, cliffhanger, why it works, risk).

## How it works
1. **Code (no AI)** `parse.py`, `candidates.py`: parse screenplays, cut every legal 20-60s window that starts on a
   line start and ends on a line end, score with transparent rules, keep ~10 diverse windows per episode.
2. **AI judge** `judge.py`: one call per episode. Model reads the full episode and rates each candidate
   (hook, cliffhanger, stands-alone), may nudge cut points. Code snaps/validates any change.
3. **AI ranker** `judge.py`: one call over the shortlisted clips ranks the top N and may stitch 2-3 clips
   (validated by code: real ids, in order, total 20-60s, no overlaps).
4. **Fallbacks:** a failed episode falls back to rule scores; `--mode heuristic` runs with no AI and no key.

## Run
    export GEMINI_API_KEY=...            # free key: aistudio.google.com
    python -m promo_finder --pack path/to/episodes --out outputs
Swap the model with flags or env vars (no code change):

    python -m promo_finder --pack ... --provider anthropic --model <model-id>    # ANTHROPIC_API_KEY
    python -m promo_finder --pack ... --provider openai --model <model-id>       # OPENAI_API_KEY (+ OPENAI_BASE_URL for Groq/OpenRouter)
    python -m promo_finder --pack ... --mode heuristic                           # no AI, no key
Options: `--top 15`, `--episodes 1,2,3` (subset). Re-running resumes from `outputs/llm_raw/` (delete it to redo).
Python 3.9+, standard library only.

## Output folder
`promos_ranked.md|csv|json` (the answer), `per_episode/` (every candidate + scores), `llm_raw/` (raw model replies), `failures.json` (if any).

Note: do not commit the screenplays; they are Kuku's confidential content.
