import argparse
from .pipeline import run

ap = argparse.ArgumentParser(prog="promo_finder", description="Find ranked 20-60s promo clips in timestamped microdrama screenplays.")
ap.add_argument("--pack", required=True, help="folder with episode-*.txt screenplays")
ap.add_argument("--out", default="outputs")
ap.add_argument("--mode", choices=["llm", "heuristic"], default="llm")
ap.add_argument("--provider", choices=["gemini", "anthropic", "openai"], help="default: env PROMO_PROVIDER or gemini")
ap.add_argument("--model", help="default: env PROMO_MODEL or a sensible default per provider")
ap.add_argument("--top", type=int, default=15, help="how many ranked promos to output")
ap.add_argument("--episodes", help="comma list to run a subset, e.g. 1,2,3")
a = ap.parse_args()
run(a.pack, a.out, a.mode, a.provider, a.model, top_n=a.top,
    only={int(x) for x in a.episodes.split(",")} if a.episodes else None)
