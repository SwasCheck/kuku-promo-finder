import csv, json, re, time
from pathlib import Path
from .parse import load_pack, fmt, Episode
from .candidates import make_candidates, MIN_LEN, MAX_LEN, Candidate
from . import judge as J


def _t(s):
    m = re.match(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$", str(s))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def _snap(ep: Episode, c: Candidate, adj):
    """Model may suggest a better cut. Accept it only if it snaps to real line boundaries and stays 20-60s."""
    if not adj: return c.start, c.end, False
    s, e = _t(adj.get("start")), _t(adj.get("end"))
    if s is None or e is None: return c.start, c.end, False
    lines = [x for x in ep.events if x.kind in ("dialogue", "action")]
    s2 = min((x.start for x in lines), key=lambda v: abs(v - s))
    e2 = min((x.end for x in lines), key=lambda v: abs(v - e))
    if MIN_LEN <= e2 - s2 <= MAX_LEN and abs(s2 - s) < 3 and abs(e2 - e) < 3:
        return s2, e2, (abs(s2 - c.start) > .05 or abs(e2 - c.end) > .05)
    return c.start, c.end, False


def _overlap(a, b):
    return a["ep"] == b["ep"] and min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0.3 * min(a["dur"], b["dur"])


def run(pack, out, mode="llm", provider=None, model=None, keep=10, top_n=15, per_ep_top=3, only=None, log=print):
    out = Path(out); (out / "per_episode").mkdir(parents=True, exist_ok=True); (out / "llm_raw").mkdir(exist_ok=True)
    eps = load_pack(pack)
    if only: eps = [e for e in eps if e.number in only]
    llm = None
    if mode == "llm":
        from .llm import LLM
        llm = LLM(provider, model)
        log(f"model: {llm.provider}/{llm.model}")
    items, cand_index, failures = [], {}, []
    for ep in eps:
        cands, pool = make_candidates(ep, keep=keep)
        for c in cands: cand_index[c.cid] = c
        base = {c.cid: dict(id=c.cid, ep=ep.number, start=c.start, end=c.end, dur=c.dur, window=c.label(),
                            rule_score=round(c.score, 3), rule_parts=c.parts, hook_line=c.hook_text,
                            cliffhanger=c.end_text, why="", weakness="", overall=round(c.score * 10, 1), source="rules")
                for c in cands}
        if mode == "llm":
            raw_path = out / "llm_raw" / f"ep{ep.number:02d}.json"
            try:
                if raw_path.exists():                      # resume support: free tiers have daily limits
                    data = json.loads(raw_path.read_text())["parsed"]
                else:
                    data, raw = J.judge_episode(llm, ep, cands)
                    raw_path.write_text(json.dumps({"model": f"{llm.provider}/{llm.model}", "parsed": data, "raw": raw}, indent=1))
                for r in data.get("clips", []):
                    b = base.get(r.get("id"))
                    if not b: continue
                    c = cand_index[b["id"]]
                    s, e, changed = _snap(ep, c, r.get("adjust"))
                    b.update(start=s, end=e, dur=e - s, window=f"{fmt(s)}-{fmt(e)}", adjusted=changed,
                             overall=float(r.get("overall", 0)), hook_line=r.get("hook_line") or b["hook_line"],
                             cliffhanger=r.get("cliffhanger_text") or b["cliffhanger"], why=r.get("why", ""),
                             weakness=r.get("weakness", ""), llm_scores={k: r.get(k) for k in ("hook_score", "cliffhanger_score", "standalone_score", "overall")},
                             source="llm")
                log(f"ep{ep.number:02d}: {len(cands)} candidates (of {pool} windows) judged")
            except Exception as ex:                        # one bad episode must not kill the run; fall back to rules
                failures.append((ep.number, str(ex)[:200]))
                log(f"ep{ep.number:02d}: LLM failed ({str(ex)[:120]}); using rule scores for this episode")
        else:
            log(f"ep{ep.number:02d}: {len(cands)} candidates (of {pool} windows) scored by rules")
        (out / "per_episode" / f"ep{ep.number:02d}_candidates.json").write_text(json.dumps(list(base.values()), indent=1))
        ranked = sorted(base.values(), key=lambda x: -x["overall"])
        items += ranked[:per_ep_top]

    # ---- global ranking ----
    items.sort(key=lambda x: -x["overall"])
    by_id = {i["id"]: i for i in items}
    final, used = [], []
    def add(ids, reason, joined=False):
        segs = [by_id[i] for i in ids if i in by_id]
        if not segs or len(segs) != len(ids): return False
        if sum(s["dur"] for s in segs) < MIN_LEN - .01 or sum(s["dur"] for s in segs) > MAX_LEN + .01: return False
        if any(_overlap(s, u) for s in segs for u in used): return False
        used.extend(segs)
        final.append(dict(segments=[dict(episode=s["ep"], start=fmt(s["start"]), end=fmt(s["end"])) for s in segs],
                          total_seconds=round(sum(s["dur"] for s in segs), 1), joined=joined or len(segs) > 1,
                          hook=segs[0]["hook_line"], cliffhanger=segs[-1]["cliffhanger"],
                          why_it_works=" ".join(s["why"] for s in segs[:1]) or "Rule-based pick (no AI judgement): "
                              + ", ".join(f"{k} {v}" for k, v in segs[0]["rule_parts"].items()),
                          risk=segs[0].get("weakness", ""), score=round(sum(s["overall"] for s in segs) / len(segs), 1),
                          ranking_reason=reason, ids=ids, source=segs[0]["source"]))
        return True
    if mode == "llm" and items:
        try:
            ctx = "\n".join(f"Ep{e.number}: {e.logline}" for e in eps)
            data, raw = J.rank_global(llm, ctx, items[:120], top_n)
            (out / "llm_raw" / "global_ranking.json").write_text(json.dumps({"parsed": data, "raw": raw}, indent=1))
            for r in sorted(data.get("ranking", []), key=lambda x: x.get("rank", 99)):
                if len(final) >= top_n: break
                add(r.get("ids", []), r.get("reason", ""))
        except Exception as ex:
            failures.append(("global", str(ex)[:200])); log(f"global ranking failed: {ex}; falling back to score order")
    per_ep = {}
    for it in items:                                         # fill / rules-mode: best score first, max 2 per episode for variety
        if len(final) >= top_n: break
        if per_ep.get(it["ep"], 0) >= 2 or it["dur"] < MIN_LEN: continue
        if add([it["id"]], "Ranked by score" + (" (AI ranking unavailable or filled slot)" if mode == "llm" else " (rules only)")):
            per_ep[it["ep"]] = per_ep.get(it["ep"], 0) + 1
    for i, p in enumerate(final, 1): p["rank"] = i

    (out / "promos_ranked.json").write_text(json.dumps(final, indent=1))
    with open(out / "promos_ranked.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank", "segments (episode start-end)", "total_s", "hook", "cliffhanger", "why_it_works", "risk", "score", "source"])
        for p in final:
            w.writerow([p["rank"], " + ".join(f'Ep{s["episode"]} {s["start"]}-{s["end"]}' for s in p["segments"]), p["total_seconds"],
                        p["hook"], p["cliffhanger"], p["why_it_works"], p["risk"], p["score"], p["source"]])
    md = [f"# Ranked promos ({'AI judge: ' + llm.provider + '/' + llm.model if llm else 'rules only, no AI'})\n"]
    for p in final:
        seg = " + ".join(f'**Ep {s["episode"]}** {s["start"]}–{s["end"]}' for s in p["segments"])
        md += [f"## #{p['rank']}  {seg}  ({p['total_seconds']}s{', joined' if p['joined'] else ''})",
               f"- **Hook (first 3s):** {p['hook']}", f"- **Cliffhanger:** {p['cliffhanger']}",
               f"- **Why it works:** {p['why_it_works']}", f"- **Risk:** {p['risk'] or 'n/a'}",
               f"- **Why this rank:** {p['ranking_reason']}", ""]
    (out / "promos_ranked.md").write_text("\n".join(md))
    if failures: (out / "failures.json").write_text(json.dumps(failures, indent=1))
    log(f"done: {len(final)} promos -> {out}/promos_ranked.md" + (f" ({len(failures)} failures)" if failures else ""))
    return final
