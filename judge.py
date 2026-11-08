"""Step 2 (AI): judge the code-made candidates; Step 3 (AI): rank them across the show and propose joins."""
import json
from .parse import Episode, fmt, transcript
from .candidates import Candidate, MIN_LEN, MAX_LEN

BRIEF = """You are a short-drama (microdrama) promo editor for vertical-video apps like ReelShort and DramaBox.
A PROMO is a 20-60 second clip taken from the show and posted on Instagram/TikTok/YouTube to pull in NEW viewers.
A good promo:
 1. HOOK: in the first 3 seconds something makes a scrolling viewer stop (a slap, a reveal, a shocking line, a threat).
 2. CLIFFHANGER: it cuts at an unresolved moment, so the viewer must know what happens next. Not a resolved scene end.
 3. STANDS ALONE: someone who has never seen the show understands who is who and what is at stake, without episode context.
 Penalise: slow set-up, openers that depend on earlier events ("What?", "He did it?"), endings that resolve the tension, spoiling the big twist in the clip itself.
Judge ONLY from the screenplay text provided. Never invent lines, names or events."""


def _cand_block(cs):
    return "\n".join(f'- {c.cid}  {fmt(c.start)}-{fmt(c.end)} ({c.dur:.0f}s)\n    opens: {c.hook_text[:160]}\n    ends:  {c.end_text[:160]}' for c in cs)


def judge_episode(llm, ep: Episode, cands):
    system = BRIEF + "\nReply with ONE JSON object only."
    user = f"""EPISODE {ep.number}
Logline: {ep.logline}
Continuity (context a cold viewer would NOT have): {ep.continuity}

SCREENPLAY (timestamps mm:ss.s, camera cues omitted):
{transcript(ep)}

CANDIDATE CLIPS (already cut on real line boundaries, each 20-60s):
{_cand_block(cands)}

Task: rate EVERY candidate. Be critical, most clips of a show are NOT good promos; use the full 0-10 range.
Optionally adjust a clip's start/end (mm:ss.s, must keep it 20-60s, start at a line start, end at a line end) if a better cut exists.
Return JSON:
{{"clips":[{{"id":"E01-C1","hook_score":0-10,"cliffhanger_score":0-10,"standalone_score":0-10,"overall":0-10,
  "hook_line":"what happens/is said in the first 3s, quoting the screenplay",
  "cliffhanger_text":"the unresolved moment it cuts on",
  "why":"one sentence on why it works (or why it is weak)",
  "weakness":"biggest risk, e.g. needs context / resolves too early / hook is slow",
  "adjust":null or {{"start":"mm:ss.s","end":"mm:ss.s"}}}}]}}"""
    data, raw = llm.complete_json(system, user)
    return data, raw


def rank_global(llm, show_context, items, top_n=15):
    """items: list of dicts (id, ep, window, scores, hook, cliffhanger, why). Model orders them and may join segments."""
    system = BRIEF + "\nReply with ONE JSON object only."
    lines = "\n".join(
        f'{it["id"]} | ep{it["ep"]} {it["window"]} ({it["dur"]:.0f}s) | overall {it["overall"]} | hook: {it["hook_line"][:140]} | ends: {it["cliffhanger"][:140]}'
        for it in items)
    user = f"""SHOW CONTEXT (episode loglines):
{show_context}

SHORTLISTED CLIPS (one per line):
{lines}

Task:
1. Choose and rank the {top_n} promos you would post FIRST for a brand-new audience. Prefer variety: different episodes, storylines and hook types, and put the strongest cold-open first. Say in one sentence why each ranks where it does.
2. OPTIONAL: you may propose up to 3 'joined' promos that stitch 2-3 listed clips in story order (e.g. a hook from one clip + a cliffhanger from another, or across episodes) ONLY if the join makes sense to a cold viewer and the total length is 20-60s. Reference clips by id in order.
Return JSON:
{{"ranking":[{{"rank":1,"ids":["E01-C1"],"reason":"..."}}, {{"rank":2,"ids":["E05-C2","E06-C1"],"reason":"joined: ..."}}]}}"""
    return llm.complete_json(system, user)
