"""Step 1 (pure code, no AI): cut every valid 20-60s window and score it with transparent rules.

Why code here: timestamps must be exact and durations legal. The model is never asked to
invent a time, it only chooses from windows that already start/end on real line boundaries.
"""
import re
from dataclasses import dataclass, field
from .parse import Episode, fmt, transcript

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WORDS = re.compile(
    r"\b(slap|slaps|slapped|strike|struck|hit|kill|killed|murder|poison|poisoned|arsenic|arrest|arrested|hang|hanged|"
    r"divorce|disown|disowned|betray|betrayed|lie|lied|liar|secret|truth|pregnant|baby|child|heir|dead|death|die|"
    r"blood|bleed|knife|sword|gun|scream|gasp|reveal|revealed|duke|duchess|king|queen|never|how dare|get out|leave|"
    r"mine|husband|wife|marry|married|wedding|kneel|beg|traitor|guilty|innocent|proof|evidence|ring|ledger|forgive)\b", re.I)
EMOTION = re.compile(r"[!?]|\b(fury|furious|rage|shock|stunned|horror|terrified|trembl|tears|weep|sob|slams?|shatter|collapse|"
                     r"snatch|grab|drag|threat|cold|ominous|menac|defian|tense)\w*", re.I)
CLIFF_SOUND = re.compile(r"cliffhanger|stinger|sting\b|swell|ominous|dramatic (sting|hit)|bass hit|silence|cut to black", re.I)
REACTION_OPEN = re.compile(r"^\W*(what|why|how|who|no|yes|oh|huh|please|wait|stop|but|and|so|then)\b[^.]{0,25}$", re.I)
PRONOUN_OPEN = re.compile(r"^\W*(he|she|they|him|her|it|that|this|those|these)\b", re.I)


@dataclass
class Candidate:
    cid: str
    ep: int
    start: float
    end: float
    score: float = 0.0
    parts: dict = field(default_factory=dict)
    hook_text: str = ""
    end_text: str = ""

    @property
    def dur(self): return self.end - self.start

    def label(self): return f"{fmt(self.start)}-{fmt(self.end)}"


def _lines(ep: Episode):
    return [e for e in ep.events if e.kind in ("dialogue", "action")]


def _first_text(ev_list, a, b):
    return " ".join(e.text for e in ev_list if e.start < b and e.end > a)


def score_window(ep: Episode, lines, s, e):
    inside = [x for x in lines if x.start >= s - 0.01 and x.end <= e + 0.01]
    dlg = [x for x in inside if x.kind == "dialogue"]
    hook_txt = _first_text(inside, s, s + 3.0)
    last = inside[-1] if inside else None
    end_txt = last.text if last else ""
    sounds = [x for x in ep.events if x.kind == "sound" and x.end > e - 4 and x.start < e + 0.5]

    hook = min(1.0, 0.35 * len(HOOK_WORDS.findall(hook_txt)) + 0.25 * len(EMOTION.findall(hook_txt)))
    first = inside[0] if inside else None
    if first and first.kind == "dialogue" and len(first.text.split()) >= 4 and first.text.strip().startswith('"'):
        hook += 0.15                                   # opens mid-speech with a real line, not a stage direction
    hook = min(1.0, hook)

    cliff = 0.0
    if e >= ep.duration - 1.5: cliff += 0.5            # episode ending is a built-in cliffhanger point
    if any(CLIFF_SOUND.search(x.text) for x in sounds): cliff += 0.35
    if end_txt.rstrip().endswith(("?", "...", "…", '?"', '..."')): cliff += 0.25
    if HOOK_WORDS.search(end_txt): cliff += 0.15
    if any(sc[3] and abs(sc[3] - e) < 1.0 for sc in ep.scenes): cliff += 0.1
    cliff = min(1.0, cliff)

    speakers = {x.speaker for x in dlg}
    standalone = 0.4 + 0.15 * min(len(speakers), 3)
    if first and first.kind == "dialogue" and (REACTION_OPEN.match(first.text.strip('" ')) or PRONOUN_OPEN.match(first.text.strip('" '))):
        standalone -= 0.3                              # opens on a context-dependent reply
    if len(dlg) < 3: standalone -= 0.2
    standalone = max(0.0, min(1.0, standalone))

    tension = min(1.0, len(EMOTION.findall(" ".join(x.text for x in inside))) / max(6.0, (e - s) / 3))
    dur = e - s
    length = 1.0 if 28 <= dur <= 50 else 0.7
    parts = dict(hook=round(hook, 2), cliffhanger=round(cliff, 2), standalone=round(standalone, 2),
                 tension=round(tension, 2), length=length)
    total = 0.32 * hook + 0.32 * cliff + 0.2 * standalone + 0.1 * tension + 0.06 * length
    return total, parts, hook_txt, end_txt


def make_candidates(ep: Episode, keep=10, max_overlap=0.35):
    lines = _lines(ep)
    starts = sorted({x.start for x in lines})
    ends = sorted({x.end for x in lines})
    pool = []
    for s in starts:
        for e in ends:
            if MIN_LEN <= e - s <= MAX_LEN:
                total, parts, h, t = score_window(ep, lines, s, e)
                pool.append(Candidate("", ep.number, s, e, total, parts, h, t))
    pool.sort(key=lambda c: -c.score)
    chosen = []
    for c in pool:                                      # non-max suppression: avoid 10 near-identical windows
        ok = True
        for k in chosen:
            ov = min(c.end, k.end) - max(c.start, k.start)
            if ov > max_overlap * min(c.dur, k.dur):
                ok = False; break
        if ok:
            chosen.append(c)
        if len(chosen) >= keep: break
    for i, c in enumerate(chosen, 1):
        c.cid = f"E{ep.number:02d}-C{i}"
    return chosen, len(pool)
