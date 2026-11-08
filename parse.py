"""Parse Kuku-style timestamped screenplays into a flat, time-ordered event list."""
import re
from dataclasses import dataclass, field
from pathlib import Path

TS = r"(\d+):(\d+(?:\.\d+)?)"
BLOCK = re.compile(
    r"^\*\*(?P<label>[^*]+?):\*\*\s*\n\s*Start:\s*\(" + TS + r"\)\s*\|\s*End:\s*\(" + TS + r"\)\s*\n(?P<body>.*?)(?=\n\s*\n|\Z)",
    re.M | re.S | re.I,
)
SCENE = re.compile(r"^SCENE\s+(\d+)\s*\n\s*scene heading:\s*(.+?)\s*\n\s*scene start:\s*" + TS + r"\s*\n\s*scene end:\s*" + TS,
                   re.M | re.I)


def _sec(m, s):
    return int(m) * 60 + float(s)


@dataclass
class Event:
    kind: str          # dialogue | action | camera | sound
    speaker: str       # "" for non-dialogue
    start: float
    end: float
    text: str
    scene: int = 0


@dataclass
class Episode:
    number: int
    logline: str
    continuity: str
    duration: float
    events: list = field(default_factory=list)
    scenes: list = field(default_factory=list)   # (num, heading, start, end)
    path: str = ""


def fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def parse_episode(path) -> Episode:
    t = Path(path).read_text(encoding="utf-8", errors="replace")
    num = int(re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", t, re.I).group(1))
    logline = (re.search(r"EPISODE LOGLINE:\s*(.+)", t, re.I) or [None, ""])[1].strip()
    cont = (re.search(r"CONTINUITY NOTES:\s*(.+)", t, re.I) or [None, ""])[1].strip()
    scenes = [(int(m.group(1)), m.group(2), _sec(m.group(3), m.group(4)), _sec(m.group(5), m.group(6)))
              for m in SCENE.finditer(t)]
    events = []
    for m in BLOCK.finditer(t):
        label = m.group("label").strip()
        s, e = _sec(m.group(2), m.group(3)), _sec(m.group(4), m.group(5))
        body = " ".join(m.group("body").split())
        up = label.upper()
        if up == "CAMERA": kind, spk = "camera", ""
        elif up == "ACTION": kind, spk = "action", ""
        elif up.startswith("SOUND"): kind, spk = "sound", ""
        else:
            kind, spk = "dialogue", re.sub(r"\s*\(.*\)\s*$", "", label).strip()
        sc = next((n for n, _, a, b in scenes if a - 0.01 <= s < b + 0.01), 0)
        events.append(Event(kind, spk, s, e, body, sc))
    events.sort(key=lambda x: (x.start, x.end))
    dur = max([e.end for e in events] + [sc[3] for sc in scenes] + [0])
    return Episode(num, logline, cont, dur, events, scenes, str(path))


def load_pack(folder) -> list:
    files = sorted(Path(folder).glob("episode-*.txt"))
    return sorted((parse_episode(f) for f in files), key=lambda e: e.number)


def transcript(ep: Episode, start=0.0, end=1e9, with_camera=False) -> str:
    """Compact text rendering used both for LLM prompts and human review."""
    out = []
    for ev in ep.events:
        if ev.end <= start or ev.start >= end:
            continue
        if ev.kind == "camera" and not with_camera:
            continue
        who = ev.speaker.title() if ev.kind == "dialogue" else ev.kind.upper()
        out.append(f"[{fmt(ev.start)}-{fmt(ev.end)}] {who}: {ev.text}")
    return "\n".join(out)
