from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path("C:/Users/Windows/.codex/visualizations/2026/07/21/019f8283-e4e6-7793-93e8-d6d032dcdc44/exquisitor-temporal-search-download.png")
W, H = 1800, 1000
BG = "white"
INK = "#111827"
MUTED = "#4b5563"
LINE = "#6b7280"
BOX = "#f9fafb"
LLM = "#dbeafe"
QUERY = "#dcfce7"
VECTOR = "#fef3c7"
CHAIN = "#fee2e2"
RESULT = "#ffedd5"
DB = "#fef9c3"
FRAME = "#f3f4f6"
BLUE = "#2563eb"
RED = "#dc2626"


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


F_TITLE = font(22, True)
F = font(18)
F_B = font(18, True)
F_S = font(14)
F_XS = font(12)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def text_center(xy, text, f=F, fill=INK):
    x1, y1, x2, y2 = xy
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2), text, font=f, fill=fill)


def box(x, y, w, h, title, lines=(), fill=BOX):
    d.rectangle((x, y, x + w, y + h), fill=fill, outline=LINE, width=2)
    d.text((x + 16, y + 14), title, font=F_B, fill=INK)
    yy = y + 44
    for line in lines:
        d.text((x + 16, yy), line, font=F_S, fill=MUTED)
        yy += 20


def cylinder(x, y, w, h, title, lines=()):
    d.ellipse((x, y, x + w, y + 36), fill=DB, outline="#ca8a04", width=2)
    d.rectangle((x, y + 18, x + w, y + h - 18), fill=DB, outline="#ca8a04", width=2)
    d.ellipse((x, y + h - 36, x + w, y + h), fill=DB, outline="#ca8a04", width=2)
    d.arc((x, y + h - 36, x + w, y + h), 0, 180, fill="#ca8a04", width=2)
    text_center((x, y + 22, x + w, y + 54), title, F_B)
    yy = y + 60
    for line in lines:
        text_center((x, yy, x + w, yy + 18), line, F_S, MUTED)
        yy += 19


def arrow(x1, y1, x2, y2, fill=LINE, width=3):
    d.line((x1, y1, x2, y2), fill=fill, width=width)
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    size = 12
    pts = [
        (x2, y2),
        (x2 - size * math.cos(ang - 0.45), y2 - size * math.sin(ang - 0.45)),
        (x2 - size * math.cos(ang + 0.45), y2 - size * math.sin(ang + 0.45)),
    ]
    d.polygon(pts, fill=fill)


def poly_arrow(points, fill=LINE, width=3):
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        d.line((x1, y1, x2, y2), fill=fill, width=width)
    arrow(points[-2][0], points[-2][1], points[-1][0], points[-1][1], fill=fill, width=width)


d.line((20, 42, W - 20, 42), fill=INK, width=4)
for x in (430, 850, 1320):
    d.line((x, 42, x, H - 50), fill="#9ca3af", width=2)
    for yy in range(42, H - 50, 20):
        d.line((x, yy, x, yy + 10), fill=BG, width=2)
d.text((30, 58), "ONLINE TEMPORAL SEARCH / SEQUENCE-CHAIN METHOD", font=F_S, fill=MUTED)

box(50, 140, 290, 112, "Temporal query", ['"scene A happens, then', 'scene B appears after it"'])
arrow(340, 196, 470, 196)
box(470, 118, 300, 156, "LLM decomposition", ["Create ordered anchors q1..qN", "q1 is the first search anchor", "later queries must occur after q1"], LLM)
arrow(770, 196, 900, 196)

box(900, 85, 310, 58, "q1: first visible event", fill=QUERY)
box(900, 165, 310, 58, "q2: later visible event", fill=QUERY)
box(900, 245, 310, 58, "q3: final visible event", fill=QUERY)
d.text((910, 325), "Ordered anchors: q1 -> q2 -> q3", font=F_S, fill=MUTED)

arrow(1210, 196, 1370, 196)
box(1370, 120, 300, 152, "Text encoder", ["Each sub-query becomes", "an embedding vector"], VECTOR)
for i, val in enumerate(["0.12", "-0.4", "0.87", "..."]):
    x = 1410 + i * 58
    d.rounded_rectangle((x, 205, x + 48, 232), radius=5, fill="#fde68a", outline=LINE, width=1)
    text_center((x, 205, x + 48, 232), val, F_XS)
d.text((1410, 246), "vector(q1), vector(q2), vector(q3)", font=F_S, fill=MUTED)

box(60, 455, 250, 96, "Indexed keyframes", ["video_id + timestamp", "keyframe embedding vector"])
arrow(310, 503, 415, 503)
cylinder(415, 432, 330, 138, "Keyframe vector index", ["search top-r for q1", "search candidate pool for q2..qN"])
poly_arrow([(1520, 272), (1520, 360), (590, 360), (590, 432)])
arrow(745, 503, 900, 503)
box(900, 440, 320, 132, "Initial anchor search", ["Run q1 and retrieve top-r segments", "Default paper value: r = 1000", "q1 matches start candidate chains"], CHAIN)
arrow(1060, 572, 1060, 650)
box(900, 650, 320, 132, "Build candidate pool Sc", ["Collect segments strictly after", "each matched q1 segment", "timestamp(next) > timestamp(prev)"], CHAIN)
arrow(1220, 716, 1370, 716)
box(1370, 650, 330, 132, "Batched later searches", ["Run q2..qN only over Sc", "Choose highest-ranked later segment", "walk forward in time per video"], CHAIN)

d.text((70, 660), "Chain assembly per video", font=F_S, fill=MUTED)
frame_y = 700
xs = [70, 205, 340, 475, 610]
labels = ["q1", "", "q2", "", "q3"]
fills = ["#bfdbfe", FRAME, "#fecaca", FRAME, "#fecaca"]
for x, lab, fill in zip(xs, labels, fills):
    d.rounded_rectangle((x, frame_y, x + 90, frame_y + 58), radius=4, fill=fill, outline=LINE, width=2)
    if lab:
        text_center((x, frame_y, x + 90, frame_y + 58), lab, F_B)
arrow(160, frame_y + 29, 340, frame_y + 29, RED, 4)
arrow(430, frame_y + 29, 610, frame_y + 29, RED, 4)
d.line((70, frame_y + 88, 700, frame_y + 88), fill="#9ca3af", width=2)
d.text((70, frame_y + 102), "time increases left to right", font=F_XS, fill=MUTED)

arrow(1535, 782, 1535, 855)
box(1370, 855, 330, 86, "Rank sequence chains", ["Longest chain first", "RRF as tie-break"], RESULT)
arrow(1370, 898, 1180, 898)
box(855, 855, 325, 86, "Select best chain per video", ["Remove redundant overlaps", "keep top distinct videos"], RESULT)
arrow(855, 898, 660, 898)
box(365, 855, 295, 86, "Final output", ["Ranked keyframes / chains", "Evaluate Recall@1,5,10"], RESULT)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT)
