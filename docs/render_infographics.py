"""Rebuild the README's accessible, dependency-free SVG engineering figures.

Run from any directory: python docs/render_infographics.py
The figures illustrate source structure and toy grids, not benchmark results.
"""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent / "assets"
NAVY = "#10243a"
MUTED = "#53677d"
BLUE = "#2563eb"
TEAL = "#087f75"
PURPLE = "#7251b5"
BORDER = "#d8e2ed"


class Figure:
    def __init__(self, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{height}" '
            f'viewBox="0 0 1440 {height}" role="img" aria-labelledby="title description">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="description">{escape(description)}</desc>',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" '
            'refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#53677d"/>'
            '</marker></defs>',
            '<style>text{font-family:Arial,Helvetica,sans-serif} '
            '.title{font-size:36px;font-weight:700;fill:#10243a} '
            '.subtitle{font-size:20px;fill:#53677d} '
            '.label{font-size:22px;font-weight:700;fill:#10243a} '
            '.body{font-size:19px;fill:#53677d} '
            '.small{font-size:17px;fill:#53677d} '
            '.kicker{font-size:16px;font-weight:700;letter-spacing:2px;fill:#2563eb}'
            '</style>',
        ]
        self.rect(0, 0, 1440, height, "#f4f7fb", radius=0, stroke="none")
        self.text(44, 44, "NEXT-CONCEPT MODELING FOR TPU", "kicker")
        self.text(44, 91, title, "title")

    def rect(self, x, y, w, h, fill="white", radius=14, stroke=BORDER):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def text(self, x, y, value, css="body", anchor="start", fill=None):
        override = f' style="fill:{fill}"' if fill else ""
        self.parts.append(f'<text x="{x}" y="{y}" class="{css}" '
                          f'text-anchor="{anchor}"{override}>{escape(value)}</text>')

    def lines(self, x, y, values, css="body", spacing=27, anchor="start"):
        for i, line in enumerate(values):
            self.text(x, y + i * spacing, line, css, anchor)

    def arrow(self, points, dashed=False):
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.parts.append(f'<polyline points="{points}" fill="none" '
                          f'stroke="{MUTED}" stroke-width="2.3" '
                          f'marker-end="url(#arrow)"{dash}/>')

    def box(self, x, y, w, title, subtitle, color=BLUE, fill="#eef4ff"):
        self.rect(x, y, w, 100, fill)
        self.rect(x, y, 5, 100, color, radius=2, stroke="none")
        self.text(x + w / 2, y + 39, title, "label", "middle")
        self.text(x + w / 2, y + 71, subtitle, "small", "middle")

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text("\n".join([*self.parts, "</svg>"]) + "\n", encoding="utf-8")


def architecture():
    f = Figure(780, "Two paths, one discrete hierarchy",
               "Training encodes text with a residual vector quantizer and decodes it. "
               "Generation predicts code positions together at each level, conditioned "
               "only on accumulated earlier-level vectors. EMA state is explicit.")
    f.text(44, 127, "Text codec + shared level-wise prior · explicit state · static masked arrays", "subtitle")
    f.rect(40, 157, 1360, 232)
    f.text(66, 195, "01 / TOKENIZER LEARNING", "kicker")
    f.box(68, 222, 198, "Input token IDs", "valid + fallback masks")
    f.box(302, 222, 208, "Text encoder", "residual 1D convolutions")
    f.box(546, 222, 240, "Multiscale RVQ", "codes + cumulative vectors", TEAL, "#eaf8f4")
    f.box(822, 222, 208, "Text decoder", "residual blocks + head")
    f.box(1066, 222, 304, "Token distributions", "masked reconstruction CE")
    for left, right in [(266, 302), (510, 546), (786, 822), (1030, 1066)]:
        f.arrow(f"{left},272 {right-6},272")
    f.text(68, 363, "RVQ carries E [K,V,D], C [K,V], S [K,V,D]; padding and byte slots do not enter code statistics.", "body")

    f.rect(40, 413, 1360, 287)
    f.text(66, 451, "02 / FREE-RUNNING GENERATION", "kicker")
    f.box(68, 480, 198, "Zero initial carry", "known length / geometry")
    f.box(302, 480, 240, "Shared Transformer", "resize + project prior carry")
    f.box(578, 480, 208, "Level-k codes", "all positions together", PURPLE, "#f3effc")
    f.box(822, 480, 208, "Accumulate", "embed + interpolate", TEAL, "#eaf8f4")
    f.box(1066, 480, 304, "Decode after level K", "add supplied byte features")
    for left, right in [(266, 302), (542, 578), (786, 822)]:
        f.arrow(f"{left},530 {right-6},530")
    f.arrow("1030,530 1060,530")
    f.arrow("926,582 926,627 422,627 422,584", dashed=True)
    f.text(674, 657, "Repeat coarse → fine; earlier-level predictions become the next context.", "small", "middle")
    f.text(44, 742, "Implemented in JAX/Flax. Rust data and orchestration tooling is planned. No trained weights are bundled.", "body")
    f.save("architecture.svg")


def level_generation():
    f = Figure(690, "Generate a complete level in one evaluation",
               "An illustrative hierarchy with lengths two, four, and eight requires "
               "three sequential level evaluations. All positions in a level are "
               "predicted in parallel. Code vectors accumulate between levels.")
    f.text(44, 127, "Illustrative schedule (2, 4, 8); colored blocks represent code positions, not semantic labels.", "subtitle")
    f.rect(40, 157, 936, 459)
    rows = [(211, 2, BLUE, "#eaf1ff", "Coarse", "k = 1  ·  n₁ = 2"),
            (348, 4, TEAL, "#eaf8f4", "Intermediate", "k = 2  ·  n₂ = 4"),
            (485, 8, PURPLE, "#f3effc", "Fine", "k = 3  ·  n₃ = 8")]
    for y, count, color, tint, name, meta in rows:
        f.text(70, y + 26, name, "label")
        f.text(70, y + 58, meta, "body")
        for i in range(count):
            x = 356 + i * 70
            f.rect(x, y, 54, 58, tint, radius=9, stroke=color)
            f.text(x + 27, y + 37, str(i + 1), "label", "middle", color)
        if count < 8:
            f.arrow(f"382,{y+65} 382,{y+109}")
            f.text(412, y + 96, "accumulate → resize → condition next level", "small")
    f.rect(1002, 157, 398, 459, "#10243a", stroke="#10243a")
    f.text(1032, 204, "K level evaluations", "label", fill="white")
    f.lines(1032, 249, ["Positions are predicted together.", "The Transformer shares weights", "across hierarchy levels."], spacing=29)
    # Explicit fills keep text readable against the dark callout.
    f.parts[-3:] = [part.replace('class="body"', 'class="body" style="fill:#cbd8e7"') for part in f.parts[-3:]]
    for y, title, desc in [(368, "Conditioning", "Cumulative earlier-level vectors"),
                           (456, "Code selection", "Greedy argmax in this snapshot"),
                           (544, "Training labels", "Added after current logits")]:
        f.text(1032, y, title, "label", fill="#93c5fd")
        f.text(1032, y + 30, desc, "small", fill="#d6e0eb")
    f.text(44, 658, "Static storage is padded to L at every level. Masking preserves correctness; padding still has a compute cost.", "body")
    f.save("level-generation.svg")


def spmd_mesh():
    f = Figure(770, "Partition data and weights; replicate codebook state",
               "An illustrative two by two dp and fsdp mesh. Data groups differ "
               "along dp; parameter shards differ along fsdp. Codebook embeddings, "
               "counts, and residual sums replicate after global aggregation.")
    f.text(44, 127, "NamedSharding contracts for batches, parameters, optimizer state, and functional EMA", "subtitle")
    f.rect(40, 157, 850, 522)
    f.text(325, 199, "fsdp = 0", "label", "middle")
    f.text(662, 199, "fsdp = 1", "label", "middle")
    for row in range(2):
        y = 227 + row * 216
        f.text(67, y + 90, f"dp = {row}", "label")
        for col in range(2):
            x = 168 + col * 337
            f.rect(x, y, 307, 181, "#fafcff")
            f.text(x + 18, y + 33, f"Device ({row}, {col})", "label")
            f.rect(x + 16, y + 50, 275, 32, "#eaf8f4", radius=5, stroke="none")
            f.text(x + 28, y + 73, f"Examples: group {'A' if row == 0 else 'B'}", "small", fill=TEAL)
            f.rect(x + 16, y + 92, 275, 32, "#eaf1ff", radius=5, stroke="none")
            f.text(x + 28, y + 115, f"Weights / moments: shard {col}", "small", fill=BLUE)
            f.text(x + 28, y + 155, "E, C, S: replicated", "small", fill=PURPLE)
    f.rect(916, 157, 484, 522)
    f.text(944, 202, "EMA synchronization", "label")
    f.lines(944, 241, ["Local assignments contribute counts", "and residual sums across the data axis."], spacing=27)
    f.rect(946, 295, 196, 53, "#eaf8f4")
    f.rect(1163, 295, 207, 53, "#eaf8f4")
    f.text(1044, 329, "counts c [K,V]", "body", "middle")
    f.text(1267, 329, "sums s [K,V,D]", "body", "middle")
    f.arrow("1159,356 1159,398")
    f.text(1159, 430, "Reduce over dp exactly once", "label", "middle")
    f.arrow("1159,447 1159,479")
    f.rect(946, 493, 424, 62, "#f3effc")
    f.text(1158, 532, "Identical EMA update → replicated E, C, S", "small", "middle", PURPLE)
    f.lines(944, 598, ["Global arrays: reductions are global.", "Manual maps: use a bound-axis psum."], spacing=29)
    f.text(44, 721, "Illustrative 2 × 2 mesh. Verified with four virtual CPU devices; TPU and multi-host performance are unmeasured.", "body")
    f.save("spmd-mesh.svg")


if __name__ == "__main__":
    architecture()
    level_generation()
    spmd_mesh()
    print("Wrote three SVG infographics to", OUT)
