#!/usr/bin/env python3
"""Render a checkpoint's parameter geometry as scalable vector graphics.

Two panels, both drawn from real tensor shapes read off the checkpoint:

  Left   the whole model as a column, area proportional to parameter count, so
         the embedding table can be compared against the entire layer stack.
  Right  one transformer block with every weight matrix at its true aspect
         ratio and a shared px-per-element scale, so a matrix that holds four
         times the numbers occupies four times the area.

The same scale is used for every model, so the SVGs are comparable side by side.
Colours come from CSS custom properties with literal fallbacks, so the file
themes with its host page when inlined and still stands alone when opened
directly.
"""

ROLE_COLOR = {
    "attention": ("--anat-attn", "#4C72B0"),
    "ffn": ("--anat-ffn", "#DD8452"),
    "embedding": ("--anat-emb", "#8172B3"),
    "norm": ("--anat-norm", "#937860"),
    "output_head": ("--anat-out", "#55A868"),
    "other": ("--anat-other", "#8C8C8C"),
}

PX_PER_ELEMENT = 0.055   # shared across models; area is then proportional to params
GAP = 14
PAD = 18
LABEL_H = 26
COL_W = 132              # width of the whole-model column panel
PANEL_GAP = 54


def _fill(role):
    var, lit = ROLE_COLOR.get(role, ROLE_COLOR["other"])
    return f"var({var}, {lit})"


def _fmt(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(n)


def _esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _block_tensors(stats):
    """The 2D weight matrices of layer 0, in forward-pass order."""
    order = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "o_proj": 3, "c_attn": 0,
             "gate_proj": 4, "up_proj": 5, "c_fc": 4, "down_proj": 6}

    def leaf(n):
        return n.split(".")[-2] if n.endswith(".weight") else n.split(".")[-1]

    def key(t):
        l = leaf(t["name"])
        if l == "c_proj":
            return 3 if t["role"] == "attention" else 6
        return order.get(l, 9)

    def label(t):
        l = leaf(t["name"])
        if l == "c_proj":
            return "O" if t["role"] == "attention" else "down"
        return {"c_attn": "QKV", "c_fc": "up", "q_proj": "Q", "k_proj": "K",
                "v_proj": "V", "o_proj": "O", "gate_proj": "gate",
                "up_proj": "up", "down_proj": "down"}.get(l, l)

    ts = [t for t in stats if t["layer"] == 0 and len(t["shape"]) == 2]
    ts.sort(key=key)
    return [(label(t), t) for t in ts]


def block_canvas_width(stats):
    """Width the block panel needs, so callers can align several models."""
    ts = _block_tensors(stats)
    if not ts:
        return 0
    w = sum(t["shape"][1] * PX_PER_ELEMENT for _, t in ts) + GAP * (len(ts) - 1)
    return w


def render(stats, meta, canvas_w=None, px=None):
    """stats: the collect() list. meta: dict with name, variant, layers, total.

    px fixes the px-per-element scale. Left unset with no canvas_w, the scale
    adapts so the block panel lands near 640px — a 0.8M-parameter char model and
    a 494M one are both legible standalone. Pin px (or canvas_w) when rendering
    several models that must stay comparable."""
    tensors = _block_tensors(stats)
    if px is None:
        if canvas_w is not None:
            px = PX_PER_ELEMENT
        else:
            raw = sum(t["shape"][1] for _, t in tensors) or 1
            px = max(0.02, min(1.6, (640 - GAP * max(0, len(tensors) - 1)) / raw))
    globals_ = sorted(
        [t for t in stats if t["layer"] is None and len(t["shape"]) == 2],
        key=lambda t: -t["count"])

    n_layers = meta["layers"]
    total = meta["total"]
    block_total = sum(t["count"] for t in stats if t["layer"] == 0)
    stack_total = block_total * n_layers
    emb_total = sum(t["count"] for t in globals_)

    # ---- panel A: whole model as a column, height proportional to params ----
    col_h = 300.0
    emb_h = col_h * emb_total / total
    stack_h = col_h * stack_total / total
    band_h = stack_h / n_layers

    a = []
    y = LABEL_H
    if emb_total:
        a.append(f'<rect x="0" y="{y:.1f}" width="{COL_W}" height="{emb_h:.1f}" '
                 f'fill="{_fill("embedding")}" rx="1"/>')
        a.append(f'<text class="in" x="{COL_W / 2:.0f}" y="{y + emb_h / 2 + 3.5:.1f}" '
                 f'text-anchor="middle">embeddings</text>')
        y += emb_h + 3

    stack_y = y
    for i in range(n_layers):
        by = y + i * band_h
        x = 0.0
        for role in ("attention", "ffn", "norm"):
            c = sum(t["count"] for t in stats if t["layer"] == 0 and t["role"] == role)
            if not c:
                continue
            w = COL_W * c / block_total
            a.append(f'<rect x="{x:.2f}" y="{by:.2f}" width="{w:.2f}" '
                     f'height="{max(band_h - 1.1, 0.6):.2f}" fill="{_fill(role)}"/>')
            x += w
    a.append(f'<text class="side" x="{COL_W + 9}" y="{stack_y + stack_h / 2:.1f}" '
             f'dominant-baseline="middle">{n_layers} identical blocks</text>')
    a.append(f'<text class="cap" x="0" y="14">whole model &#183; {_fmt(total)}</text>')
    col_bottom = stack_y + stack_h

    # ---- panel B: one block, every matrix at true aspect ratio ----
    b = []
    bw = [t["shape"][1] * px for _, t in tensors]
    bh = [t["shape"][0] * px for _, t in tensors]
    max_h = max(bh) if bh else 0
    x = 0.0
    base = LABEL_H + max_h
    for (lab, t), w, h in zip(tensors, bw, bh):
        yy = base - h
        rows, cols = t["shape"]
        # A matrix thinner than the type sits its label above the rect instead
        # of inside it — GQA makes K and V only a few pixels tall.
        fits = w >= len(lab) * 6.6 + 6
        if h >= 15 and fits:
            tag = (f'<text class="in" x="{x + w / 2:.1f}" y="{yy + h / 2 + 3.5:.1f}" '
                   f'text-anchor="middle">{lab}</text>')
        elif fits:
            tag = (f'<text class="out" x="{x + w / 2:.1f}" y="{yy - 5:.1f}" '
                   f'text-anchor="middle">{lab}</text>')
        else:   # too narrow for horizontal type — set it vertically above
            tag = (f'<text class="out" x="{x + w / 2:.1f}" y="{yy - 6:.1f}" '
                   f'text-anchor="start" transform="rotate(-90 {x + w / 2:.1f} '
                   f'{yy - 6:.1f})">{lab}</text>')
        b.append(
            f'<g class="mx" tabindex="0">'
            f'<title>{_esc(t["name"])} — {rows} x {cols} = {t["count"]:,} parameters, '
            f'std {t["std"]:.4f}, max |w| {t["absmax"]:.3f}</title>'
            f'<rect x="{x:.1f}" y="{yy:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{_fill(t["role"])}" rx="1"/>{tag}'
            f'</g>')
        if w >= 40:      # otherwise adjacent captions collide; the tooltip has it
            b.append(f'<text class="dim" x="{x + w / 2:.1f}" y="{base + 13:.1f}" '
                     f'text-anchor="middle">{rows}&#215;{cols}</text>')
        x += w + GAP
    block_w = x - GAP if tensors else 0
    b.append(f'<text class="cap" x="0" y="14">one block &#183; {_fmt(block_total)} '
             f'&#215; {n_layers}</text>')

    if canvas_w is None:
        canvas_w = block_w
    total_w = COL_W + 118 + PANEL_GAP + canvas_w + PAD * 2
    total_h = PAD * 2 + LABEL_H + max(col_bottom, base + 22)

    off = COL_W + 118 + PANEL_GAP
    body = (f'<g transform="translate({PAD},{PAD})">' + "".join(a) + "</g>"
            f'<g transform="translate({PAD + off},{PAD})">' + "".join(b) + "</g>")

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.0f} {total_h:.0f}" \
role="img" aria-label="Parameter anatomy of {_esc(meta['name'])} {_esc(meta['variant'])}">
<style>
  text {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; }}
  .cap {{ font-size: 11px; fill: var(--anat-ink, #5C6675); letter-spacing: .06em;
         text-transform: uppercase; }}
  .side {{ font-size: 10.5px; fill: var(--anat-ink, #5C6675); }}
  .dim {{ font-size: 9.5px; fill: var(--anat-faint, #8C95A3); }}
  .in {{ font-size: 10px; fill: #FFF; font-weight: 500; }}
  .out {{ font-size: 10px; fill: var(--anat-ink, #5C6675); font-weight: 500; }}
  .mx rect {{ transition: opacity .12s ease; }}
  .mx:hover rect, .mx:focus rect {{ opacity: .74; }}
  .mx:focus {{ outline: none; }}
  .mx:focus rect {{ stroke: var(--anat-ink, #5C6675); stroke-width: 1.5; }}
  @media (prefers-reduced-motion: reduce) {{ .mx rect {{ transition: none; }} }}
</style>
{body}
</svg>'''
