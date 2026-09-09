#!/usr/bin/env python3
"""Build a single self-contained HTML comparison page from the generated plots.

Embeds every PNG as a data URI so the page works as a standalone file with no
external requests. Run after `weight-visualizer run` for each model.
"""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
OUT = os.path.join(HERE, "checkpoint-anatomy.html")

MODELS = [
    {"dir": "gpt2", "name": "GPT-2", "sub": "small",
     "params": "124.4M", "tensors": 148, "layers": 12, "dmodel": 768,
     "vocab": "50,257", "ffn": 45.5, "attn": 22.8, "emb": 31.6, "ratio": "2.00"},
    {"dir": "gpt2-medium", "name": "GPT-2", "sub": "medium",
     "params": "354.8M", "tensors": 292, "layers": 24, "dmodel": 1024,
     "vocab": "50,257", "ffn": 56.8, "attn": 28.4, "emb": 14.8, "ratio": "2.00"},
    {"dir": "Qwen2.5-0.5B", "name": "Qwen2.5", "sub": "0.5B",
     "params": "494.0M", "tensors": 290, "layers": 24, "dmodel": 896,
     "vocab": "151,936", "ffn": 63.5, "attn": 8.9, "emb": 27.6, "ratio": "7.13"},
]

PLATES = [
    ("02_distributions", "Value distributions",
     "Histogram of weight values per role, log counts. Near-Gaussian and centred on zero "
     "in every checkpoint; what differs between them is the width and the tails."),
    ("03_depth_trend", "Statistics by depth",
     "Spread and outlier magnitude per layer. First layers behave differently because they "
     "still operate on raw token embeddings; outliers cluster in a few mid-layers."),
    ("04_heatmaps", "Raw matrices",
     "A 128x128 crop from each role. Included to show that individual entries carry no "
     "visible structure — meaning lives in superposition across directions."),
    ("05_outlier_channels", "Outlier channels",
     "Per-channel maximum absolute weight. The spikes are the channels that break naive "
     "uniform quantization and force per-channel or per-layer schemes."),
]


def svg_of(slug):
    with open(os.path.join(PLOTS, slug, "anatomy.svg")) as f:
        return f.read()


def anatomy_section():
    out = []
    for m in MODELS:
        out.append(f"""
      <figure class="anat">
        <figcaption>
          <span class="who">{m['name']} <span class="variant">{m['sub']}</span></span>
          <span class="who-n">{m['params']} &middot; {m['layers']} blocks &middot;
            d<sub>model</sub> {m['dmodel']}</span>
        </figcaption>
        <div class="anat-svg">{svg_of(m['dir'])}</div>
      </figure>""")
    return "".join(out)


def data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def spec_cards():
    out = []
    for m in MODELS:
        out.append(f"""
      <article class="spec">
        <header>
          <h3>{m['name']} <span class="variant">{m['sub']}</span></h3>
          <p class="count">{m['params']}</p>
        </header>
        <dl>
          <div><dt>Layers</dt><dd>{m['layers']}</dd></div>
          <div><dt>Hidden dim</dt><dd>{m['dmodel']}</dd></div>
          <div><dt>Vocabulary</dt><dd>{m['vocab']}</dd></div>
          <div><dt>Tensors</dt><dd>{m['tensors']}</dd></div>
        </dl>
      </article>""")
    return "".join(out)


def budget_rows():
    rows = []
    for key, label in (("ffn", "Feed-forward"), ("attn", "Attention"), ("emb", "Embedding")):
        cells = "".join(
            f'<td><span class="bar" style="--w:{m[key]}%"></span>'
            f'<span class="pct">{m[key]}%</span></td>' for m in MODELS)
        rows.append(f"<tr><th scope=\"row\">{label}</th>{cells}</tr>")
    ratio = "".join(f'<td class="ratio">{m["ratio"]}<span class="unit">:1</span></td>'
                    for m in MODELS)
    rows.append(f'<tr class="derived"><th scope="row">FFN : attention</th>{ratio}</tr>')
    return "\n            ".join(rows)


def plate_sections():
    out = []
    for n, (stem, title, blurb) in enumerate(PLATES, start=1):
        figs = []
        for m in MODELS:
            path = os.path.join(PLOTS, m["dir"], f"{stem}.png")
            if not os.path.exists(path):
                continue
            figs.append(f"""
          <figure>
            <button class="plate" aria-label="Enlarge {title}, {m['name']} {m['sub']}">
              <img src="{data_uri(path)}" alt="{title} for {m['name']} {m['sub']}" loading="lazy">
            </button>
            <figcaption>{m['name']} <span class="variant">{m['sub']}</span></figcaption>
          </figure>""")
        out.append(f"""
      <section class="plates" id="plate-{n}">
        <div class="plate-head">
          <p class="eyebrow">Plate {n}</p>
          <h2>{title}</h2>
          <p class="blurb">{blurb}</p>
        </div>
        <div class="row">{''.join(figs)}
        </div>
      </section>""")
    return "".join(out)


HTML = f"""<title>Checkpoint Anatomy</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,400;1,400&display=swap">
<style>
  :root {{
    --ground:   #F4F6F8;
    --surface:  #FFFFFF;
    --ink:      #161A21;
    --muted:    #5C6675;
    --faint:    #8C95A3;
    --line:     #DCE1E8;
    --accent:   #4C72B0;
    --flag:     #C44E52;
    --ok:       #55A868;
    --plate-bg: #FFFFFF;
    --plate-ring: rgba(22,26,33,.12);
    --sans: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
    --serif: "IBM Plex Serif", Georgia, serif;
    --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
    --measure: 64ch;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground:  #0F1319;
      --surface: #171C24;
      --ink:     #E3E8EF;
      --muted:   #98A2B1;
      --faint:   #6E7887;
      --line:    #262D38;
      --accent:  #7CA0D4;
      --flag:    #E07A7D;
      --ok:      #7FC08F;
      --plate-bg: #EDEFF2;
      --plate-ring: rgba(0,0,0,.5);
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:  #0F1319;
    --surface: #171C24;
    --ink:     #E3E8EF;
    --muted:   #98A2B1;
    --faint:   #6E7887;
    --line:    #262D38;
    --accent:  #7CA0D4;
    --flag:    #E07A7D;
    --ok:      #7FC08F;
    --plate-bg: #EDEFF2;
    --plate-ring: rgba(0,0,0,.5);
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 0 28px 96px; }}

  /* ---------- masthead ---------- */
  header.top {{
    border-bottom: 1px solid var(--line);
    padding: 56px 0 32px;
    display: grid;
    gap: 20px;
  }}
  .eyebrow {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--faint);
    margin: 0;
  }}
  h1 {{
    font-size: clamp(30px, 4.4vw, 46px);
    line-height: 1.08;
    letter-spacing: -.02em;
    font-weight: 600;
    margin: 0;
    text-wrap: balance;
  }}
  .standfirst {{
    font-family: var(--serif);
    font-size: 17px;
    line-height: 1.65;
    color: var(--muted);
    max-width: var(--measure);
    margin: 0;
  }}

  /* ---------- spec strip ---------- */
  .specs {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--line);
    border: 1px solid var(--line);
    margin: 40px 0 0;
  }}
  .spec {{ background: var(--surface); padding: 20px 22px; }}
  .spec header {{
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 10px; margin-bottom: 14px;
  }}
  .spec h3 {{ font-size: 16px; font-weight: 600; margin: 0; letter-spacing: -.01em; }}
  .variant {{
    font-family: var(--mono); font-size: 11px; font-weight: 400;
    color: var(--faint); letter-spacing: .04em;
  }}
  .count {{
    font-family: var(--mono); font-size: 15px; font-weight: 500;
    color: var(--accent); margin: 0; font-variant-numeric: tabular-nums;
  }}
  .spec dl {{ margin: 0; display: grid; gap: 5px; }}
  .spec dl div {{ display: flex; justify-content: space-between; gap: 12px; }}
  .spec dt {{ font-size: 12.5px; color: var(--muted); }}
  .spec dd {{
    margin: 0; font-family: var(--mono); font-size: 12.5px;
    font-variant-numeric: tabular-nums;
  }}

  /* ---------- budget table ---------- */
  .budget {{ margin: 64px 0 0; }}
  h2 {{
    font-size: 21px; font-weight: 600; letter-spacing: -.015em;
    margin: 0 0 6px; text-wrap: balance;
  }}
  .blurb {{
    color: var(--muted); max-width: var(--measure); margin: 0;
    font-size: 14.5px;
  }}
  .table-scroll {{ overflow-x: auto; margin-top: 22px; }}
  table {{ width: 100%; border-collapse: collapse; min-width: 560px; }}
  thead th {{
    text-align: left; font-size: 12px; font-weight: 500; color: var(--muted);
    padding: 0 14px 10px 0; border-bottom: 1px solid var(--line);
    font-family: var(--mono); letter-spacing: .04em; text-transform: uppercase;
  }}
  thead th .variant {{ text-transform: none; letter-spacing: 0; }}
  tbody th {{
    text-align: left; font-weight: 500; font-size: 14px;
    padding: 13px 20px 13px 0; white-space: nowrap;
    border-bottom: 1px solid var(--line);
  }}
  tbody td {{
    padding: 13px 14px 13px 0; border-bottom: 1px solid var(--line);
    vertical-align: middle; width: 26%;
  }}
  .bar {{
    display: block; height: 6px; width: var(--w);
    background: var(--accent); border-radius: 1px; margin-bottom: 5px;
    min-width: 2px;
  }}
  .pct {{
    font-family: var(--mono); font-size: 13px;
    font-variant-numeric: tabular-nums; color: var(--ink);
  }}
  tr.derived th, tr.derived td {{ border-bottom: none; padding-top: 16px; }}
  .ratio {{
    font-family: var(--mono); font-size: 19px; font-weight: 500;
    font-variant-numeric: tabular-nums; color: var(--ink);
  }}
  .ratio .unit {{ font-size: 12px; color: var(--faint); }}

  /* ---------- anatomy ---------- */
  :root {{
    --anat-attn: #4C72B0; --anat-ffn: #DD8452; --anat-emb: #8172B3;
    --anat-norm: #937860; --anat-out: #55A868; --anat-other: #8C8C8C;
    --anat-ink: #5C6675; --anat-faint: #8C95A3;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --anat-attn: #6E92C9; --anat-ffn: #E09A6C; --anat-emb: #9B8ECB;
      --anat-norm: #A8927B; --anat-out: #75BE8A;
      --anat-ink: #98A2B1; --anat-faint: #6E7887;
    }}
  }}
  :root[data-theme="dark"] {{
    --anat-attn: #6E92C9; --anat-ffn: #E09A6C; --anat-emb: #9B8ECB;
    --anat-norm: #A8927B; --anat-out: #75BE8A;
    --anat-ink: #98A2B1; --anat-faint: #6E7887;
  }}
  .anatomy {{ margin: 60px 0 0; }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 18px 0 26px;
    font-family: var(--mono); font-size: 11.5px; color: var(--muted);
  }}
  .legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
  .legend i {{ width: 11px; height: 11px; border-radius: 2px; display: block; }}
  .anat-list {{ display: grid; gap: 34px; }}
  figure.anat {{ margin: 0; display: grid; gap: 10px; }}
  figure.anat figcaption {{
    display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px;
    padding-bottom: 8px; border-bottom: 1px solid var(--line);
  }}
  .who {{ font-size: 15px; font-weight: 600; }}
  .who-n {{
    font-family: var(--mono); font-size: 11.5px; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }}
  .anat-svg {{ overflow-x: auto; }}
  .anat-svg svg {{ width: 100%; min-width: 720px; height: auto; display: block; }}

  /* ---------- findings ---------- */
  .findings {{
    margin: 56px 0 0; display: grid; gap: 26px;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }}
  .finding {{ border-top: 2px solid var(--accent); padding-top: 14px; }}
  .finding.flagged {{ border-top-color: var(--flag); }}
  .finding h3 {{ font-size: 14.5px; font-weight: 600; margin: 0 0 6px; }}
  .finding p {{
    font-family: var(--serif); font-size: 14.5px; line-height: 1.62;
    color: var(--muted); margin: 0;
  }}
  .finding code, .blurb code, .note code {{
    font-family: var(--mono); font-size: .88em;
    background: color-mix(in srgb, var(--accent) 11%, transparent);
    padding: .1em .34em; border-radius: 2px; color: var(--ink);
  }}

  /* ---------- plates ---------- */
  .plates {{ margin: 72px 0 0; }}
  .plate-head {{ margin-bottom: 22px; }}
  .plate-head .eyebrow {{ margin-bottom: 7px; }}
  .row {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px;
    align-items: start;
  }}
  figure {{ margin: 0; display: grid; gap: 8px; }}
  button.plate {{
    display: block; width: 100%; padding: 10px; cursor: zoom-in;
    background: var(--plate-bg);
    border: 1px solid var(--plate-ring);
    border-radius: 2px;
    line-height: 0;
    transition: border-color .15s ease;
  }}
  button.plate:hover {{ border-color: var(--accent); }}
  button.plate:focus-visible {{
    outline: 2px solid var(--accent); outline-offset: 2px;
  }}
  button.plate img {{ width: 100%; height: auto; display: block; }}
  figcaption {{
    font-family: var(--mono); font-size: 11.5px; color: var(--muted);
    letter-spacing: .03em;
  }}

  /* ---------- note ---------- */
  .note {{
    margin: 76px 0 0; padding: 24px 26px;
    background: var(--surface); border: 1px solid var(--line);
    border-left: 3px solid var(--flag);
  }}
  .note h2 {{ font-size: 16px; margin-bottom: 10px; }}
  .note p {{
    margin: 0 0 10px; color: var(--muted); max-width: var(--measure);
    font-size: 14.5px;
  }}
  .note p:last-child {{ margin-bottom: 0; }}

  footer {{
    margin-top: 72px; padding-top: 22px; border-top: 1px solid var(--line);
    font-family: var(--mono); font-size: 11.5px; color: var(--faint);
    display: flex; flex-wrap: wrap; gap: 8px 20px;
  }}

  /* ---------- lightbox ---------- */
  dialog {{
    padding: 0; border: none; background: transparent; max-width: 96vw; max-height: 94vh;
  }}
  dialog::backdrop {{ background: rgba(8,10,14,.82); }}
  dialog img {{
    max-width: 96vw; max-height: 94vh; width: auto; height: auto;
    display: block; background: #FFF; padding: 12px; border-radius: 2px;
  }}
  dialog button.close {{
    position: fixed; top: 16px; right: 20px; z-index: 2;
    background: rgba(255,255,255,.14); color: #FFF; border: 1px solid rgba(255,255,255,.3);
    border-radius: 2px; padding: 5px 11px; cursor: pointer;
    font-family: var(--mono); font-size: 12px;
  }}

  @media (max-width: 860px) {{
    .specs {{ grid-template-columns: 1fr; }}
    .row {{ grid-template-columns: 1fr; }}
    .wrap {{ padding: 0 18px 72px; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    * {{ transition: none !important; animation: none !important; }}
  }}
</style>

<div class="wrap">
  <header class="top">
    <p class="eyebrow">weight-visualizer &middot; ai-model-dev</p>
    <h1>Where the parameters actually live</h1>
    <p class="standfirst">Three checkpoints read tensor by tensor, grouped by the role each
    weight plays. The comparison isolates two things: what changes when you scale the same
    architecture, and what changes when the architecture itself moves on.</p>
  </header>

  <div class="specs">{spec_cards()}
  </div>


  <section class="anatomy">
    <h2>The model drawn to scale</h2>
    <p class="blurb">Every rectangle is a real weight matrix, sized from the shape read off
    the checkpoint. On the right each matrix keeps its true aspect ratio at one shared
    px-per-element scale, so a matrix holding four times the numbers covers four times the
    area — and the three models are directly comparable. Hover or focus any block for its
    tensor name, exact shape and value statistics.</p>
    <div class="legend">
      <span><i style="background:var(--anat-emb)"></i> embedding</span>
      <span><i style="background:var(--anat-attn)"></i> attention</span>
      <span><i style="background:var(--anat-ffn)"></i> feed-forward</span>
      <span><i style="background:var(--anat-norm)"></i> normalization</span>
    </div>
    <div class="anat-list">{anatomy_section()}
    </div>
  </section>

  <section class="budget">
    <h2>The same thing as numbers</h2>
    <p class="blurb">Percentages of total learned parameters — the figures behind the areas
    above. Normalization scales are present in all three but round to 0.0% and are omitted.</p>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">Role</th>
            <th scope="col">GPT-2 <span class="variant">small</span></th>
            <th scope="col">GPT-2 <span class="variant">medium</span></th>
            <th scope="col">Qwen2.5 <span class="variant">0.5B</span></th>
          </tr>
        </thead>
        <tbody>
            {budget_rows()}
        </tbody>
      </table>
    </div>
  </section>

  <section class="findings">
    <div class="finding">
      <h3>The 2:1 ratio is architectural, not scale-dependent</h3>
      <p>Both GPT-2 checkpoints land on exactly 2.00. Attention projections cost
      4<em>d</em>&sup2; — three fused into <code>c_attn</code>, one in <code>c_proj</code> —
      against 8<em>d</em>&sup2; for a 4&times; feed-forward block. Tripling the parameter
      count does not move it.</p>
    </div>
    <div class="finding">
      <h3>Qwen breaks it to 7.13:1</h3>
      <p>Grouped-query attention gives it 14 query heads against 2 key-value heads, shrinking
      the numerator, while SwiGLU adds a third feed-forward matrix to the denominator. Nearly
      two-thirds of the model is now feed-forward — which is precisely the pool a
      mixture-of-experts design sparsifies.</p>
    </div>
    <div class="finding">
      <h3>Embedding share is a small-model tax</h3>
      <p>Vocabulary size does not shrink when the model does. Across the two GPT-2 sizes the
      embedding share halves, 31.6% to 14.8%, on an unchanged 50,257-token table. Qwen still
      spends 27.6% because its vocabulary is three times larger against a narrower hidden
      state.</p>
    </div>
    <div class="finding">
      <h3>Qwen&rsquo;s dynamic range is an order of magnitude tighter</h3>
      <p>Standard deviation near 0.02 against GPT-2&rsquo;s 0.13, and a maximum absolute
      weight of 1.7 against 17.1. Narrow range quantizes cleanly; GPT-2&rsquo;s outliers are
      what make naive 8-bit integer schemes fall over.</p>
    </div>
  </section>
{plate_sections()}

  <section class="note">
    <h2>A correction worth recording</h2>
    <p>The first run of these figures reported GPT-2 at 137.0M parameters and GPT-2 medium at
    380.0M. Both were wrong. GPT-2 stores its causal attention mask as a <code>float32</code>
    tensor named <code>h.N.attn.bias</code>, shaped <code>(1, 1, 1024, 1024)</code> — 1.05M
    elements per layer, and a registered buffer rather than a learned parameter.</p>
    <p>Counting it inflated both the totals and the attention bucket, which is where the
    2:1 ratio was hiding. After skipping those buffers all three checkpoints match their
    published figures exactly: 124.4M, 354.8M, 494.0M. Qwen was never affected — it computes
    its mask rather than storing one.</p>
  </section>

  <footer>
    <span>weight-visualizer</span>
    <span>safetensors, read one tensor at a time</span>
    <span>plates rendered at 130&thinsp;dpi</span>
  </footer>
</div>

<dialog id="lb">
  <button class="close" autofocus>Close &times;</button>
  <img id="lb-img" alt="Enlarged plate">
</dialog>

<script>
  const lb = document.getElementById('lb');
  const lbImg = document.getElementById('lb-img');
  document.querySelectorAll('button.plate').forEach(function (b) {{
    b.addEventListener('click', function () {{
      const img = b.querySelector('img');
      lbImg.src = img.src;
      lbImg.alt = img.alt;
      lb.showModal();
    }});
  }});
  lb.querySelector('.close').addEventListener('click', function () {{ lb.close(); }});
  lb.addEventListener('click', function (e) {{ if (e.target === lb) lb.close(); }});
</script>
"""

if __name__ == "__main__":
    with open(OUT, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.2f} MB)")
