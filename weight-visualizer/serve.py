#!/usr/bin/env python3
"""Live weight explorer — serves any window of any tensor straight off the checkpoint.

Nothing is precomputed and nothing is written to disk. safetensors supports lazy
slicing, so a request for rows 400-600 of a 4864x896 matrix reads only those rows
from the file. The browser asks for the window it is currently showing, at a
stride matched to the zoom level, and draws it on a canvas.

    weight-visualizer serve --model gpt2
    weight-visualizer serve --model Qwen/Qwen2.5-0.5B --port 8888

Stdlib only — no Flask, no extra dependency in the image.
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import BUFFER_RE, classify  # noqa: E402
from loader import Checkpoint  # noqa: E402

MAX_CELLS = 240_000     # per request, keeps a window under ~1 MB on the wire
STATE = {"ckpt": None, "index": {}, "model": ""}


def build_index(model):
    """Catalogue every 2D weight tensor. safetensors reads headers only; a .pt
    is loaded once by the Checkpoint accessor and served from memory."""
    ckpt = Checkpoint(model)
    index = {}
    for name in ckpt.keys():
        if BUFFER_RE.search(name):
            continue
        shape = ckpt.shape(name)
        if len(shape) != 2:
            continue
        index[name] = {"shape": list(shape), "role": classify(name)}
    STATE.update(ckpt=ckpt, index=index, model=model)
    return index


def read_window(name, r0, c0, rows, cols, stride):
    """Read one rectangle, striding to stay under the cell budget."""
    info = STATE["index"][name]
    R, C = info["shape"]
    r0 = max(0, min(r0, R - 1))
    c0 = max(0, min(c0, C - 1))
    r1 = min(R, r0 + max(1, rows) * stride)
    c1 = min(C, c0 + max(1, cols) * stride)

    arr = np.asarray(STATE["ckpt"].window(name, r0, r1, c0, c1))
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if stride > 1:
        arr = arr[::stride, ::stride]
    if arr.size > MAX_CELLS:                      # belt and braces
        k = int(np.ceil(np.sqrt(arr.size / MAX_CELLS)))
        arr = arr[::k, ::k]
        stride *= k
    return np.ascontiguousarray(arr, dtype=np.float32), r0, c0, stride


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                                       # quiet; the console is the UI

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")

        if u.path == "/api/tensors":
            payload = {
                "model": STATE["model"],
                "tensors": [
                    {"name": n, "shape": v["shape"], "role": v["role"],
                     "params": v["shape"][0] * v["shape"][1]}
                    for n, v in sorted(STATE["index"].items(),
                                       key=lambda kv: -kv[1]["shape"][0] * kv[1]["shape"][1])
                ],
            }
            return self._send(200, json.dumps(payload).encode(), "application/json")

        if u.path == "/api/window":
            try:
                name = q["name"][0]
                if name not in STATE["index"]:
                    return self._send(404, b"unknown tensor", "text/plain")
                arr, r0, c0, stride = read_window(
                    name,
                    int(q.get("r0", [0])[0]), int(q.get("c0", [0])[0]),
                    int(q.get("rows", [256])[0]), int(q.get("cols", [256])[0]),
                    max(1, int(q.get("stride", [1])[0])))
            except (KeyError, ValueError) as e:
                return self._send(400, str(e).encode(), "text/plain")

            head = {"r0": r0, "c0": c0, "rows": arr.shape[0], "cols": arr.shape[1],
                    "stride": stride,
                    "absmax": float(np.abs(arr).max()) if arr.size else 0.0}
            return self._send(200, arr.tobytes(), "application/octet-stream",
                              {"X-Window": json.dumps(head)})

        if u.path == "/api/value":
            try:
                name = q["name"][0]
                r, c = int(q["r"][0]), int(q["c"][0])
                v = float(np.asarray(
                    STATE["ckpt"].window(name, r, r + 1, c, c + 1)).ravel()[0])
            except (KeyError, ValueError, IndexError) as e:
                return self._send(400, str(e).encode(), "text/plain")
            return self._send(200, json.dumps({"v": v, "r": r, "c": c}).encode(),
                              "application/json")

        self._send(404, b"not found", "text/plain")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live weight explorer</title>
<style>
  :root {
    --ground:#F4F6F8; --surface:#FFF; --ink:#161A21; --muted:#5C6675;
    --faint:#8C95A3; --line:#DCE1E8; --accent:#4C72B0; --pos:#DD8452;
    --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
    --sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  }
  @media (prefers-color-scheme:dark){:root{
    --ground:#0F1319; --surface:#171C24; --ink:#E3E8EF; --muted:#98A2B1;
    --faint:#6E7887; --line:#262D38; --accent:#7CA0D4; --pos:#E09A6C;
  }}
  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
       font-size:14px;line-height:1.5}
  .bar{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;
       padding:12px 18px;border-bottom:1px solid var(--line);background:var(--surface)}
  .bar h1{font-size:14px;font-weight:600;margin:0}
  .bar .sub{font-family:var(--mono);font-size:11px;color:var(--faint)}
  select{font-family:var(--mono);font-size:12px;padding:5px 8px;max-width:52ch;
         background:var(--surface);color:var(--ink);border:1px solid var(--line);
         border-radius:2px}
  main{display:grid;grid-template-columns:1fr 250px;gap:0;height:calc(100vh - 49px)}
  .stage{position:relative;background:#FFF;overflow:hidden;touch-action:none}
  @media (prefers-color-scheme:dark){.stage{background:#EDEFF2}}
  canvas{display:block;width:100%;height:100%;cursor:crosshair}
  canvas.panning{cursor:grabbing}
  .hud{position:absolute;left:12px;top:12px;font-family:var(--mono);font-size:11px;
       color:#5C6675;background:rgba(255,255,255,.92);padding:4px 9px;border-radius:2px;
       font-variant-numeric:tabular-nums}
  .zoombar{position:absolute;left:12px;bottom:12px;display:flex;gap:5px}
  .zoombar button{font-family:var(--mono);font-size:13px;cursor:pointer;width:30px;
       height:28px;border:1px solid rgba(22,26,33,.18);background:rgba(255,255,255,.92);
       color:#161A21;border-radius:2px}
  .zoombar button.wide{width:auto;padding:0 10px;font-size:11px}
  aside{border-left:1px solid var(--line);background:var(--surface);padding:16px;
        display:grid;gap:14px;align-content:start;overflow-y:auto}
  h4{margin:0;font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
     text-transform:uppercase;color:var(--faint);font-weight:400}
  .val{font-family:var(--mono);font-size:20px;font-weight:500;word-break:break-all;
       font-variant-numeric:tabular-nums}
  .val.p{color:var(--pos)} .val.n{color:var(--accent)}
  .val.none{color:var(--faint);font-size:13px;font-weight:400}
  .kv{display:grid;gap:4px}
  .kv div{display:flex;justify-content:space-between;gap:10px}
  .kv span:first-child{font-size:12px;color:var(--muted)}
  .kv span:last-child{font-family:var(--mono);font-size:12px;
                      font-variant-numeric:tabular-nums}
  .ramp{height:9px;border-radius:1px;
        background:linear-gradient(90deg,#4C72B0,#F7F7F5,#DD8452)}
  .ends{display:flex;justify-content:space-between;font-family:var(--mono);
        font-size:10px;color:var(--faint);font-variant-numeric:tabular-nums}
  .hint{font-size:12px;color:var(--muted);border-top:1px solid var(--line);
        padding-top:11px}
  @media (max-width:820px){main{grid-template-columns:1fr;height:auto}
    .stage{height:60vh}aside{border-left:none;border-top:1px solid var(--line)}}
</style></head><body>

<div class="bar">
  <h1>Live weight explorer</h1>
  <span class="sub" id="model">&mdash;</span>
  <select id="pick"></select>
  <span class="sub" id="shape">&mdash;</span>
</div>

<main>
  <div class="stage">
    <canvas id="cv"></canvas>
    <div class="hud" id="hud">&mdash;</div>
    <div class="zoombar">
      <button id="zin">+</button><button id="zout">&minus;</button>
      <button id="zfit" class="wide">fit</button>
    </div>
  </div>
  <aside>
    <div><h4>Value</h4><p class="val none" id="val">click a cell</p></div>
    <div class="kv">
      <div><span>row</span><span id="r">&mdash;</span></div>
      <div><span>column</span><span id="c">&mdash;</span></div>
      <div><span>window max |w|</span><span id="mx">&mdash;</span></div>
      <div><span>stride</span><span id="st">&mdash;</span></div>
    </div>
    <div><div class="ramp" id="ramp"></div>
      <div class="ends"><span id="lo">&mdash;</span><span>0</span>
        <span id="hi">&mdash;</span></div></div>
    <p class="hint">Scroll to zoom, drag to pan. Below 1&times; the server strides the
      read, so you are seeing a sampled view; every cell is exact at 1&times; and above.
      Past 26&times; each cell prints its own number. Clicking always fetches the true
      value from the checkpoint.</p>
  </aside>
</main>

<script>
(function(){
var cv=document.getElementById('cv'),ctx=cv.getContext('2d');
var off=document.createElement('canvas'),octx=off.getContext('2d');
var NEG=[76,114,176],POS=[221,132,82],MID=[247,247,245];
var tensors=[],cur=null,zoom=1,panX=0,panY=0,win=null,pinned=null,pend=null;

function ramp(t){var a=t<0?NEG:POS,k=Math.min(1,Math.abs(t));k=Math.pow(k,0.55);
  return[Math.round(MID[0]+(a[0]-MID[0])*k),Math.round(MID[1]+(a[1]-MID[1])*k),
         Math.round(MID[2]+(a[2]-MID[2])*k)];}

function resize(){var r=cv.getBoundingClientRect(),d=window.devicePixelRatio||1;
  cv.width=Math.round(r.width*d);cv.height=Math.round(r.height*d);
  ctx.setTransform(d,0,0,d,0,0);}

function fit(){var r=cv.getBoundingClientRect();
  zoom=Math.min(r.width/cur.shape[1],r.height/cur.shape[0]);
  panX=(r.width-cur.shape[1]*zoom)/2;panY=(r.height-cur.shape[0]*zoom)/2;
  fetchWindow();}

function fetchWindow(){
  if(!cur)return;
  var r=cv.getBoundingClientRect();
  var stride=Math.max(1,Math.round(1/zoom));
  var c0=Math.max(0,Math.floor(-panX/zoom)),r0=Math.max(0,Math.floor(-panY/zoom));
  var cols=Math.ceil(r.width/zoom/stride)+2,rows=Math.ceil(r.height/zoom/stride)+2;
  var url='/api/window?name='+encodeURIComponent(cur.name)+'&r0='+r0+'&c0='+c0+
          '&rows='+rows+'&cols='+cols+'&stride='+stride;
  if(pend)pend.abort();
  var ac=new AbortController();pend=ac;
  fetch(url,{signal:ac.signal}).then(function(res){
    var h=JSON.parse(res.headers.get('X-Window'));
    return res.arrayBuffer().then(function(b){return{h:h,v:new Float32Array(b)};});
  }).then(function(d){
    pend=null;win=d.h;win.vals=d.v;paint();
  }).catch(function(e){if(e.name!=='AbortError')console.error(e);});
}

function paint(){
  var r=cv.getBoundingClientRect();
  ctx.clearRect(0,0,r.width,r.height);
  if(!win)return;
  var W=win.cols,H=win.rows,am=win.absmax||1;
  off.width=W;off.height=H;
  var img=octx.createImageData(W,H),d=img.data;
  for(var i=0;i<W*H;i++){var col=ramp(win.vals[i]/am);
    d[i*4]=col[0];d[i*4+1]=col[1];d[i*4+2]=col[2];d[i*4+3]=255;}
  octx.putImageData(img,0,0);
  ctx.imageSmoothingEnabled=false;
  var cell=zoom*win.stride;
  ctx.drawImage(off,panX+win.c0*zoom,panY+win.r0*zoom,W*cell,H*cell);

  if(zoom>=26&&win.stride===1){
    var fs=Math.min(11,zoom/3.4);
    ctx.font='500 '+fs.toFixed(1)+'px '+getComputedStyle(document.body)
      .getPropertyValue('--mono');
    ctx.textAlign='center';ctx.textBaseline='middle';
    for(var rr=0;rr<H;rr++)for(var cc=0;cc<W;cc++){
      var v=win.vals[rr*W+cc];
      var x=panX+(win.c0+cc+0.5)*zoom,y=panY+(win.r0+rr+0.5)*zoom;
      if(x<-40||y<-20||x>r.width+40||y>r.height+20)continue;
      ctx.fillStyle=Math.abs(v)/am>0.55?'#FFF':'#2A2F38';
      ctx.fillText(v.toFixed(2),x,y);
    }
  }
  if(pinned){ctx.strokeStyle='#161A21';ctx.lineWidth=2;
    ctx.strokeRect(panX+pinned.c*zoom,panY+pinned.r*zoom,zoom,zoom);}
  document.getElementById('hud').textContent=
    zoom.toFixed(zoom<1?3:1)+'×  rows '+win.r0+'–'+
    (win.r0+H*win.stride)+'  cols '+win.c0+'–'+(win.c0+W*win.stride);
  document.getElementById('mx').textContent=am.toFixed(4);
  document.getElementById('st').textContent=win.stride;
  document.getElementById('lo').textContent=(-am).toFixed(2);
  document.getElementById('hi').textContent='+'+am.toFixed(2);
}

function zoomBy(f,cx,cy){var r=cv.getBoundingClientRect();
  if(cx===undefined){cx=r.width/2;cy=r.height/2;}
  var nz=Math.max(0.02,Math.min(90,zoom*f));
  panX=cx-(cx-panX)*(nz/zoom);panY=cy-(cy-panY)*(nz/zoom);zoom=nz;
  paint();fetchWindow();}

cv.addEventListener('wheel',function(e){e.preventDefault();
  var r=cv.getBoundingClientRect();
  zoomBy(e.deltaY<0?1.18:1/1.18,e.clientX-r.left,e.clientY-r.top);},{passive:false});

var drag=null,moved=0;
cv.addEventListener('pointerdown',function(e){drag={x:e.clientX,y:e.clientY};moved=0;
  cv.setPointerCapture(e.pointerId);cv.classList.add('panning');});
cv.addEventListener('pointermove',function(e){if(!drag)return;
  panX+=e.clientX-drag.x;panY+=e.clientY-drag.y;
  moved+=Math.abs(e.clientX-drag.x)+Math.abs(e.clientY-drag.y);
  drag={x:e.clientX,y:e.clientY};paint();});
cv.addEventListener('pointerup',function(e){cv.classList.remove('panning');
  if(drag&&moved<4)click(e);else fetchWindow();drag=null;});

function click(e){
  var r=cv.getBoundingClientRect();
  var c=Math.floor((e.clientX-r.left-panX)/zoom),rw=Math.floor((e.clientY-r.top-panY)/zoom);
  if(c<0||rw<0||c>=cur.shape[1]||rw>=cur.shape[0])return;
  pinned={r:rw,c:c};paint();
  fetch('/api/value?name='+encodeURIComponent(cur.name)+'&r='+rw+'&c='+c)
    .then(function(x){return x.json();}).then(function(d){
      var el=document.getElementById('val');
      el.textContent=(d.v>=0?'+':'')+d.v.toPrecision(7);
      el.className='val '+(d.v>=0?'p':'n');
      document.getElementById('r').textContent=d.r;
      document.getElementById('c').textContent=d.c;
    });
}

document.getElementById('zin').onclick=function(){zoomBy(1.5);};
document.getElementById('zout').onclick=function(){zoomBy(1/1.5);};
document.getElementById('zfit').onclick=fit;
window.addEventListener('resize',function(){resize();paint();fetchWindow();});

fetch('/api/tensors').then(function(r){return r.json();}).then(function(d){
  document.getElementById('model').textContent=d.model;
  tensors=d.tensors;
  var sel=document.getElementById('pick');
  tensors.forEach(function(t,i){
    var o=document.createElement('option');o.value=i;
    o.textContent=t.name+'  ['+t.shape[0]+'×'+t.shape[1]+']';
    sel.appendChild(o);});
  sel.onchange=function(){pick(+sel.value);};
  // Open on the biggest matrix that is not an extreme sliver — an embedding
  // table is 65:1 and reads as a hairline at fit zoom.
  var start=0;
  for(var i=0;i<tensors.length;i++){
    var s=tensors[i].shape,ar=Math.max(s[0],s[1])/Math.min(s[0],s[1]);
    if(ar<=4.5){start=i;break;}
  }
  sel.value=start;
  resize();pick(start);
});

function pick(i){cur=tensors[i];pinned=null;win=null;
  document.getElementById('shape').textContent=
    cur.shape[0]+'×'+cur.shape[1]+' = '+cur.params.toLocaleString()+' weights';
  var el=document.getElementById('val');el.textContent='click a cell';el.className='val none';
  document.getElementById('r').textContent='—';
  document.getElementById('c').textContent='—';
  fit();}
})();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--host", default="0.0.0.0")
    a = p.parse_args()

    print(f"indexing {a.model} ...")
    idx = build_index(a.model)
    total = sum(v["shape"][0] * v["shape"][1] for v in idx.values())
    print(f"  {STATE['ckpt'].kind}: {len(idx)} 2D tensors, "
          f"{total / 1e6:.2f}M weights reachable")
    print(f"serving on http://127.0.0.1:{a.port}  (ctrl-c to stop)")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
