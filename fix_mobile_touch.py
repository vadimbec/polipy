"""Fix mobile pinch-zoom: disable Plotly drag, handle all touch ourselves."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('vizu.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][0]['source'])

# 1. Fix mobile axis title
src = src.replace(
    'text="← Modeste ── Revenu commune naissance ── Aisée →"',
    'text="← Modeste ── Revenu circo (€/UC) ── Aisée →"'
)

# 2. Change dragmode from 'pan' to False
src = src.replace("dragmode='pan',", "dragmode=False,")

# 3. Fix mobile info card label
src = src.replace(
    '<span class="lbl">Revenu commune naissance :</span>',
    '<span class="lbl">Revenu circo :</span>'
)

# 4. Replace the entire touch handling section
OLD_START = "// ── Gestion touch custom : pan 1 doigt + pinch-zoom 2 doigts + tap"
OLD_END = "}}, {{ passive: true }});\n"

idx_start = src.find(OLD_START)
if idx_start < 0:
    print("ERROR: could not find touch section start")
    sys.exit(1)

# Find the LAST occurrence of the passive:true listener close after the start
remaining = src[idx_start:]
# We need to find the 3rd occurrence of "}}, {{ passive: true }});" after the start
# (touchstart, touchmove, touchend)
pos = 0
for i in range(3):
    found = remaining.find(OLD_END, pos)
    if found < 0:
        print(f"ERROR: could not find {i+1}th passive listener end")
        sys.exit(1)
    pos = found + len(OLD_END)

idx_end = idx_start + pos

print(f"Touch section: chars {idx_start} to {idx_end}")
print(f"First 80 chars: {src[idx_start:idx_start+80]!r}")
print(f"Last 80 chars:  {src[idx_end-80:idx_end]!r}")

NEW_TOUCH = """// ── Gestion touch : pan 1 doigt + pinch-zoom 2 doigts + tap ──────────
// dragmode=false → on gère TOUT. Listeners capture + non-passive.

let gesture = null; // null | 'pan' | 'pinch'
let touch1  = null;
let pinch   = null;
let rafId   = null;

function getRange() {{
  return {{
    x: chartDiv._fullLayout.xaxis.range.slice(),
    y: chartDiv._fullLayout.yaxis.range.slice(),
  }};
}}

function px2x(px) {{ const a = chartDiv._fullLayout.xaxis; return a.p2d(px - a._offset); }}
function px2y(py) {{ const a = chartDiv._fullLayout.yaxis; return a.p2d(py - a._offset); }}

function findNearest(cx, cy) {{
  const ax = chartDiv._fullLayout.xaxis;
  const ay = chartDiv._fullLayout.yaxis;
  let best = null, bestD = Infinity;
  for (let ti = 1; ti < chartDiv.data.length; ti++) {{
    const tr = chartDiv.data[ti];
    if (tr.visible === false || !tr.customdata) continue;
    for (let pi = 0; pi < tr.x.length; pi++) {{
      const sx = ax.d2p(tr.x[pi]) + ax._offset;
      const sy = ay.d2p(tr.y[pi]) + ay._offset;
      const d = Math.hypot(cx - sx, cy - sy);
      if (d < bestD && d < 28) {{ bestD = d; best = {{ti, pi}}; }}
    }}
  }}
  return best;
}}

function doRelayout(xr, yr) {{
  if (rafId) cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(() => {{
    Plotly.relayout(chartDiv, {{'xaxis.range': xr, 'yaxis.range': yr}});
    rafId = null;
  }});
}}

chartDiv.addEventListener('touchstart', function(e) {{
  e.preventDefault();
  e.stopPropagation();

  if (e.touches.length >= 2) {{
    gesture = 'pinch';
    const t1 = e.touches[0], t2 = e.touches[1];
    const rect = chartDiv.getBoundingClientRect();
    pinch = {{
      dist: Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY),
      midPxX: (t1.clientX + t2.clientX) / 2 - rect.left,
      midPxY: (t1.clientY + t2.clientY) / 2 - rect.top,
      startRanges: getRange(),
    }};
    touch1 = null;
  }} else if (e.touches.length === 1) {{
    gesture = null;
    const t = e.touches[0];
    const rect = chartDiv.getBoundingClientRect();
    touch1 = {{
      clientX: t.clientX, clientY: t.clientY,
      time: Date.now(),
      startPxX: t.clientX - rect.left,
      startPxY: t.clientY - rect.top,
      startRanges: getRange(),
      moved: false,
    }};
  }}
}}, {{ passive: false, capture: true }});

chartDiv.addEventListener('touchmove', function(e) {{
  e.preventDefault();
  e.stopPropagation();

  if (e.touches.length >= 2 && pinch) {{
    gesture = 'pinch';
    const t1 = e.touches[0], t2 = e.touches[1];
    const rect = chartDiv.getBoundingClientRect();
    const newDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
    const newMidPxX = (t1.clientX + t2.clientX) / 2 - rect.left;
    const newMidPxY = (t1.clientY + t2.clientY) / 2 - rect.top;
    const scale = pinch.dist / newDist;

    // Anchor = where pinch started, in data coords at start
    const sr = pinch.startRanges;
    const ax = chartDiv._fullLayout.xaxis;
    const ay = chartDiv._fullLayout.yaxis;

    // Use start ranges to compute anchor in data coords
    const xSpan0 = sr.x[1] - sr.x[0];
    const ySpan0 = sr.y[1] - sr.y[0];
    const plotW = ax._length;
    const plotH = ay._length;

    // Anchor in data coords (relative to start ranges)
    const anchorX = sr.x[0] + (pinch.midPxX - ax._offset) / plotW * xSpan0;
    const anchorY = sr.y[1] - (pinch.midPxY - ay._offset) / plotH * ySpan0;

    // New mid in data coords (relative to start ranges)
    const newMidX = sr.x[0] + (newMidPxX - ax._offset) / plotW * xSpan0;
    const newMidY = sr.y[1] - (newMidPxY - ay._offset) / plotH * ySpan0;

    // Scale around anchor
    let x0 = anchorX + (sr.x[0] - anchorX) * scale;
    let x1 = anchorX + (sr.x[1] - anchorX) * scale;
    let y0 = anchorY + (sr.y[0] - anchorY) * scale;
    let y1 = anchorY + (sr.y[1] - anchorY) * scale;

    // Pan: shift by finger movement (in data units at current scale)
    const newXSpan = x1 - x0;
    const newYSpan = y1 - y0;
    const panDxPx = newMidPxX - pinch.midPxX;
    const panDyPx = newMidPxY - pinch.midPxY;
    const panDx = panDxPx / plotW * newXSpan;
    const panDy = panDyPx / plotH * newYSpan;

    doRelayout([x0 - panDx, x1 - panDx], [y0 + panDy, y1 + panDy]);
  }} else if (e.touches.length === 1 && touch1 && gesture !== 'pinch') {{
    gesture = 'pan';
    touch1.moved = true;
    const t = e.touches[0];
    const rect = chartDiv.getBoundingClientRect();
    const curPxX = t.clientX - rect.left;
    const curPxY = t.clientY - rect.top;

    const sr = touch1.startRanges;
    const ax = chartDiv._fullLayout.xaxis;
    const ay = chartDiv._fullLayout.yaxis;
    const xSpan = sr.x[1] - sr.x[0];
    const ySpan = sr.y[1] - sr.y[0];
    const plotW = ax._length;
    const plotH = ay._length;

    const dx = (curPxX - touch1.startPxX) / plotW * xSpan;
    const dy = (curPxY - touch1.startPxY) / plotH * ySpan;

    doRelayout([sr.x[0] - dx, sr.x[1] - dx], [sr.y[0] + dy, sr.y[1] + dy]);
  }}
}}, {{ passive: false, capture: true }});

chartDiv.addEventListener('touchend', function(e) {{
  e.stopPropagation();

  if (gesture === 'pinch' && e.touches.length < 2) {{
    gesture = null;
    pinch = null;
    if (e.touches.length === 1) {{
      const t = e.touches[0];
      const rect = chartDiv.getBoundingClientRect();
      touch1 = {{
        clientX: t.clientX, clientY: t.clientY,
        time: Date.now(),
        startPxX: t.clientX - rect.left,
        startPxY: t.clientY - rect.top,
        startRanges: getRange(),
        moved: false,
      }};
    }} else {{
      touch1 = null;
    }}
    return;
  }}

  if (e.touches.length === 0 && touch1 && !touch1.moved) {{
    const t = e.changedTouches[0];
    const dx = Math.abs(t.clientX - touch1.clientX);
    const dy = Math.abs(t.clientY - touch1.clientY);
    const dt = Date.now() - touch1.time;

    if (dx <= 12 && dy <= 12 && dt <= 300) {{
      const rect = chartDiv.getBoundingClientRect();
      const hit = findNearest(t.clientX - rect.left, t.clientY - rect.top);
      if (hit) {{
        showCard(chartDiv.data[hit.ti], hit.pi);
      }} else {{
        card.classList.remove('show');
      }}
    }}
  }}

  if (e.touches.length === 0) {{
    gesture = null;
    touch1 = null;
    pinch = null;
  }}
}}, {{ passive: false, capture: true }});

chartDiv.addEventListener('touchcancel', function() {{
  gesture = null; touch1 = null; pinch = null;
}}, {{ passive: true }});
"""

src = src[:idx_start] + NEW_TOUCH + src[idx_end:]

nb['cells'][0]['source'] = [src]

with open('vizu.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("OK - vizu.ipynb updated with new touch handling")
