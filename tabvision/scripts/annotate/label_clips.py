"""Browser-based labeling tool — closes Phase 3 + Phase 4 acceptance gates.

Single-file Flask app.  Run::

    python -m scripts.annotate.label_clips --clips test-data/eval-clips
    # (then open http://localhost:5005 in any browser — works fine in WSL
    #  pointed at a Windows browser)

Three label modes:

- ``framing``  — per-clip "good"/"bad" + tag list (Phase 3 preflight gate).
- ``fretboard`` — click 4 fret-intersections (frets 5+12, top+bottom edges)
  on one representative frame per clip (Phase 3 keypoint-fretboard gate).
- ``fingering`` — per-frame ``(string, fret)`` for each fretting finger,
  on N evenly-spaced frames per clip (Phase 4 acceptance gate).

Storage is plain JSON under ``tabvision/data/eval/{framing,fretboard,fingering}/``;
schema lives in :mod:`scripts.annotate.storage`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.annotate import storage
from scripts.annotate.frames import (
    encode_jpeg,
    evenly_spaced_frame_indices,
    probe_clip,
    read_frame,
    representative_frame_idx,
)

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


def make_app(clips: list[Path], eval_root: Path | None, fingering_frames: int):  # noqa: PLR0915
    """Build the Flask app.  Returns the `Flask` instance.

    Kept as a factory so unit tests can construct an app over a fixture
    set without needing a real CLI invocation.
    """
    try:
        from flask import Flask, abort, jsonify, render_template_string, request
    except ImportError as exc:
        raise SystemExit(
            "flask is required for the labeling tool. "
            "Install with: pip install flask"
        ) from exc

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    clip_index = {storage.clip_id(c): c for c in clips}
    if len(clip_index) != len(clips):
        dups = [c for c in clips if list(clip_index.values()).count(c) == 0]
        raise SystemExit(f"two clips resolve to the same clip_id; rename one of: {dups}")
    sorted_cids: list[str] = sorted(clip_index.keys())

    def _neighbours(cid: str) -> tuple[str | None, str | None]:
        """Return ``(prev_cid, next_cid)`` in the sorted clip order."""
        try:
            i = sorted_cids.index(cid)
        except ValueError:
            return None, None
        prev_cid = sorted_cids[i - 1] if i > 0 else None
        next_cid = sorted_cids[i + 1] if i < len(sorted_cids) - 1 else None
        return prev_cid, next_cid

    # ----- routes -----

    @app.route("/")
    def index():
        rows = []
        for cid, path in sorted(clip_index.items()):
            rows.append({
                "id": cid,
                "path": str(path),
                "framing": _summary_framing(path, eval_root),
                "fretboard": _summary_fretboard(path, eval_root),
                "fingering": _summary_fingering(path, eval_root, fingering_frames),
            })
        return render_template_string(_TPL_INDEX, rows=rows)

    @app.route("/clip/<cid>/frame/<int:frame_idx>.jpg")
    def clip_frame(cid: str, frame_idx: int):
        path = _resolve_clip(cid, clip_index)
        try:
            return encode_jpeg(read_frame(path, frame_idx)), 200, {"Content-Type": "image/jpeg"}
        except IndexError:
            abort(404)

    # framing ---------------------------------------------------------

    @app.route("/framing/<cid>", methods=["GET"])
    def framing_get(cid: str):
        path = _resolve_clip(cid, clip_index)
        meta = probe_clip(path)
        frame_idx = representative_frame_idx(meta)
        existing = storage.load_framing(path, eval_root=eval_root)
        prev_cid, next_cid = _neighbours(cid)
        return render_template_string(
            _TPL_FRAMING,
            cid=cid,
            clip_path=str(path),
            frame_idx=frame_idx,
            tags=storage.FRAMING_TAGS,
            existing=existing.to_json() if existing else None,
            prev_cid=prev_cid,
            next_cid=next_cid,
        )

    @app.route("/framing/<cid>", methods=["POST"])
    def framing_post(cid: str):
        path = _resolve_clip(cid, clip_index)
        body = request.get_json(force=True)
        label = storage.FramingLabel(
            clip_path=str(path),
            label=body["label"],
            tags=list(body.get("tags", [])),
            notes=str(body.get("notes", "")).strip(),
        )
        storage.save_framing(label, eval_root=eval_root)
        return jsonify(status="ok")

    # fretboard -------------------------------------------------------

    @app.route("/fretboard/<cid>", methods=["GET"])
    def fretboard_get(cid: str):
        path = _resolve_clip(cid, clip_index)
        meta = probe_clip(path)
        frame_idx = representative_frame_idx(meta)
        existing = storage.load_fretboard(path, eval_root=eval_root)
        if existing:
            frame_idx = existing.frame_idx
        prev_cid, next_cid = _neighbours(cid)
        return render_template_string(
            _TPL_FRETBOARD,
            cid=cid,
            clip_path=str(path),
            frame_idx=frame_idx,
            n_frames=meta.n_frames,
            existing=existing.to_json() if existing else None,
            prev_cid=prev_cid,
            next_cid=next_cid,
        )

    @app.route("/fretboard/<cid>", methods=["POST"])
    def fretboard_post(cid: str):
        path = _resolve_clip(cid, clip_index)
        body = request.get_json(force=True)
        label = storage.FretboardLabel(
            clip_path=str(path),
            frame_idx=int(body["frame_idx"]),
            points=[
                storage.FretIntersection(
                    fret=int(p["fret"]),
                    edge=p["edge"],
                    x=float(p["x"]),
                    y=float(p["y"]),
                )
                for p in body["points"]
            ],
            notes=str(body.get("notes", "")).strip(),
        )
        if not label.is_complete():
            return jsonify(status="incomplete"), 400
        storage.save_fretboard(label, eval_root=eval_root)
        return jsonify(status="ok")

    # fingering -------------------------------------------------------

    @app.route("/fingering/<cid>", methods=["GET"])
    def fingering_get(cid: str):
        path = _resolve_clip(cid, clip_index)
        meta = probe_clip(path)
        sampled = evenly_spaced_frame_indices(meta.n_frames, fingering_frames)
        existing = storage.load_fingering(path, eval_root=eval_root)
        existing_by_idx: dict[int, list[dict]] = {}
        if existing:
            for fr in existing.frames:
                existing_by_idx[fr.frame_idx] = [
                    {"finger": fl.finger, "string": fl.string, "fret": fl.fret}
                    for fl in fr.fingers
                ]
        prev_cid, next_cid = _neighbours(cid)
        return render_template_string(
            _TPL_FINGERING,
            cid=cid,
            clip_path=str(path),
            sampled=sampled,
            n_strings=6,
            max_fret=12,
            fingers=storage.FINGER_NAMES,
            existing_by_idx=existing_by_idx,
            prev_cid=prev_cid,
            next_cid=next_cid,
        )

    @app.route("/fingering/<cid>", methods=["POST"])
    def fingering_post(cid: str):
        path = _resolve_clip(cid, clip_index)
        body = request.get_json(force=True)
        label = storage.FingeringLabel(
            clip_path=str(path),
            frames=[
                storage.FrameLabel(
                    frame_idx=int(fr["frame_idx"]),
                    fingers=[
                        storage.FingerLabel(
                            finger=fl["finger"],
                            string=_optional_int(fl.get("string")),
                            fret=_optional_int(fl.get("fret")),
                        )
                        for fl in fr.get("fingers", [])
                    ],
                )
                for fr in body["frames"]
            ],
        )
        storage.save_fingering(label, eval_root=eval_root)
        return jsonify(status="ok")

    return app


# ----- summary helpers -----


def _summary_framing(path: Path, eval_root: Path | None) -> dict:
    label = storage.load_framing(path, eval_root=eval_root)
    if label is None:
        return {"done": False, "text": "—"}
    suffix = f" ({', '.join(label.tags)})" if label.tags else ""
    return {"done": True, "text": label.label + suffix}


def _optional_int(value) -> int | None:  # noqa: ANN001 — JSON value, any of None/str/int
    """Coerce a request-body value to int, treating None and empty string as None."""
    if value in (None, ""):
        return None
    return int(value)


def _summary_fretboard(path: Path, eval_root: Path | None) -> dict:
    label = storage.load_fretboard(path, eval_root=eval_root)
    if label is None:
        return {"done": False, "text": "0 / 4"}
    return {
        "done": label.is_complete(),
        "text": f"{len(label.points)} / 4",
    }


def _summary_fingering(path: Path, eval_root: Path | None, expected: int) -> dict:
    label = storage.load_fingering(path, eval_root=eval_root)
    if label is None:
        return {"done": False, "text": f"0 / {expected}"}
    return {
        "done": len(label.frames) >= expected,
        "text": f"{len(label.frames)} / {expected}",
    }


def _resolve_clip(cid: str, clip_index: dict[str, Path]) -> Path:
    if cid not in clip_index:
        from flask import abort
        abort(404)
    return clip_index[cid]


# ----- HTML templates -----
# Inlined to keep the package single-file.  Vanilla JS, no frameworks.

_BASE_CSS = """
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, system-ui, sans-serif; background:#1a1a1a; color:#e8e8e8;
         margin: 1.2em; }
  h1, h2 { font-weight: 500; }
  a { color:#7fc8ff; }
  table { border-collapse: collapse; width: 100%; }
  th, td { padding: 0.4em 0.6em; border-bottom: 1px solid #333; text-align: left; }
  th { color:#888; font-weight: 500; }
  .done { color: #6ed68a; }
  .pending { color: #d6a96e; }
  button { background:#2a4d80; border: none; color:#fff; padding:0.5em 0.9em;
           border-radius: 4px; cursor: pointer; font-size: 0.95em; margin-right: 0.5em; }
  button:hover { background:#386bb1; }
  button.danger { background:#7a3434; }
  button.ghost { background:#333; }
  button:disabled { background:#444; color:#888; cursor: not-allowed; }
  .tagrow label { margin-right: 1em; }
  .marker { position: absolute; width: 14px; height: 14px; border-radius: 50%;
            border: 2px solid #ff5cb0; transform: translate(-50%, -50%); pointer-events: none; }
  .marker-label { position: absolute; transform: translate(-50%, calc(-50% - 18px));
                  font-size: 0.75em; color: #ff5cb0; pointer-events: none;
                  text-shadow: 0 0 3px #000; }
  .frame-wrap { position: relative; display: inline-block; max-width: 95%; }
  .frame-wrap img { max-width: 100%; display: block; }
  select, input[type=text], input[type=number] {
    background:#222; color:#e8e8e8; border:1px solid #444; padding: 0.3em; border-radius: 3px; }
  .toolbar { margin: 0.6em 0; }
  .footer { margin-top: 1em; color: #888; font-size: 0.9em; }
  .frame-card { display: inline-block; background:#222; padding: 0.6em; margin: 0.4em;
                border-radius: 6px; vertical-align: top; }
  .frame-card img { max-width: 240px; display: block; margin-bottom: 0.4em; }
  .finger-row { display: flex; gap: 0.5em; align-items: center; margin-bottom: 0.3em;
                font-size: 0.9em; }
  .finger-row label { min-width: 4em; }
</style>
"""

_TPL_INDEX = (
    _BASE_CSS
    + """
<h1>TabVision labeling harness</h1>
<p>Three label types map to the Phase 3 + Phase 4 acceptance gates.
Save labels go under <code>tabvision/data/eval/</code> as JSON.</p>
<table>
<tr><th>clip</th><th>framing</th><th>fretboard (4 pts)</th><th>fingering (frames)</th></tr>
{% for r in rows %}
<tr>
  <td><code>{{ r.id }}</code><br><span style="color:#888;font-size:0.85em;">{{ r.path }}</span></td>
  <td class="{{ 'done' if r.framing.done else 'pending' }}">
    {{ r.framing.text }} —
    <a href="{{ url_for('framing_get', cid=r.id) }}">label</a>
  </td>
  <td class="{{ 'done' if r.fretboard.done else 'pending' }}">
    {{ r.fretboard.text }} —
    <a href="{{ url_for('fretboard_get', cid=r.id) }}">label</a>
  </td>
  <td class="{{ 'done' if r.fingering.done else 'pending' }}">
    {{ r.fingering.text }} —
    <a href="{{ url_for('fingering_get', cid=r.id) }}">label</a>
  </td>
</tr>
{% endfor %}
</table>
<p class="footer">Tip: when all three columns are green, run
<code>pytest -m "fretboard_eval or preflight_eval or hand_eval"</code> to
re-evaluate the Phase 3/4 acceptance gates against your labels.</p>
"""
)

_TPL_FRAMING = (
    _BASE_CSS
    + """
<h1>Framing — <code>{{ cid }}</code></h1>
<p>
  {% if prev_cid %}
    <a href="{{ url_for('framing_get', cid=prev_cid) }}">&larr; prev</a> &middot;
  {% endif %}
  <a href="{{ url_for('index') }}">back to index</a>
  {% if next_cid %}
    &middot; <a href="{{ url_for('framing_get', cid=next_cid) }}">next &rarr;</a>
  {% endif %}
  <br><span style="color:#888;font-size:0.85em;">{{ clip_path }}</span>
</p>
<img src="{{ url_for('clip_frame', cid=cid, frame_idx=frame_idx) }}" style="max-width: 90%;">
<div class="toolbar">
  <button onclick="setLabel('good')" id="btn-good">Good (g)</button>
  <button class="danger" onclick="setLabel('bad')" id="btn-bad">Bad (b)</button>
</div>
<div class="tagrow" id="tagrow">
  <strong>Issues (for "bad"):</strong>
  {% for t in tags %}
  <label><input type="checkbox" name="tag" value="{{ t }}"> {{ t }}</label>
  {% endfor %}
</div>
<p><textarea id="notes" rows="2" cols="60" placeholder="Notes (optional)"></textarea></p>
<button id="save" onclick="save()" disabled>Save</button>
<button id="save-next" onclick="saveNext()" disabled>Save &amp; Next &rarr; (Enter)</button>
<span id="status" class="footer"></span>

<script>
const NEXT_CID = {{ next_cid | tojson }};
let chosen = null;
const existing = {{ existing | tojson }};
if (existing) {
  chosen = existing.label;
  highlight();
  document.getElementById('notes').value = existing.notes || '';
  for (const t of (existing.tags || [])) {
    const cb = document.querySelector(`input[name=tag][value=${t}]`);
    if (cb) cb.checked = true;
  }
}

function setLabel(l) {
  chosen = l;
  highlight();
  document.getElementById('save').disabled = false;
  document.getElementById('save-next').disabled = false;
}
function highlight() {
  document.getElementById('btn-good').style.outline = chosen === 'good' ? '2px solid #6ed68a' : '';
  document.getElementById('btn-bad').style.outline = chosen === 'bad' ? '2px solid #ff7575' : '';
}
async function save() {
  const tags = Array.from(document.querySelectorAll('input[name=tag]:checked')).map(c => c.value);
  const notes = document.getElementById('notes').value;
  const r = await fetch('', { method: 'POST', headers: {'Content-Type': 'application/json'},
                              body: JSON.stringify({label: chosen, tags, notes}) });
  document.getElementById('status').textContent = r.ok ? 'saved.' : `error: ${r.status}`;
  return r.ok;
}
async function saveNext() {
  if (!chosen) return;
  if (!(await save())) return;
  window.location = NEXT_CID
    ? `/framing/${NEXT_CID}`
    : '{{ url_for("index") }}';
}

document.addEventListener('keydown', (e) => {
  const inText = ['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName);
  if (e.key === 'g' && !inText) setLabel('good');
  if (e.key === 'b' && !inText) setLabel('bad');
  if (e.key === 's' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); save(); }
  if (e.key === 'Enter' && !inText && chosen) { e.preventDefault(); saveNext(); }
});
</script>
"""
)

_TPL_FRETBOARD = (
    _BASE_CSS
    + """
<h1>Fretboard — <code>{{ cid }}</code></h1>
<p>
  {% if prev_cid %}
    <a href="{{ url_for('fretboard_get', cid=prev_cid) }}">&larr; prev</a> &middot;
  {% endif %}
  <a href="{{ url_for('index') }}">back to index</a>
  {% if next_cid %}
    &middot; <a href="{{ url_for('fretboard_get', cid=next_cid) }}">next &rarr;</a>
  {% endif %}
  <br><span style="color:#888;font-size:0.85em;">{{ clip_path }}</span>
</p>
<p>Click in this order: <strong>fret 5 top, fret 5 bottom, fret 12 top, fret 12 bottom</strong>.
"top" / "bottom" = top / bottom of the image you see on screen.</p>
<div class="toolbar">
  Frame:
  <input type="number" id="frame-idx" value="{{ frame_idx }}"
         min="0" max="{{ n_frames - 1 }}" style="width:6em;">
  <button class="ghost" onclick="loadFrame()">Reload</button>
  <button class="ghost" onclick="undo()">Undo last point</button>
  <button class="ghost" onclick="reset()">Clear all</button>
</div>
<div class="frame-wrap" id="wrap">
  <img id="img" src="" alt="" onload="onImgLoaded()">
</div>
<p>
  <textarea id="notes" rows="2" cols="60" placeholder="Notes (optional)"></textarea>
</p>
<button id="save" onclick="save()" disabled>Save (need 4 points)</button>
<button id="save-next" onclick="saveNext()" disabled>Save &amp; Next &rarr; (Enter)</button>
<span id="status" class="footer"></span>

<script>
const cid = {{ cid | tojson }};
const NEXT_CID = {{ next_cid | tojson }};
const order = [
  {fret: 5,  edge: 'top'},
  {fret: 5,  edge: 'bottom'},
  {fret: 12, edge: 'top'},
  {fret: 12, edge: 'bottom'},
];
let points = [];
let imgNaturalW = 0, imgNaturalH = 0;
const existing = {{ existing | tojson }};

function frameUrl(idx) { return `/clip/${cid}/frame/${idx}.jpg`; }

function loadFrame() {
  const idx = parseInt(document.getElementById('frame-idx').value, 10);
  document.getElementById('img').src = frameUrl(idx) + '?t=' + Date.now();
}

function onImgLoaded() {
  const img = document.getElementById('img');
  imgNaturalW = img.naturalWidth;
  imgNaturalH = img.naturalHeight;
  redraw();
}

document.getElementById('img').addEventListener('click', (e) => {
  if (points.length >= 4) return;
  const img = document.getElementById('img');
  const rect = img.getBoundingClientRect();
  const sx = imgNaturalW / rect.width;
  const sy = imgNaturalH / rect.height;
  const x = (e.clientX - rect.left) * sx;
  const y = (e.clientY - rect.top) * sy;
  const slot = order[points.length];
  points.push({fret: slot.fret, edge: slot.edge, x, y});
  redraw();
});

function undo() { if (points.length) { points.pop(); redraw(); } }
function reset() { points = []; redraw(); }

function redraw() {
  const wrap = document.getElementById('wrap');
  for (const m of wrap.querySelectorAll('.marker, .marker-label')) m.remove();
  const img = document.getElementById('img');
  const rect = img.getBoundingClientRect();
  const wrapRect = wrap.getBoundingClientRect();
  const sx = rect.width / imgNaturalW;
  const sy = rect.height / imgNaturalH;
  for (const p of points) {
    const dot = document.createElement('div'); dot.className = 'marker';
    dot.style.left = (rect.left - wrapRect.left + p.x * sx) + 'px';
    dot.style.top = (rect.top - wrapRect.top + p.y * sy) + 'px';
    wrap.appendChild(dot);
    const lbl = document.createElement('div'); lbl.className = 'marker-label';
    lbl.textContent = `f${p.fret}-${p.edge}`;
    lbl.style.left = (rect.left - wrapRect.left + p.x * sx) + 'px';
    lbl.style.top = (rect.top - wrapRect.top + p.y * sy) + 'px';
    wrap.appendChild(lbl);
  }
  document.getElementById('save').disabled = points.length !== 4;
  document.getElementById('save-next').disabled = points.length !== 4;
  document.getElementById('save').textContent = points.length === 4
    ? 'Save' : `Save (need ${4 - points.length} more)`;
}

async function save() {
  const r = await fetch('', { method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      frame_idx: parseInt(document.getElementById('frame-idx').value, 10),
      points: points,
      notes: document.getElementById('notes').value,
    })});
  document.getElementById('status').textContent = r.ok ? 'saved.' : `error: ${r.status}`;
  return r.ok;
}
async function saveNext() {
  if (points.length !== 4) return;
  if (!(await save())) return;
  window.location = NEXT_CID
    ? `/fretboard/${NEXT_CID}`
    : '{{ url_for("index") }}';
}

if (existing) {
  points = existing.points.slice();
  document.getElementById('notes').value = existing.notes || '';
  document.getElementById('frame-idx').value = existing.frame_idx;
}
loadFrame();
window.addEventListener('resize', redraw);

document.addEventListener('keydown', (e) => {
  const inText = ['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName);
  if (e.key === 'Enter' && !inText && points.length === 4) { e.preventDefault(); saveNext(); }
  if (e.key === 's' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); save(); }
});
</script>
"""
)

_TPL_FINGERING = (
    _BASE_CSS
    + """
<h1>Fingering — <code>{{ cid }}</code></h1>
<p>
  {% if prev_cid %}
    <a href="{{ url_for('fingering_get', cid=prev_cid) }}">&larr; prev</a> &middot;
  {% endif %}
  <a href="{{ url_for('index') }}">back to index</a>
  {% if next_cid %}
    &middot; <a href="{{ url_for('fingering_get', cid=next_cid) }}">next &rarr;</a>
  {% endif %}
  <br><span style="color:#888;font-size:0.85em;">{{ clip_path }}</span>
</p>
<p>For each sampled frame, label which (string, fret) each fretting finger is on.
Strings: 1 = high E, 6 = low E. Use <code>0</code> for an open string.
Set "string" to <em>—</em> for a finger that isn't fretting in this frame.</p>
<div id="grid">
{% for fi in sampled %}
  <div class="frame-card" data-frame-idx="{{ fi }}">
    <img src="{{ url_for('clip_frame', cid=cid, frame_idx=fi) }}">
    <div style="font-size: 0.85em; color:#888;">frame {{ fi }}</div>
    {% for f in fingers %}
    <div class="finger-row">
      <label>{{ f }}</label>
      <select data-finger="{{ f }}" data-axis="string">
        <option value="">—</option>
        {% for s in range(1, n_strings + 1) %}
        <option value="{{ s }}">{{ s }}</option>
        {% endfor %}
      </select>
      <select data-finger="{{ f }}" data-axis="fret">
        <option value="">—</option>
        {% for k in range(0, max_fret + 1) %}
        <option value="{{ k }}">{{ k }}</option>
        {% endfor %}
      </select>
    </div>
    {% endfor %}
  </div>
{% endfor %}
</div>
<button id="save" onclick="save()">Save all</button>
<button id="save-next" onclick="saveNext()">Save &amp; Next &rarr;</button>
<span id="status" class="footer"></span>

<script>
const existing_by_idx = {{ existing_by_idx | tojson }};
const NEXT_CID = {{ next_cid | tojson }};

function loadExisting() {
  for (const card of document.querySelectorAll('.frame-card')) {
    const idx = parseInt(card.dataset.frameIdx, 10);
    const labels = existing_by_idx[idx] || [];
    for (const fl of labels) {
      const ss = card.querySelector(`select[data-finger="${fl.finger}"][data-axis="string"]`);
      const sf = card.querySelector(`select[data-finger="${fl.finger}"][data-axis="fret"]`);
      if (ss) ss.value = fl.string === null ? "" : String(fl.string);
      if (sf) sf.value = fl.fret === null ? "" : String(fl.fret);
    }
  }
}

async function save() {
  const frames = [];
  for (const card of document.querySelectorAll('.frame-card')) {
    const idx = parseInt(card.dataset.frameIdx, 10);
    const finger_rows = [];
    for (const fr of card.querySelectorAll('.finger-row')) {
      const finger = fr.querySelector('select[data-axis="string"]').dataset.finger;
      const sval = fr.querySelector('select[data-axis="string"]').value;
      const fval = fr.querySelector('select[data-axis="fret"]').value;
      const string = sval === "" ? null : parseInt(sval, 10);
      const fret = fval === "" ? null : parseInt(fval, 10);
      finger_rows.push({finger, string, fret});
    }
    frames.push({frame_idx: idx, fingers: finger_rows});
  }
  const r = await fetch('', { method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frames}) });
  document.getElementById('status').textContent = r.ok ? 'saved.' : `error: ${r.status}`;
  return r.ok;
}
async function saveNext() {
  if (!(await save())) return;
  window.location = NEXT_CID
    ? `/fingering/${NEXT_CID}`
    : '{{ url_for("index") }}';
}

loadExisting();
</script>
"""
)


# ----- CLI -----


def discover_clips(clip_dir: Path) -> list[Path]:
    if not clip_dir.exists():
        raise SystemExit(f"clips dir not found: {clip_dir}")
    found = sorted(
        p for p in clip_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if not found:
        raise SystemExit(f"no video files found in {clip_dir} (suffixes: {VIDEO_SUFFIXES})")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--clips",
        type=Path,
        required=True,
        help="directory containing the eval clips to label",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="output JSON root (default: $TABVISION_EVAL_ROOT or "
        "tabvision/data/eval)",
    )
    parser.add_argument(
        "--fingering-frames",
        type=int,
        default=20,
        help="frames sampled per clip for fingering labels (default 20; "
        "5 clips × 20 = 100 labeled frames per the Phase 4 spec)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind host (use 0.0.0.0 to expose on the LAN)",
    )
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args(argv)

    clips = discover_clips(args.clips)
    print(f"[label_clips] found {len(clips)} clips under {args.clips}", file=sys.stderr)
    print(f"[label_clips] open http://{args.host}:{args.port}/ in a browser", file=sys.stderr)

    app = make_app(clips, args.eval_root, args.fingering_frames)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
