# FretCam

FretCam is a quarantined local prototype for TabVision's live fretboard and
playing-position HUD. F1 is intentionally an echo-only transport scaffold:
the browser captures JPEG frames, sends one frame at a time over a localhost
WebSocket, and displays the echoed frame with measured FPS and round-trip
latency.

Nothing is recorded or persisted by the server.

## Run

```powershell
cd fretcam
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pip install --no-deps -e ..\tabvision
.venv\Scripts\fretcam
```

The second editable install exposes TabVision's existing vision modules as a
library without installing its unrelated audio/render extras. FretCam's own
package declares the pre-approved vision dependencies it uses.

Open <http://127.0.0.1:8765>, select **Start camera**, and grant camera
permission. A rear/environment camera is preferred when the browser exposes
one.

## Verify

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m fretcam.benchmark --rounds 100
.venv\Scripts\python -m fretcam.replay_gaps
.venv\Scripts\python -m fretcam.replay_position
```

The benchmark starts a temporary loopback server and round-trips an in-memory
synthetic JPEG. It does not access a camera or write image data to disk.

The GAPS replay samples three public cached MP4s at 640 px, runs the F2 chain,
and prints a JSON gate report with neck/anchor outcomes and per-stage latency.
It reads from `~/.tabvision/cache/gaps_video/` and writes nothing.

The position replay samples six seconds of public `031_vpswc`, runs F2b + F3,
and writes a machine-local diagnostic MP4 and still under
`~/.tabvision/cache/fretcam_artifacts/`. These reproducible artifacts are not
committed and are not position-accuracy evidence.
