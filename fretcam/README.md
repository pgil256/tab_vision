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
.venv\Scripts\fretcam
```

Open <http://127.0.0.1:8765>, select **Start camera**, and grant camera
permission. A rear/environment camera is preferred when the browser exposes
one.

## Verify

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m fretcam.benchmark --rounds 100
```

The benchmark starts a temporary loopback server and round-trips an in-memory
synthetic JPEG. It does not access a camera or write image data to disk.
