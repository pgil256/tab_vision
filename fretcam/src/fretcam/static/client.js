const video = document.querySelector("#camera");
const canvas = document.querySelector("#preview");
const context = canvas.getContext("2d", { alpha: false });
const toggle = document.querySelector("#toggle");
const emptyState = document.querySelector("#empty-state");
const connection = document.querySelector("#connection");
const statusLine = document.querySelector("#status");
const fpsLabel = document.querySelector("#fps");
const latencyLabel = document.querySelector("#latency");
const frameSizeLabel = document.querySelector("#frame-size");

let stream = null;
let socket = null;
let animationId = null;
let frameInFlight = false;
let sentAt = 0;
let completedFrames = 0;
let fpsWindowStarted = performance.now();

function setConnection(label, state) {
  connection.textContent = label;
  connection.dataset.state = state;
}

function updateFps(now) {
  const elapsed = now - fpsWindowStarted;
  if (elapsed >= 1000) {
    fpsLabel.textContent = `${((completedFrames * 1000) / elapsed).toFixed(1)}`;
    completedFrames = 0;
    fpsWindowStarted = now;
  }
}

function nextFrame() {
  animationId = requestAnimationFrame(nextFrame);
  updateFps(performance.now());
  if (!stream || !socket || socket.readyState !== WebSocket.OPEN || frameInFlight || video.readyState < 2) {
    return;
  }

  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    if (!blob || !socket || socket.readyState !== WebSocket.OPEN || frameInFlight) return;
    frameInFlight = true;
    sentAt = performance.now();
    frameSizeLabel.textContent = `${(blob.size / 1024).toFixed(1)} KB`;
    socket.send(blob);
  }, "image/jpeg", 0.72);
}

async function renderEcho(payload) {
  const blob = payload instanceof Blob ? payload : new Blob([payload], { type: "image/jpeg" });
  const bitmap = await createImageBitmap(blob);
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  latencyLabel.textContent = `${(performance.now() - sentAt).toFixed(1)} ms`;
  completedFrames += 1;
  frameInFlight = false;
}

function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.binaryType = "blob";
  setConnection("connecting", "idle");
  socket.addEventListener("open", () => setConnection("echo live", "live"));
  socket.addEventListener("message", (event) => {
    renderEcho(event.data).catch((error) => fail(`Could not draw echoed frame: ${error.message}`));
  });
  socket.addEventListener("close", () => {
    frameInFlight = false;
    if (stream) fail("WebSocket closed. Stop and restart the camera to reconnect.");
    else setConnection("idle", "idle");
  });
  socket.addEventListener("error", () => fail("Could not connect to the local FretCam server."));
}

function fail(message) {
  statusLine.textContent = message;
  setConnection("error", "error");
  frameInFlight = false;
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: { ideal: "environment" } },
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    emptyState.hidden = true;
    toggle.textContent = "Stop camera";
    statusLine.textContent = "Echo mode: one JPEG at a time, held only in memory.";
    connectSocket();
    nextFrame();
  } catch (error) {
    stop();
    fail(`Camera unavailable: ${error.message}`);
  }
}

function stop() {
  if (animationId !== null) cancelAnimationFrame(animationId);
  animationId = null;
  if (socket) socket.close();
  socket = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  frameInFlight = false;
  emptyState.hidden = false;
  toggle.textContent = "Start camera";
  statusLine.textContent = "Ready. Use a rear/environment camera when available.";
  fpsLabel.textContent = "—";
  latencyLabel.textContent = "—";
  frameSizeLabel.textContent = "—";
  setConnection("idle", "idle");
}

toggle.addEventListener("click", () => (stream ? stop() : start()));
