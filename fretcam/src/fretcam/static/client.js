const video = document.querySelector("#camera");
const canvas = document.querySelector("#preview");
const context = canvas.getContext("2d", { alpha: false });
const toggle = document.querySelector("#toggle");
const emptyState = document.querySelector("#empty-state");
const connection = document.querySelector("#connection");
const statusLine = document.querySelector("#status");
const fpsLabel = document.querySelector("#fps");
const latencyLabel = document.querySelector("#latency");
const serverLatencyLabel = document.querySelector("#server-latency");
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

function point(point, scaleX, scaleY) {
  return [point[0] * scaleX, point[1] * scaleY];
}

function strokeLine(start, end, scaleX, scaleY, colour, width) {
  const [x1, y1] = point(start, scaleX, scaleY);
  const [x2, y2] = point(end, scaleX, scaleY);
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.strokeStyle = colour;
  context.lineWidth = width;
  context.stroke();
}

function drawHud(payload) {
  const sourceWidth = payload.frame.width;
  const sourceHeight = payload.frame.height;
  const scaleX = canvas.width / sourceWidth;
  const scaleY = canvas.height / sourceHeight;
  const detection = payload.detection;
  const position = payload.position;

  if (detection.neck_quad.length === 4) {
    context.beginPath();
    detection.neck_quad.forEach((raw, index) => {
      const [x, y] = point(raw, scaleX, scaleY);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.closePath();
    context.fillStyle = "rgba(87, 238, 132, 0.08)";
    context.fill();
    context.strokeStyle = "#69ee8e";
    context.lineWidth = 3;
    context.stroke();
  }

  detection.fret_ticks.forEach((tick) => {
    strokeLine(tick.start, tick.end, scaleX, scaleY, "rgba(255,255,255,0.42)", 1);
  });

  detection.hand_points.forEach((handPoint) => {
    const isIndex = handPoint.name === "index";
    context.beginPath();
    context.arc(handPoint.x * scaleX, handPoint.y * scaleY, isIndex ? 8 : 4, 0, Math.PI * 2);
    context.fillStyle = isIndex ? "#ffd65a" : "rgba(255, 214, 90, 0.68)";
    context.fill();
    if (isIndex) {
      context.strokeStyle = "#16130a";
      context.lineWidth = 2;
      context.stroke();
    }
  });

  const panelWidth = Math.min(300, canvas.width - 24);
  context.fillStyle = "rgba(8, 12, 9, 0.82)";
  context.fillRect(12, 12, panelWidth, 83);
  context.fillStyle = position.state === "locked" ? "#9af5ad" : "#ffd65a";
  context.font = "700 22px system-ui, sans-serif";
  context.fillText(position.label.replace("…", "..."), 25, 43);
  context.fillStyle = "#aeb7af";
  context.font = "12px ui-monospace, monospace";
  context.fillText(`window {${position.window_frets.join(",")}}`, 25, 63);
  context.fillStyle = "#29352c";
  context.fillRect(25, 75, panelWidth - 26, 8);
  context.fillStyle = position.confidence >= 0.5 ? "#69ee8e" : "#ffd65a";
  context.fillRect(25, 75, (panelWidth - 26) * position.confidence, 8);

  const guide = payload.guidance;
  const guideWidth = Math.min(canvas.width - 24, Math.max(330, guide.message.length * 7.5));
  const guideY = canvas.height - 54;
  context.fillStyle = "rgba(8, 12, 9, 0.84)";
  context.fillRect(12, guideY, guideWidth, 42);
  context.fillStyle = guide.level === "warning" ? "#ffc76b" : guide.level === "good" ? "#9af5ad" : "#dce7dd";
  context.font = "600 14px system-ui, sans-serif";
  context.fillText(guide.message, 24, guideY + 26, guideWidth - 24);
  statusLine.textContent = guide.message;
  statusLine.dataset.level = guide.level;
}

function renderHud(rawPayload) {
  let payload;
  try {
    payload = JSON.parse(rawPayload);
  } catch (error) {
    fail(`Invalid HUD response: ${error.message}`);
    return;
  }
  if (payload.type === "error") {
    fail(payload.message || "Frame processing failed.");
    return;
  }
  drawHud(payload);
  latencyLabel.textContent = `${(performance.now() - sentAt).toFixed(1)} ms`;
  serverLatencyLabel.textContent = `${payload.server_ms.toFixed(1)} ms`;
  completedFrames += 1;
  frameInFlight = false;
}

function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  setConnection("connecting", "idle");
  socket.addEventListener("open", () => setConnection("HUD live", "live"));
  socket.addEventListener("message", (event) => renderHud(event.data));
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
    statusLine.textContent = "Starting the local vision chain; the first frame may take a few seconds.";
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
  serverLatencyLabel.textContent = "—";
  frameSizeLabel.textContent = "—";
  setConnection("idle", "idle");
}

toggle.addEventListener("click", () => (stream ? stop() : start()));
