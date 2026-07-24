const video = document.querySelector("#camera");
const canvas = document.querySelector("#preview");
const context = canvas.getContext("2d");
const inferenceCanvas = document.querySelector("#inference");
const inferenceContext = inferenceCanvas.getContext("2d", { alpha: false });
const viewer = document.querySelector(".viewer");
const toggle = document.querySelector("#toggle");
const emptyState = document.querySelector("#empty-state");
const connection = document.querySelector("#connection");
const statusLine = document.querySelector("#status");
const fpsLabel = document.querySelector("#fps");
const latencyLabel = document.querySelector("#latency");
const serverLatencyLabel = document.querySelector("#server-latency");
const frameSizeLabel = document.querySelector("#frame-size");
const fretboardReadout = document.querySelector("#fretboard-readout");
const fretboardStatus = document.querySelector("#fretboard-status");
const positionReadout = document.querySelector("#position-readout");
const positionStatus = document.querySelector("#position-status");
const positionDetail = document.querySelector("#position-detail");
const cameraSelect = document.querySelector("#camera-select");
const handednessSelect = document.querySelector("#player-handedness");
const mirrorPreview = document.querySelector("#mirror-preview");
const calibrateButton = document.querySelector("#calibrate");
const calibrateTwoPointButton = document.querySelector("#calibrate-two-point");
const continueCalibrationButton = document.querySelector("#continue-calibration");
const calibrationUpperPosition = document.querySelector("#calibration-upper-position");
const resetCalibrationButton = document.querySelector("#reset-calibration");
const exportDiagnosticsButton = document.querySelector("#export-diagnostics");
const lockReason = document.querySelector("#lock-reason");
const calibrationStatus = document.querySelector("#calibration-status");
const factorBoard = document.querySelector("#factor-board");
const factorFreshness = document.querySelector("#factor-freshness");
const factorStability = document.querySelector("#factor-stability");
const factorLandmarks = document.querySelector("#factor-landmarks");
const factorOnNeck = document.querySelector("#factor-on-neck");
const factorAgreement = document.querySelector("#factor-agreement");
const factorTemporal = document.querySelector("#factor-temporal");
const traceToggleButton = document.querySelector("#trace-toggle");
const traceCancelButton = document.querySelector("#trace-cancel");
const failureBufferToggle = document.querySelector("#failure-buffer");
const failurePosition = document.querySelector("#failure-position");
const failureFingers = document.querySelector("#failure-fingers");
const failureNote = document.querySelector("#failure-note");
const markFailureButton = document.querySelector("#mark-failure");
const captureStatus = document.querySelector("#capture-status");

const INFERENCE_PRESETS = [
  { maxWidth: 640, maxHeight: 480, quality: 0.72 },
  { maxWidth: 576, maxHeight: 432, quality: 0.68 },
  { maxWidth: 512, maxHeight: 384, quality: 0.64 },
  { maxWidth: 448, maxHeight: 336, quality: 0.60 },
];
const MAX_DIAGNOSTIC_SAMPLES = 300;

let stream = null;
let socket = null;
let animationId = null;
let frameInFlight = false;
let encodeInProgress = false;
let sentAt = 0;
let completedFrames = 0;
let fpsWindowStarted = performance.now();
let presetIndex = 0;
let highLatencySamples = 0;
let lowLatencySamples = 0;
let intentionalClose = false;
let lastHud = null;
let diagnosticSamples = [];
let sessionStartedAt = performance.now();
let sessionGeneration = 0;
let liveControlsEnabled = false;
let traceActive = false;
let failureBufferActive = false;
let captureFrameSequence = 0;
let traceStartPending = false;
let failureTogglePrevious = null;

function setConnection(label, state) {
  connection.textContent = label;
  connection.dataset.state = state;
}

function setLiveControls(enabled) {
  liveControlsEnabled = enabled;
  const inferenceControlsEnabled = enabled && !traceActive;
  handednessSelect.disabled = traceActive;
  calibrateButton.disabled = !inferenceControlsEnabled;
  calibrateTwoPointButton.disabled = !inferenceControlsEnabled;
  continueCalibrationButton.disabled = true;
  calibrationUpperPosition.disabled = !inferenceControlsEnabled;
  resetCalibrationButton.disabled = !inferenceControlsEnabled;
  exportDiagnosticsButton.disabled = !enabled;
  setCaptureControls(enabled);
}

function setCaptureControls(enabled) {
  traceToggleButton.disabled = !enabled;
  traceCancelButton.disabled = !enabled || !traceActive;
  failureBufferToggle.disabled = !enabled;
  const failureReady = enabled && failureBufferActive;
  failurePosition.disabled = !failureReady;
  failureFingers.disabled = !failureReady;
  failureNote.disabled = !failureReady;
  markFailureButton.disabled = !failureReady;
  traceToggleButton.textContent = traceActive
    ? "Save exact comparison trace"
    : "Start exact comparison trace";
}

function updateCalibrationControls(calibration) {
  if (!calibration) return;
  calibrationStatus.textContent = calibration.message;
  const collecting = calibration.status === "collecting";
  const isTwoPoint = calibration.mode === "two_point";
  calibrateButton.textContent = collecting && !isTwoPoint
    ? `Hold Position I... (${calibration.samples})`
    : "Calibrate Position I";
  const upper = Number.parseInt(calibrationUpperPosition.value, 10) || 5;
  const activeTarget = calibration.target_position || 1;
  calibrateTwoPointButton.textContent = collecting && isTwoPoint
    ? `Hold Position ${activeTarget}... (${calibration.samples})`
    : `Calibrate I + ${upper === 9 ? "IX" : "V"}`;
  continueCalibrationButton.disabled = (
    !liveControlsEnabled || traceActive || calibration.status !== "awaiting_second"
  );
  calibrationUpperPosition.disabled = (
    !liveControlsEnabled
    || traceActive
    || (isTwoPoint && ["collecting", "awaiting_second"].includes(calibration.status))
  );
}

function updateFps(now) {
  const elapsed = now - fpsWindowStarted;
  if (elapsed >= 1000) {
    fpsLabel.textContent = `${((completedFrames * 1000) / elapsed).toFixed(1)}`;
    completedFrames = 0;
    fpsWindowStarted = now;
  }
}

function fitInferenceSize(sourceWidth, sourceHeight, preset = INFERENCE_PRESETS[presetIndex]) {
  const scale = Math.min(1, preset.maxWidth / sourceWidth, preset.maxHeight / sourceHeight);
  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

function syncCanvasSize() {
  const sourceWidth = video.videoWidth || 640;
  const sourceHeight = video.videoHeight || 480;
  const size = fitInferenceSize(sourceWidth, sourceHeight);
  if (inferenceCanvas.width !== size.width || inferenceCanvas.height !== size.height) {
    inferenceCanvas.width = size.width;
    inferenceCanvas.height = size.height;
  }
  viewer.style.aspectRatio = `${sourceWidth} / ${sourceHeight}`;
  return size;
}

function captureInferenceFrame() {
  const size = syncCanvasSize();
  inferenceContext.drawImage(video, 0, 0, size.width, size.height);
  return size;
}

function nextFrame() {
  animationId = requestAnimationFrame(nextFrame);
  updateFps(performance.now());
  if (
    !stream ||
    !socket ||
    socket.readyState !== WebSocket.OPEN ||
    frameInFlight ||
    encodeInProgress ||
    video.readyState < 2
  ) {
    return;
  }

  const inferenceSize = captureInferenceFrame();
  encodeInProgress = true;
  const encodeGeneration = sessionGeneration;
  const encodeSocket = socket;
  const preset = INFERENCE_PRESETS[presetIndex];
  inferenceCanvas.toBlob((blob) => {
    if (sessionGeneration !== encodeGeneration || socket !== encodeSocket) return;
    encodeInProgress = false;
    if (
      !blob ||
      !encodeSocket ||
      encodeSocket.readyState !== WebSocket.OPEN ||
      frameInFlight
    ) return;
    frameInFlight = true;
    sentAt = performance.now();
    frameSizeLabel.textContent = `${(blob.size / 1024).toFixed(1)} KB`;
    if (traceActive || failureBufferActive) {
      captureFrameSequence += 1;
      encodeSocket.send(blob);
      encodeSocket.send(JSON.stringify({
        type: "frame_context",
        sequence: captureFrameSequence,
        session_offset_ms: sentAt - sessionStartedAt,
        source_width: video.videoWidth || inferenceSize.width,
        source_height: video.videoHeight || inferenceSize.height,
        inference_width: inferenceSize.width,
        inference_height: inferenceSize.height,
        jpeg_quality: preset.quality,
        payload_bytes: blob.size,
      }));
    } else {
      encodeSocket.send(blob);
    }
  }, "image/jpeg", preset.quality);
}

function geometryPoint(raw, scaleX, scaleY) {
  const scaledX = raw[0] * scaleX;
  return [mirrorPreview.checked ? canvas.width - scaledX : scaledX, raw[1] * scaleY];
}

function strokeLine(start, end, scaleX, scaleY, colour, width) {
  const [x1, y1] = geometryPoint(start, scaleX, scaleY);
  const [x2, y2] = geometryPoint(end, scaleX, scaleY);
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.strokeStyle = colour;
  context.lineWidth = width;
  context.stroke();
}

function updateLiveReadouts(detection, position) {
  const geometryStatus = detection.geometry_status || "missing";
  const fretboardLocked = detection.neck_locked && detection.neck_quad.length === 4;
  if (!fretboardLocked) {
    fretboardReadout.dataset.state = "active";
    fretboardStatus.textContent = "Searching...";
  } else if (geometryStatus === "stale") {
    fretboardReadout.dataset.state = "active";
    fretboardStatus.textContent = "Reacquiring...";
  } else {
    fretboardReadout.dataset.state = "locked";
    fretboardStatus.textContent = geometryStatus === "tracked" ? "Tracked" : "Locked";
  }

  positionStatus.textContent = position.label.replace("…", "...");
  if (position.state === "locked") {
    positionReadout.dataset.state = "locked";
    positionDetail.textContent = "Multi-finger position evidence is locked.";
  } else if (position.state === "holding") {
    positionReadout.dataset.state = "active";
    positionDetail.textContent = "Hand briefly hidden; holding the last position.";
  } else {
    positionReadout.dataset.state = "active";
    positionDetail.textContent = "Keep several fingertips and the full neck visible.";
  }
}

function resetLiveReadouts() {
  fretboardReadout.dataset.state = "idle";
  fretboardStatus.textContent = "Waiting for camera";
  positionReadout.dataset.state = "idle";
  positionStatus.textContent = "—";
  positionDetail.textContent = "Hold your fretting hand on the neck.";
}

function drawHud(payload) {
  const sourceWidth = payload.frame.width;
  const sourceHeight = payload.frame.height;
  if (canvas.width !== sourceWidth || canvas.height !== sourceHeight) {
    canvas.width = sourceWidth;
    canvas.height = sourceHeight;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width / sourceWidth;
  const scaleY = canvas.height / sourceHeight;
  const detection = payload.detection;
  const position = payload.position;
  const stale = detection.geometry_status === "stale";
  const held = detection.geometry_status === "held";

  if (detection.neck_quad.length === 4) {
    context.save();
    context.beginPath();
    detection.neck_quad.forEach((raw, index) => {
      const [x, y] = geometryPoint(raw, scaleX, scaleY);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.closePath();
    context.fillStyle = stale ? "rgba(255, 184, 86, 0.08)" : "rgba(87, 238, 132, 0.10)";
    context.fill();
    context.strokeStyle = stale ? "#ffb856" : held ? "#b9d36a" : "#59ff88";
    context.lineWidth = 5;
    context.lineJoin = "round";
    context.setLineDash(stale ? [12, 8] : held ? [4, 5] : []);
    context.shadowColor = stale ? "transparent" : "rgba(36, 255, 105, 0.85)";
    context.shadowBlur = stale ? 0 : 10;
    context.stroke();
    context.restore();
  }

  detection.fret_ticks.forEach((tick) => {
    strokeLine(tick.start, tick.end, scaleX, scaleY, "rgba(255,255,255,0.42)", 1);
  });

  detection.hand_points.forEach((handPoint) => {
    const isIndex = handPoint.name === "index";
    const [x, y] = geometryPoint([handPoint.x, handPoint.y], scaleX, scaleY);
    context.beginPath();
    context.arc(x, y, isIndex ? 8 : 4, 0, Math.PI * 2);
    context.fillStyle = isIndex ? "#ffd65a" : "rgba(255, 214, 90, 0.68)";
    context.fill();
    if (isIndex) {
      context.strokeStyle = "#16130a";
      context.lineWidth = 2;
      context.stroke();
    }
  });

  const panelWidth = Math.min(320, canvas.width - 24);
  context.fillStyle = "rgba(8, 12, 9, 0.82)";
  context.fillRect(12, 12, panelWidth, 88);
  context.fillStyle = position.state === "locked" ? "#9af5ad" : "#ffd65a";
  context.font = "700 22px system-ui, sans-serif";
  context.fillText(position.label.replace("…", "..."), 25, 43);
  context.fillStyle = "#aeb7af";
  context.font = "12px ui-monospace, monospace";
  context.fillText(`window {${position.window_frets.join(",")}}`, 25, 64);
  context.fillStyle = "#29352c";
  context.fillRect(25, 78, panelWidth - 26, 8);
  context.fillStyle = position.confidence >= 0.5 ? "#69ee8e" : "#ffd65a";
  context.fillRect(25, 78, (panelWidth - 26) * position.confidence, 8);

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
  updateLiveReadouts(detection, position);
  updateDiagnostics(payload);
}

function percent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—";
}

function updateDiagnostics(payload) {
  const factors = payload.detection.confidence_factors || {};
  const blockers = Array.isArray(factors.blockers) ? factors.blockers : [];
  lockReason.textContent = payload.position.state === "locked"
    ? "Locked: no active blocker."
    : `${payload.guidance.message}${blockers.length ? ` (${blockers.join(", ")})` : ""}`;
  factorBoard.textContent = percent(factors.board);
  factorFreshness.textContent = percent(factors.freshness);
  factorStability.textContent = percent(factors.stability);
  factorLandmarks.textContent = percent(factors.landmark_quality);
  factorOnNeck.textContent = percent(factors.on_neck);
  factorAgreement.textContent = percent(factors.finger_agreement);
  factorTemporal.textContent = percent(payload.position.temporal_agreement);
  updateCalibrationControls(payload.calibration);
}

function adaptPerformance(serverMs, e2eMs) {
  if (serverMs > 90 || e2eMs > 150) {
    highLatencySamples += 1;
    lowLatencySamples = 0;
  } else if (serverMs < 55 && e2eMs < 90) {
    lowLatencySamples += 1;
    highLatencySamples = 0;
  } else {
    highLatencySamples = 0;
    lowLatencySamples = 0;
  }
  if (highLatencySamples >= 3 && presetIndex < INFERENCE_PRESETS.length - 1) {
    presetIndex += 1;
    highLatencySamples = 0;
    syncCanvasSize();
  } else if (lowLatencySamples >= 30 && presetIndex > 0) {
    presetIndex -= 1;
    lowLatencySamples = 0;
    syncCanvasSize();
  }
}

function recordDiagnostics(payload, e2eMs) {
  const detection = payload.detection;
  const position = payload.position;
  const stage = detection.stage_latency || {};
  const factors = detection.confidence_factors || {};
  const preset = INFERENCE_PRESETS[presetIndex];
  diagnosticSamples.push({
    t_ms: Math.round(performance.now() - sessionStartedAt),
    e2e_ms: Number(e2eMs.toFixed(3)),
    server_ms: payload.server_ms,
    payload_kb: Number.parseFloat(frameSizeLabel.textContent) || 0,
    inference: {
      width: payload.frame.width,
      height: payload.frame.height,
      jpeg_quality: preset.quality,
    },
    stage_ms: {
      detector: stage.detector_ms || 0,
      homography: stage.homography_ms || 0,
      hand: stage.hand_ms || 0,
      anchor: stage.anchor_ms || 0,
      total: stage.total_ms || 0,
    },
    geometry: {
      status: detection.geometry_status,
      age_ms: detection.geometry_age_ms,
      detector_age_ms: detection.detector_age_ms,
      confidence: detection.homography_confidence,
      stability: detection.geometry_stability,
      fret_refinement_support: detection.fret_refinement_support || 0,
      string_refinement_support: detection.string_refinement_support || 0,
      distortion_residual: detection.geometry_distortion_residual || 0,
      nut_x: detection.nut_x,
      body_joint_x: detection.body_joint_x,
      boundary_support: detection.boundary_support || 0,
      body_joint_fret: detection.body_joint_fret,
    },
    position: {
      state: position.state,
      value: position.position,
      confidence: position.confidence,
      temporal_agreement: position.temporal_agreement,
      reason: position.reason,
      hand_source: detection.hand_source || "none",
      hand_search_source: detection.hand_search_source || "none",
      hand_schedule_mode: detection.hand_schedule_mode || "default",
      hand_detector_interval_ms: detection.hand_detector_interval_ms || 0,
      hand_detector_calls: detection.hand_detector_calls || 0,
      hand_search_attempts: detection.hand_search_attempts || [],
      hand_refresh_reason: detection.hand_refresh_reason || "unknown",
      detector_result_consumed: Boolean(detection.detector_result_consumed),
      detector_result_accepted: Boolean(detection.detector_result_accepted),
      hand_pose_quality: detection.hand_pose_quality || 0,
      hand_pose_continuity: detection.hand_pose_continuity || 0,
      hand_pose_predicted: Boolean(detection.hand_pose_predicted),
    },
    confidence: {
      board: factors.board || 0,
      freshness: factors.freshness || 0,
      stability: factors.stability || 0,
      landmark_quality: factors.landmark_quality || 0,
      on_neck: factors.on_neck || 0,
      finger_agreement: factors.finger_agreement || 0,
      coarse_agreement: factors.coarse_agreement || 0,
      support_sufficiency: factors.support_sufficiency || 0,
      combined: factors.combined || 0,
      blockers: Array.isArray(factors.blockers) ? [...factors.blockers] : [],
    },
    guidance_code: payload.guidance.code,
    calibration: {
      status: payload.calibration?.status || "idle",
      offset_fret: payload.calibration?.offset_fret || 0,
      scale: payload.calibration?.scale || 1,
      mode: payload.calibration?.mode || "single",
      anchors: payload.calibration?.anchors || [],
    },
  });
  if (diagnosticSamples.length > MAX_DIAGNOSTIC_SAMPLES) diagnosticSamples.shift();
}

function handleCaptureControl(payload) {
  const capture = payload.capture;
  if (capture) {
    traceActive = Boolean(capture.trace_enabled);
    failureBufferActive = Boolean(capture.failure_enabled);
    traceStartPending = false;
    failureTogglePrevious = null;
    failureBufferToggle.checked = failureBufferActive;
    setLiveControls(liveControlsEnabled);
  }
  if (payload.status === "trace_started") {
    captureStatus.textContent = "Exact packets are buffered in memory. Select Save to write this trace locally.";
  } else if (payload.status === "trace_saved") {
    captureStatus.textContent = `Comparison trace saved locally: ${payload.package_id}`;
  } else if (payload.status === "trace_cancelled") {
    captureStatus.textContent = "Trace cancelled. Its in-memory frames were discarded.";
  } else if (payload.status === "failure_buffer_enabled") {
    captureStatus.textContent = "A rolling 2-second window is in memory. Nothing has been saved.";
  } else if (payload.status === "failure_buffer_disabled") {
    captureStatus.textContent = "Failure buffer disabled. Its in-memory frames were discarded.";
  } else if (payload.status === "failure_saved") {
    captureStatus.textContent = `Marked failure saved locally: ${payload.package_id}`;
  }
}

function handleSocketMessage(rawPayload) {
  let payload;
  try {
    payload = JSON.parse(rawPayload);
  } catch (error) {
    frameInFlight = false;
    fail(`Invalid HUD response: ${error.message}`);
    return;
  }
  if (payload.type === "error") {
    if (payload.scope === "capture") {
      if (payload.capture) {
        handleCaptureControl(payload);
      } else {
        if (traceStartPending) traceActive = false;
        traceStartPending = false;
        if (failureTogglePrevious !== null) {
          failureBufferActive = failureTogglePrevious;
          failureBufferToggle.checked = failureBufferActive;
        }
        failureTogglePrevious = null;
        setLiveControls(liveControlsEnabled);
      }
      captureStatus.textContent = `Local accuracy tool: ${payload.message || "request rejected"}`;
      return;
    }
    frameInFlight = false;
    fail(payload.message || "Frame processing failed.");
    return;
  }
  if (payload.type === "control") {
    updateCalibrationControls(payload.calibration);
    handleCaptureControl(payload);
    return;
  }
  if (payload.type !== "hud") return;

  const e2eMs = performance.now() - sentAt;
  if (payload.capture_warning) {
    traceActive = false;
    failureBufferActive = false;
    traceStartPending = false;
    failureTogglePrevious = null;
    failureBufferToggle.checked = false;
    setLiveControls(liveControlsEnabled);
    captureStatus.textContent = `Local capture stopped: ${payload.capture_warning}`;
  }
  adaptPerformance(payload.server_ms, e2eMs);
  lastHud = payload;
  drawHud(payload);
  latencyLabel.textContent = `${e2eMs.toFixed(1)} ms`;
  serverLatencyLabel.textContent = `${payload.server_ms.toFixed(1)} ms`;
  completedFrames += 1;
  frameInFlight = false;
  recordDiagnostics(payload, e2eMs);
}

function sendControl(payload) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
}

function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${location.host}/ws`);
  socket = ws;
  setConnection("connecting", "idle");
  ws.addEventListener("open", () => {
    if (socket !== ws) return;
    setConnection("HUD live", "live");
    setLiveControls(true);
    sendControl({
      type: "settings",
      player_handedness: handednessSelect.value,
    });
  });
  ws.addEventListener("message", (event) => {
    if (socket !== ws) return;
    handleSocketMessage(event.data);
  });
  ws.addEventListener("close", () => {
    if (socket !== ws) return;
    socket = null;
    frameInFlight = false;
    traceActive = false;
    failureBufferActive = false;
    traceStartPending = false;
    failureTogglePrevious = null;
    failureBufferToggle.checked = false;
    captureStatus.textContent = "Local capture stopped; unsaved frames were discarded.";
    setLiveControls(false);
    if (stream && !intentionalClose) fail("WebSocket closed. Stop and restart the camera to reconnect.");
    else setConnection("idle", "idle");
  });
  ws.addEventListener("error", () => {
    if (socket !== ws) return;
    fail("Could not connect to the local FretCam server.");
  });
}

function fail(message) {
  statusLine.textContent = message;
  setConnection("error", "error");
  frameInFlight = false;
  encodeInProgress = false;
}

async function populateCameras(preferredDeviceId = "") {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");
  cameraSelect.replaceChildren();
  if (cameras.length === 0) {
    cameraSelect.append(new Option("No camera found", ""));
    cameraSelect.disabled = true;
    return;
  }
  cameras.forEach((camera, index) => {
    cameraSelect.append(new Option(camera.label || `Camera ${index + 1}`, camera.deviceId));
  });
  const available = cameras.some((camera) => camera.deviceId === preferredDeviceId);
  cameraSelect.value = available ? preferredDeviceId : cameras[0].deviceId;
  cameraSelect.disabled = false;
}

async function start() {
  const startGeneration = ++sessionGeneration;
  try {
    intentionalClose = false;
    const selectedDevice = cameraSelect.value;
    const videoConstraints = selectedDevice
      ? { deviceId: { exact: selectedDevice }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: "environment" } };
    const acquiredStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: videoConstraints,
    });
    if (sessionGeneration !== startGeneration) {
      acquiredStream.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = acquiredStream;
    video.srcObject = stream;
    await video.play();
    const activeDevice = stream.getVideoTracks()[0]?.getSettings().deviceId || "";
    await populateCameras(activeDevice);
    if (sessionGeneration !== startGeneration) return;
    syncCanvasSize();
    emptyState.hidden = true;
    toggle.textContent = "Stop camera";
    statusLine.textContent = "Starting the local vision chain; the first frame may take a few seconds.";
    fretboardReadout.dataset.state = "active";
    fretboardStatus.textContent = "Searching...";
    positionReadout.dataset.state = "active";
    positionStatus.textContent = "Acquiring...";
    positionDetail.textContent = "Keep the full neck and fretting hand visible.";
    diagnosticSamples = [];
    traceActive = false;
    failureBufferActive = false;
    failureBufferToggle.checked = false;
    captureFrameSequence = 0;
    captureStatus.textContent = "Nothing is being retained or saved.";
    sessionStartedAt = performance.now();
    connectSocket();
    nextFrame();
  } catch (error) {
    if (sessionGeneration !== startGeneration) return;
    stop();
    fail(`Camera unavailable: ${error.message}`);
  }
}

function stop() {
  sessionGeneration += 1;
  if (animationId !== null) cancelAnimationFrame(animationId);
  animationId = null;
  intentionalClose = true;
  const closingSocket = socket;
  socket = null;
  if (closingSocket) closingSocket.close();
  const oldStream = stream;
  stream = null;
  if (oldStream) oldStream.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
  frameInFlight = false;
  encodeInProgress = false;
  emptyState.hidden = false;
  toggle.textContent = "Start camera";
  statusLine.textContent = "Ready. Use a rear/environment camera when available.";
  fpsLabel.textContent = "—";
  latencyLabel.textContent = "—";
  serverLatencyLabel.textContent = "—";
  frameSizeLabel.textContent = "—";
  context.clearRect(0, 0, canvas.width, canvas.height);
  resetLiveReadouts();
  traceActive = false;
  failureBufferActive = false;
  traceStartPending = false;
  failureTogglePrevious = null;
  failureBufferToggle.checked = false;
  captureFrameSequence = 0;
  captureStatus.textContent = "Nothing is being retained or saved.";
  setLiveControls(false);
  setConnection("idle", "idle");
  lastHud = null;
  lockReason.textContent = "Start the camera to inspect lock evidence.";
}

async function restartCamera() {
  if (!stream) return;
  cameraSelect.disabled = true;
  stop();
  await start();
}

function exportDiagnostics() {
  const payload = {
    schema: "fretcam-diagnostics-v1",
    app_version: 2,
    preferences: {
      player_handedness: handednessSelect.value,
      mirror_preview: mirrorPreview.checked,
    },
    sample_limit: MAX_DIAGNOSTIC_SAMPLES,
    samples: diagnosticSamples,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "fretcam-diagnostics.json";
  link.click();
  URL.revokeObjectURL(url);
}

function savePreferences() {
  try {
    localStorage.setItem("fretcam-player-handedness", handednessSelect.value);
    localStorage.setItem("fretcam-mirror-preview", mirrorPreview.checked ? "1" : "0");
  } catch {
    // Preferences are optional; private browsing may disable localStorage.
  }
}

function loadPreferences() {
  try {
    handednessSelect.value = localStorage.getItem("fretcam-player-handedness") || "right";
    mirrorPreview.checked = localStorage.getItem("fretcam-mirror-preview") === "1";
  } catch {
    handednessSelect.value = "right";
    mirrorPreview.checked = false;
  }
  video.classList.toggle("mirrored", mirrorPreview.checked);
}

toggle.addEventListener("click", () => (stream ? stop() : start()));
cameraSelect.addEventListener("change", restartCamera);
handednessSelect.addEventListener("change", () => {
  savePreferences();
  sendControl({ type: "settings", player_handedness: handednessSelect.value });
});
mirrorPreview.addEventListener("change", () => {
  video.classList.toggle("mirrored", mirrorPreview.checked);
  savePreferences();
  if (lastHud) drawHud(lastHud);
});
calibrateButton.addEventListener("click", () => sendControl({ type: "calibrate" }));
calibrateTwoPointButton.addEventListener("click", () => sendControl({
  type: "calibrate_two_point",
  upper_position: Number.parseInt(calibrationUpperPosition.value, 10) || 5,
}));
continueCalibrationButton.addEventListener("click", () => sendControl({
  type: "continue_calibration",
}));
calibrationUpperPosition.addEventListener("change", () => {
  const upper = Number.parseInt(calibrationUpperPosition.value, 10) || 5;
  calibrateTwoPointButton.textContent = `Calibrate I + ${upper === 9 ? "IX" : "V"}`;
});
resetCalibrationButton.addEventListener("click", () => sendControl({ type: "reset_calibration" }));
exportDiagnosticsButton.addEventListener("click", exportDiagnostics);
traceToggleButton.addEventListener("click", () => {
  if (traceActive) {
    sendControl({ type: "trace_save", confirm_save: true });
    return;
  }
  traceActive = true;
  traceStartPending = true;
  setLiveControls(liveControlsEnabled);
  captureStatus.textContent = "Starting a clean in-memory comparison trace...";
  sendControl({ type: "trace_start" });
});
traceCancelButton.addEventListener("click", () => sendControl({ type: "trace_cancel" }));
failureBufferToggle.addEventListener("change", () => {
  failureTogglePrevious = failureBufferActive;
  failureBufferActive = failureBufferToggle.checked;
  setCaptureControls(liveControlsEnabled);
  sendControl({
    type: "failure_buffer",
    enabled: failureBufferToggle.checked,
  });
});
markFailureButton.addEventListener("click", () => {
  const selectedPosition = Number.parseInt(failurePosition.value, 10);
  const pressingFingers = [...failureFingers.querySelectorAll("input:checked")]
    .map((input) => input.value);
  sendControl({
    type: "failure_mark",
    confirm_save: true,
    expectation: {
      position: Number.isFinite(selectedPosition) ? selectedPosition : "unknown",
      pressing_fingers: pressingFingers,
      note: failureNote.value,
    },
  });
});
video.addEventListener("resize", syncCanvasSize);
navigator.mediaDevices?.addEventListener?.("devicechange", () => {
  const current = cameraSelect.value;
  window.setTimeout(() => populateCameras(current).catch(() => {}), 250);
});

loadPreferences();
if (navigator.mediaDevices?.enumerateDevices) {
  populateCameras().catch(() => {
    cameraSelect.disabled = false;
  });
}
