function formatBboxFromBounds(bounds) {
  const west = bounds.getWest().toFixed(6);
  const south = bounds.getSouth().toFixed(6);
  const east = bounds.getEast().toFixed(6);
  const north = bounds.getNorth().toFixed(6);
  return `${west}, ${south}, ${east}, ${north}`;
}

function initBboxMap() {
  const mapEl = document.getElementById("bbox-map");
  const bboxInput = document.getElementById("bbox-input");
  const bboxPreview = document.getElementById("bbox-preview");
  const clearBtn = document.getElementById("clear-bbox-btn");
  const fitWorldBtn = document.getElementById("fit-world-btn");
  const form = document.getElementById("analysis-form");

  if (!mapEl || !bboxInput || !bboxPreview || typeof L === "undefined") return;

  const map = L.map("bbox-map", { worldCopyJump: true }).setView([20, 0], 2);

  const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  });

  const esriSatellite = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { attribution: "Tiles &copy; Esri", maxZoom: 19 }
  );

  const topo = L.tileLayer(
    "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    { attribution: "&copy; OpenTopoMap contributors", maxZoom: 17 }
  );

  osm.addTo(map);

  L.control.layers(
    { "Street Map": osm, "Satellite": esriSatellite, "Topo": topo },
    {},
    { collapsed: false }
  ).addTo(map);

  const drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  function clearBboxUi() {
    drawnItems.clearLayers();
    bboxInput.value = "";
    bboxPreview.textContent = "None selected yet";
  }

  function syncBboxFromLayer(layer) {
    const bounds = layer.getBounds();
    const bbox = formatBboxFromBounds(bounds);
    bboxInput.value = bbox;
    bboxPreview.textContent = bbox;
  }

  function redrawFromInput() {
    const raw = bboxInput.value.trim();
    const parts = raw.split(",").map((p) => p.trim());
    if (parts.length !== 4) return false;

    const west = parseFloat(parts[0]);
    const south = parseFloat(parts[1]);
    const east = parseFloat(parts[2]);
    const north = parseFloat(parts[3]);
    if ([west, south, east, north].some(Number.isNaN)) return false;
    if (!(west < east && south < north)) return false;

    drawnItems.clearLayers();
    const bounds = L.latLngBounds([south, west], [north, east]);
    const rect = L.rectangle(bounds, { color: "#d6b36a", weight: 2 });
    drawnItems.addLayer(rect);
    bboxPreview.textContent = `${west.toFixed(6)}, ${south.toFixed(6)}, ${east.toFixed(6)}, ${north.toFixed(6)}`;
    map.fitBounds(bounds, { padding: [20, 20] });
    return true;
  }

  if (L.Control && L.Control.Draw && L.Draw && L.Draw.Event) {
    const drawControl = new L.Control.Draw({
      draw: {
        polygon: false,
        polyline: false,
        circle: false,
        circlemarker: false,
        marker: false,
        rectangle: { shapeOptions: { color: "#d6b36a", weight: 2 } },
      },
      edit: {
        featureGroup: drawnItems,
        edit: true,
        remove: true,
      },
    });
    map.addControl(drawControl);

    map.on(L.Draw.Event.CREATED, function (event) {
      drawnItems.clearLayers();
      const layer = event.layer;
      drawnItems.addLayer(layer);
      syncBboxFromLayer(layer);
    });

    map.on(L.Draw.Event.EDITED, function () {
      drawnItems.eachLayer(function (layer) {
        syncBboxFromLayer(layer);
      });
    });

    map.on(L.Draw.Event.DELETED, function () {
      clearBboxUi();
    });
  } else {
    console.warn("Leaflet Draw not available; rectangle toolbar disabled, manual bbox entry remains active.");
  }

  clearBtn?.addEventListener("click", function () {
    clearBboxUi();
  });

  fitWorldBtn?.addEventListener("click", function () {
    map.setView([20, 0], 2);
  });

  bboxInput.addEventListener("change", redrawFromInput);
  bboxInput.addEventListener("blur", redrawFromInput);
  bboxInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      event.preventDefault();
      redrawFromInput();
    }
  });

  if (bboxInput.value.trim()) redrawFromInput();

  form?.addEventListener("submit", function (e) {
    if (!bboxInput.value.trim()) {
      e.preventDefault();
      alert("Please draw a bounding box on the map or type one manually before launching analysis.");
    }
  });

  setTimeout(() => map.invalidateSize(), 0);
  setTimeout(() => map.invalidateSize(), 250);
}

function grayRamp(v) {
  const g = Math.round(20 + v * 220);
  return `rgb(${g}, ${g}, ${g})`;
}

function terrainTint(v) {
  if (v < 0.18) return `rgb(24, ${60 + Math.round(v * 100)}, ${110 + Math.round(v * 60)})`;
  if (v < 0.38) return `rgb(${35 + Math.round(v * 50)}, ${90 + Math.round(v * 90)}, 55)`;
  if (v < 0.62) return `rgb(${95 + Math.round(v * 70)}, ${105 + Math.round(v * 60)}, ${55 + Math.round(v * 25)})`;
  if (v < 0.82) return `rgb(${145 + Math.round(v * 55)}, ${125 + Math.round(v * 40)}, ${95 + Math.round(v * 30)})`;
  return `rgb(220, 220, 220)`;
}

function warmRelief(v) {
  return `hsl(${35 - Math.round(v * 20)}, 55%, ${15 + Math.round(v * 55)}%)`;
}

function coolRelief(v) {
  return `hsl(${210 - Math.round(v * 40)}, 45%, ${12 + Math.round(v * 58)}%)`;
}

function getLayerRgb(layerName, value) {
  const v = Math.max(0, Math.min(1, Number(value) || 0));
  const clamp = (n) => Math.max(0, Math.min(255, Math.round(n)));
  if (layerName === "terrain_texture") {
    const vv = Math.max(0, Math.min(1, (v - 0.5) * 1.22 + 0.5));
    const g = clamp(18 + vv * 226);
    return [g, g, g];
  }
  if (layerName === "hillshade") {
    const g = clamp(20 + v * 220);
    return [g, g, g];
  }
  if (layerName === "elevation") {
    if (v < 0.18) return [24, clamp(60 + v * 100), clamp(110 + v * 60)];
    if (v < 0.38) return [clamp(35 + v * 50), clamp(90 + v * 90), 55];
    if (v < 0.62) return [clamp(95 + v * 70), clamp(105 + v * 60), clamp(55 + v * 25)];
    if (v < 0.82) return [clamp(145 + v * 55), clamp(125 + v * 40), clamp(95 + v * 30)];
    return [220, 220, 220];
  }
  if (layerName === "slope") return [clamp(36 + v * 190), clamp(38 + v * 120), clamp(32 + v * 70)];
  if (layerName === "local_relief") return [clamp(20 + v * 120), clamp(44 + v * 140), clamp(78 + v * 150)];
  if (layerName === "openness") return [clamp(22 + v * 92), clamp(72 + v * 130), clamp(82 + v * 120)];
  if (layerName === "srv") return [clamp(20 + v * 150), clamp(28 + v * 160), clamp(40 + v * 185)];
  if (layerName === "archaeology") return [clamp(56 + v * 190), clamp(34 + v * 112), clamp(18 + v * 78)];
  if (layerName === "discovery") return [clamp(24 + v * 90), clamp(74 + v * 138), clamp(100 + v * 132)];
  const g = clamp(20 + v * 220);
  return [g, g, g];
}

function getLayerColor(layerName, value) {
  const [r, g, b] = getLayerRgb(layerName, value);
  return `rgb(${r}, ${g}, ${b})`;
}

function hslToRgb(h, s, l) {
  h = ((h % 360) + 360) % 360;
  s = Math.max(0, Math.min(100, s)) / 100;
  l = Math.max(0, Math.min(100, l)) / 100;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((h / 60) % 2 - 1));
  const m = l - c / 2;
  let rp = 0, gp = 0, bp = 0;
  if (h < 60) [rp, gp, bp] = [c, x, 0];
  else if (h < 120) [rp, gp, bp] = [x, c, 0];
  else if (h < 180) [rp, gp, bp] = [0, c, x];
  else if (h < 240) [rp, gp, bp] = [0, x, c];
  else if (h < 300) [rp, gp, bp] = [x, 0, c];
  else [rp, gp, bp] = [c, 0, x];
  return [Math.round((rp + m) * 255), Math.round((gp + m) * 255), Math.round((bp + m) * 255)];
}

function getLayerRgb(layerName, value) {
  const v = Math.max(0, Math.min(1, Number(value) || 0));
  if (layerName === "terrain_texture") {
    const vv = Math.max(0, Math.min(1, (v - 0.5) * 1.22 + 0.5));
    const g = Math.round(16 + vv * 232);
    return [g, g, g];
  }
  if (layerName === "hillshade") {
    const g = Math.round(18 + v * 226);
    return [g, g, g];
  }
  if (layerName === "elevation") {
    if (v < 0.18) return [24, 60 + Math.round(v * 100), 110 + Math.round(v * 60)];
    if (v < 0.38) return [35 + Math.round(v * 50), 90 + Math.round(v * 90), 55];
    if (v < 0.62) return [95 + Math.round(v * 70), 105 + Math.round(v * 60), 55 + Math.round(v * 25)];
    if (v < 0.82) return [145 + Math.round(v * 55), 125 + Math.round(v * 40), 95 + Math.round(v * 30)];
    return [220, 220, 220];
  }
  if (layerName === "slope") return hslToRgb(35 - Math.round(v * 20), 55, 15 + Math.round(v * 55));
  if (layerName === "local_relief") return hslToRgb(210 - Math.round(v * 40), 45, 12 + Math.round(v * 58));
  if (layerName === "openness") return hslToRgb(165 + Math.round(v * 35), 30, 18 + Math.round(v * 50));
  if (layerName === "srv") return hslToRgb(210 - Math.round(v * 25), 18, 10 + Math.round(v * 75));
  if (layerName === "archaeology") return hslToRgb(20 + Math.round(v * 10), 65, 12 + Math.round(v * 45));
  if (layerName === "discovery") return hslToRgb(205 - Math.round(v * 35), 55, 14 + Math.round(v * 52));
  const g = Math.round(20 + v * 220);
  return [g, g, g];
}

const FAST_RASTER_CACHE = new Map();

function getFastRasterCanvas(matrix, layerName) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const key = `${layerName}:${rows}x${cols}`;
  if (FAST_RASTER_CACHE.has(key)) return FAST_RASTER_CACHE.get(key);

  const off = document.createElement("canvas");
  off.width = cols;
  off.height = rows;
  const offCtx = off.getContext("2d", { willReadFrequently: false });
  const img = offCtx.createImageData(cols, rows);
  let p = 0;
  for (let y = 0; y < rows; y++) {
    const row = matrix[y] || [];
    for (let x = 0; x < cols; x++) {
      const [r, g, b] = getLayerRgb(layerName, row[x]);
      img.data[p++] = r;
      img.data[p++] = g;
      img.data[p++] = b;
      img.data[p++] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);
  FAST_RASTER_CACHE.set(key, off);
  return off;
}

function drawRasterLayer(ctx, canvas, matrix, layerName, zoomState) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const raster = getFastRasterCanvas(matrix, layerName);

  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#081018";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.imageSmoothingEnabled = zoomState.scale < 2.2;
  ctx.translate(zoomState.offsetX, zoomState.offsetY);
  ctx.scale(zoomState.scale, zoomState.scale);
  ctx.drawImage(raster, 0, 0, canvas.width, canvas.height);

  if ((layerName === "terrain_texture" || layerName === "hillshade" || layerName === "srv" || layerName === "local_relief") && zoomState.scale >= 2.5) {
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1 / zoomState.scale;
    const step = Math.max(8, Math.round(Math.min(rows, cols) / 56));
    for (let y = 0; y < rows; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y * cellH);
      ctx.lineTo(canvas.width, y * cellH);
      ctx.stroke();
    }
    for (let x = 0; x < cols; x += step) {
      ctx.beginPath();
      ctx.moveTo(x * cellW, 0);
      ctx.lineTo(x * cellW, canvas.height);
      ctx.stroke();
    }
  }

  ctx.restore();

  ctx.fillStyle = "rgba(238,243,249,0.94)";
  ctx.font = "600 14px Arial";
  ctx.fillText(`Raster review · ${layerName.replaceAll("_", " ")} · ${cols} x ${rows}`, 18, 24);
}

async function initRunViewer() {
  const canvas = document.getElementById("terrain-layer-canvas");
  if (!canvas) return;

  const runId = canvas.dataset.runId;
  const layerNameEl = document.getElementById("viewer-layer-name");
  const layerDescEl = document.getElementById("viewer-layer-description");
  const rasterPanel = document.getElementById("raster-view-panel");
  const threePanel = document.getElementById("three-view-panel");
  const layerBtns = Array.from(document.querySelectorAll(".viewer-layer-btn"));
  const modeBtns = Array.from(document.querySelectorAll(".viewer-mode-btn"));

  const response = await fetch(`/api/runs/${runId}/layers`);
  if (!response.ok) return;

  const payload = await response.json();
  const layers = payload.layers || {};
  const legend = payload.legend || {};
  let currentLayer = payload.default_layer || (layers.terrain_texture ? "terrain_texture" : "hillshade");
  let currentMode = payload.default_mode || "raster";

  const layerDisplayNames = {
    terrain_texture: "Terrain Texture Composite",
    elevation: "Elevation Hypsometry",
    hillshade: "Analytical Hillshade",
    slope: "Slope Intensity",
    local_relief: "Local Relief Model",
    openness: "Positive Openness",
    srv: "Sky View Factor",
    archaeology: "Archaeology Signal",
    discovery: "Discovery Priority"
  };

  const ctx = canvas.getContext("2d");
  const zoomState = { scale: 1, offsetX: 0, offsetY: 0 };
  let dragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let originX = 0;
  let originY = 0;

  function updateLegend() {
    if (layerNameEl) layerNameEl.textContent = layerDisplayNames[currentLayer] || currentLayer.replaceAll("_", " ");
    if (layerDescEl) layerDescEl.textContent = legend[currentLayer] || "";
  }

  function syncButtons() {
    layerBtns.forEach((btn) => {
      const active = btn.dataset.layer === currentLayer;
      btn.classList.toggle("active", active);
      btn.classList.toggle("btn-primary", active);
      btn.classList.toggle("btn-secondary", !active);
    });

    modeBtns.forEach((btn) => {
      const active = btn.dataset.mode === currentMode;
      btn.classList.toggle("active", active);
      btn.classList.toggle("btn-primary", active);
      btn.classList.toggle("btn-secondary", !active);
    });

    rasterPanel?.classList.toggle("active", currentMode === "raster");
    threePanel?.classList.toggle("active", currentMode === "three");
  }

  function renderRaster() {
    const matrix = layers[currentLayer];
    if (!matrix) return;
    updateLegend();
    syncButtons();
    drawRasterLayer(ctx, canvas, matrix, currentLayer, zoomState);
  }

  canvas.addEventListener("mousedown", (e) => {
    if (currentMode !== "raster") return;
    dragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    originX = zoomState.offsetX;
    originY = zoomState.offsetY;
  });

  window.addEventListener("mouseup", () => {
    dragging = false;
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging || currentMode !== "raster") return;
    zoomState.offsetX = originX + (e.clientX - dragStartX);
    zoomState.offsetY = originY + (e.clientY - dragStartY);
    renderRaster();
  });

  canvas.addEventListener("wheel", (e) => {
    if (currentMode !== "raster") return;
    e.preventDefault();
    const delta = e.deltaY < 0 ? 1.12 : 0.9;
    zoomState.scale = Math.max(1, Math.min(8, zoomState.scale * delta));
    renderRaster();
  }, { passive: false });

  layerBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      currentLayer = btn.dataset.layer;
      renderRaster();
      window.dispatchEvent(new CustomEvent("monahinga-layer-change", { detail: { layer: currentLayer } }));
    });
  });

  modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      currentMode = btn.dataset.mode;
      syncButtons();
      if (currentMode === "raster") renderRaster();
      window.dispatchEvent(new CustomEvent("monahinga-mode-change", { detail: { mode: currentMode } }));
    });
  });

  renderRaster();
}

async function initThreeTerrain() {
  const container = document.getElementById("three-terrain-container");
  if (!container || typeof THREE === "undefined") return;

  const runId = container.dataset.runId;
  const resetBtn = document.getElementById("three-reset-btn");

  const response = await fetch(`/api/runs/${runId}/layers`);
  if (!response.ok) return;

  const payload = await response.json();
  const layers = payload.layers || {};
  let currentLayer = "elevation";
  let currentMatrix = layers.elevation || payload.heightmap_3d || layers.hillshade;

  if (!currentMatrix) return;

  const rows = currentMatrix.length;
  const cols = currentMatrix[0].length;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x081018);

  const camera = new THREE.PerspectiveCamera(
    45,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
  );
  camera.position.set(0, 18, 24);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = false;
  controls.target.set(0, 0, 0);
  controls.update();

  const ambient = new THREE.AmbientLight(0xffffff, 1.1);
  scene.add(ambient);

  const directional = new THREE.DirectionalLight(0xffffff, 1.35);
  directional.position.set(12, 18, 10);
  scene.add(directional);

  const geometry = new THREE.PlaneGeometry(22, 22, cols - 1, rows - 1);
  geometry.rotateX(-Math.PI / 2);

  const position = geometry.attributes.position;
  const colorArray = new Float32Array(position.count * 3);

  function cssToThreeColor(css) {
    const c = new THREE.Color();
    c.setStyle(css);
    return c;
  }

  function applyLayer(layerName) {
    currentLayer = layerName;
    currentMatrix = layers[layerName] || layers.elevation || payload.heightmap_3d || layers.hillshade;

    let colorIndex = 0;
    for (let z = 0; z < rows; z++) {
      for (let x = 0; x < cols; x++) {
        const i = z * cols + x;
        const h = currentMatrix[z][x];
        position.setY(i, h * 4.8);

        const c = cssToThreeColor(getLayerColor(layerName, h));
        colorArray[colorIndex++] = c.r;
        colorArray[colorIndex++] = c.g;
        colorArray[colorIndex++] = c.b;
      }
    }

    geometry.setAttribute("color", new THREE.BufferAttribute(colorArray, 3));
    position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
    geometry.computeVertexNormals();
  }

  applyLayer(currentLayer);

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.95,
    metalness: 0.02,
    flatShading: false,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  const wire = new THREE.LineSegments(
    new THREE.WireframeGeometry(geometry),
    new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.07,
    })
  );
  scene.add(wire);

  function rebuildWireframe() {
    wire.geometry.dispose();
    wire.geometry = new THREE.WireframeGeometry(geometry);
  }

  function resetView() {
    camera.position.set(0, 18, 24);
    controls.target.set(0, 0, 0);
    controls.update();
  }

  resetBtn?.addEventListener("click", resetView);

  window.addEventListener("monahinga-layer-change", (event) => {
    applyLayer(event.detail.layer);
    rebuildWireframe();
  });

  function onResize() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  window.addEventListener("resize", onResize);

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  animate();
}


async function initRunPins() {
  const mapEl = document.getElementById("run-pin-map");
  if (!mapEl || typeof L === "undefined") return;

  const runId = mapEl.dataset.runId;
  const pinLat = document.getElementById("pin-lat");
  const pinLon = document.getElementById("pin-lon");
  const rawBbox = (mapEl.dataset.bbox || "").split(",").map((p) => parseFloat(p.trim()));
  const hasBbox = rawBbox.length === 4 && rawBbox.every((n) => !Number.isNaN(n));

  const map = L.map(mapEl).setView([20, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap",
  }).addTo(map);

  const markers = L.layerGroup().addTo(map);

  if (hasBbox) {
    const bounds = L.latLngBounds([rawBbox[1], rawBbox[0]], [rawBbox[3], rawBbox[2]]);
    L.rectangle(bounds, { color: "#d6b36a", weight: 2, fillOpacity: 0.08 }).addTo(map);
    map.fitBounds(bounds, { padding: [20, 20] });
  }

  function renderPins(data) {
    markers.clearLayers();
    const features = data.features || [];
    features.forEach((feature) => {
      const coords = feature.geometry?.coordinates || [];
      const props = feature.properties || {};
      if (coords.length < 2) return;
      const marker = L.marker([coords[1], coords[0]]);
      marker.bindPopup(`<strong>${props.label || "Pin"}</strong><br>${props.pin_type || "note"}<br>${props.notes || ""}`);
      markers.addLayer(marker);
    });
  }

  try {
    const response = await fetch(`/api/runs/${runId}/pins`, { cache: "no-store" });
    if (response.ok) {
      renderPins(await response.json());
    }
  } catch (error) {
    console.error(error);
  }

  map.on("click", function (e) {
    if (pinLat) pinLat.value = e.latlng.lat.toFixed(6);
    if (pinLon) pinLon.value = e.latlng.lng.toFixed(6);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  initBboxMap();
  initRunViewer().catch(console.error);
  initRunPins().catch(console.error);
});

/* === Site Insight startup map calm patch === */
(function () {
  function calmStartupMaps() {
    if (!window.L) return;

    var mapEls = Array.prototype.slice.call(document.querySelectorAll(".leaflet-container"));
    mapEls.forEach(function (el) {
      if (el.querySelector(".map-loading-note")) return;
      var note = document.createElement("div");
      note.className = "map-loading-note";
      note.textContent = "Loading map tiles...";
      el.appendChild(note);

      window.setTimeout(function () {
        note.classList.add("is-hidden");
      }, 2400);
    });

    // Leaflet can paint gray tile gaps if the map initializes before its container finishes layout.
    // This delayed resize nudge is intentionally lightweight and frontend-only.
    window.setTimeout(function () {
      window.dispatchEvent(new Event("resize"));
      document.querySelectorAll(".leaflet-container").forEach(function (el) {
        el.style.willChange = "transform";
        window.setTimeout(function () {
          el.style.willChange = "";
        }, 900);
      });
    }, 450);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", calmStartupMaps);
  } else {
    calmStartupMaps();
  }

  window.addEventListener("load", function () {
    window.setTimeout(calmStartupMaps, 250);
  });
})();

