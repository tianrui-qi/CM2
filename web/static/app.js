const STORAGE_KEY = "cm2_web_roi_state_v1";
const DEFAULT_ROI_COLORS = [
  "#f2559c",
  "#41a85f",
  "#4c9ee3",
  "#c6802b",
  "#7b5fd2",
  "#d64c3b",
];
const UNASSIGNED_POINT_COLOR = "rgba(248,248,248,0.62)";
const UNASSIGNED_LINE_COLOR = "rgba(32,27,22,0.28)";
const TRACE_SOURCE_ORDER = ["c", "c_plus_yra"];
const TRACE_SOURCE_UI_LABELS = {
  c: "C",
  c_plus_yra: "C + YrA",
};

const state = {
  meta: null,
  points: null,
  tracesBySource: {},
  rois: [],
  activeRoiId: null,
  activeSignalSource: "c",
  mapPlotReady: false,
  mapViewportKey: null,
};

function setStatus(message, isError = false) {
  const el = document.getElementById("status-banner");
  const hasMessage = Boolean(message);
  el.textContent = message ?? "";
  el.classList.toggle("hidden", !hasMessage);
  el.classList.toggle("error", isError);
}

function quantizedFloat(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : "nan";
}

function buildNeuronHoverText(index) {
  const p = state.points;
  const metrics = p.metrics;
  return [
    `Neuron ${p.id[index]}`,
    `Patch ${p.patch_name[index]} / component ${p.component_index[index]}`,
    `x=${p.x[index]}, y=${p.y[index]}`,
    `snr=${quantizedFloat(metrics.snr[index])}`,
    `r=${quantizedFloat(metrics.r_value[index])}`,
    `g0=${quantizedFloat(metrics.g_0[index])}`,
    `g1=${quantizedFloat(metrics.g_1[index])}`,
    `t_peak=${quantizedFloat(metrics.t_peak[index], 1)} ms`,
    `t_1/2=${quantizedFloat(metrics.t_half[index], 1)} ms`,
  ].join("<br>");
}

function makeRoi(name = null, color = null, neuronIds = []) {
  const roiIndex = state.rois.length;
  return {
    id: `roi-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name: name ?? `ROI ${roiIndex + 1}`,
    color: color ?? DEFAULT_ROI_COLORS[roiIndex % DEFAULT_ROI_COLORS.length],
    neuronIds: [...new Set(neuronIds)],
  };
}

function getRoiById(roiId) {
  return state.rois.find((roi) => roi.id === roiId) ?? null;
}

function findAssignedRoiId(neuronId) {
  for (const roi of state.rois) {
    if (roi.neuronIds.includes(neuronId)) {
      return roi.id;
    }
  }
  return null;
}

function removeNeuronFromAllRois(neuronId) {
  for (const roi of state.rois) {
    roi.neuronIds = roi.neuronIds.filter((id) => id !== neuronId);
  }
}

function saveUiState() {
  const payload = {
    rois: state.rois,
    activeRoiId: state.activeRoiId,
    activeSignalSource: state.activeSignalSource,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function loadUiState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return false;
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed.rois) || parsed.rois.length === 0) {
      return false;
    }
    const validNeuronIds = new Set(state.points.id);
    const seenNeurons = new Set();
    state.rois = parsed.rois.map((roi, idx) => ({
      id: typeof roi.id === "string" ? roi.id : makeRoi().id,
      name: typeof roi.name === "string" && roi.name.trim() ? roi.name : `ROI ${idx + 1}`,
      color: typeof roi.color === "string" ? roi.color : DEFAULT_ROI_COLORS[idx % DEFAULT_ROI_COLORS.length],
      neuronIds: Array.isArray(roi.neuronIds)
        ? roi.neuronIds.filter((id) => {
            const keep = validNeuronIds.has(id) && !seenNeurons.has(id);
            if (keep) {
              seenNeurons.add(id);
            }
            return keep;
          })
        : [],
    }));
    state.activeRoiId = getRoiById(parsed.activeRoiId)?.id ?? state.rois[0].id;
    const storedSource = typeof parsed.activeSignalSource === "string"
      ? parsed.activeSignalSource
      : parsed.activeTraceSource;
    if (
      typeof storedSource === "string"
      && TRACE_SOURCE_ORDER.includes(storedSource)
    ) {
      state.activeSignalSource = storedSource;
    }
    return true;
  } catch (error) {
    console.error(error);
    return false;
  }
}

function getAvailableTraceSourceKeys() {
  if (!state.meta) {
    return [];
  }
  return TRACE_SOURCE_ORDER.filter((sourceKey) => Boolean(state.meta.trace_sources?.[sourceKey]));
}

function ensureValidActiveTraceSource() {
  const available = getAvailableTraceSourceKeys();
  if (!available.length) {
    return;
  }
  if (!available.includes(state.activeSignalSource)) {
    state.activeSignalSource = available[0];
  }
}

function ensureAtLeastOneRoi() {
  if (state.rois.length === 0) {
    const roi = makeRoi();
    state.rois = [roi];
    state.activeRoiId = roi.id;
  }
}

function buildMapMarkerColors() {
  const fill = [];
  const line = [];
  const size = [];
  for (const neuronId of state.points.id) {
    const roiId = findAssignedRoiId(neuronId);
    const roi = roiId ? getRoiById(roiId) : null;
    fill.push(roi ? roi.color : UNASSIGNED_POINT_COLOR);
    line.push(roi ? "rgba(20,16,12,0.75)" : UNASSIGNED_LINE_COLOR);
    size.push(roi ? 9 : 6);
  }
  return { fill, line, size };
}

function computeMapHeight() {
  return window.innerHeight;
}

function computeMapCoverRanges() {
  const fullWidth = Number(state.meta.full_width);
  const fullHeight = Number(state.meta.full_height);
  const viewportWidth = Math.max(window.innerWidth, 1);
  const viewportHeight = Math.max(computeMapHeight(), 1);
  const viewportAspect = viewportWidth / viewportHeight;
  const imageAspect = fullWidth / fullHeight;

  let viewWidth;
  let viewHeight;
  if (viewportAspect >= imageAspect) {
    viewWidth = fullWidth;
    viewHeight = fullWidth / viewportAspect;
  } else {
    viewHeight = fullHeight;
    viewWidth = fullHeight * viewportAspect;
  }

  const centerX = fullWidth / 2;
  const centerY = fullHeight / 2;
  return {
    xRange: [centerX - viewWidth / 2, centerX + viewWidth / 2],
    yRange: [centerY + viewHeight / 2, centerY - viewHeight / 2],
  };
}

function renderRoiList() {
  const container = document.getElementById("roi-list");
  container.innerHTML = "";
  for (const roi of state.rois) {
    const item = document.createElement("div");
    item.className = `roi-item${roi.id === state.activeRoiId ? " active" : ""}`;
    item.style.background = roi.color;
    item.title = `${roi.name} • ${roi.neuronIds.length} neurons • double-click to change color`;

    const face = document.createElement("div");
    face.className = "roi-card-face";

    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = roi.color;
    colorInput.className = "roi-color-input";
    colorInput.addEventListener("input", () => {
      roi.color = colorInput.value;
      item.style.background = roi.color;
      saveUiState();
      renderMap();
      renderRoiList();
      updatePlots();
    });
    item.addEventListener("dblclick", (event) => {
      if (event.target instanceof HTMLButtonElement) {
        return;
      }
      colorInput.click();
    });
    item.addEventListener("click", (event) => {
      if (event.target instanceof HTMLButtonElement) {
        return;
      }
      colorInput.click();
    });

    const controls = document.createElement("div");
    controls.className = "roi-controls";

    const selectBtn = document.createElement("button");
    selectBtn.className = `roi-chip-btn${roi.id === state.activeRoiId ? " is-active" : ""}`;
    selectBtn.textContent = "✓";
    selectBtn.title = roi.id === state.activeRoiId ? "Deactivate ROI" : "Activate ROI";
    selectBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      state.activeRoiId = roi.id === state.activeRoiId ? null : roi.id;
      saveUiState();
      renderRoiList();
    });

    const clearBtn = document.createElement("button");
    clearBtn.className = "roi-chip-btn";
    clearBtn.textContent = "⌫";
    clearBtn.title = "Clear ROI neurons";
    clearBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      roi.neuronIds = [];
      saveUiState();
      renderMap();
      renderRoiList();
      updatePlots();
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "roi-chip-btn";
    deleteBtn.textContent = "✕";
    deleteBtn.title = "Delete ROI";
    deleteBtn.disabled = state.rois.length <= 1;
    deleteBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      state.rois = state.rois.filter((r) => r.id !== roi.id);
      ensureAtLeastOneRoi();
      if (!getRoiById(state.activeRoiId)) {
        state.activeRoiId = state.rois[0].id;
      }
      saveUiState();
      renderMap();
      renderRoiList();
      updatePlots();
    });

    controls.appendChild(selectBtn);
    controls.appendChild(clearBtn);
    controls.appendChild(deleteBtn);

    item.appendChild(face);
    item.appendChild(controls);
    item.appendChild(colorInput);
    container.appendChild(item);
  }
}

function buildMapLayout() {
  const background = state.meta.backgrounds.find((bg) => bg.key === state.meta.default_background_key) ?? state.meta.backgrounds[0];
  const { xRange, yRange } = computeMapCoverRanges();
  return {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    height: computeMapHeight(),
    autosize: true,
    xaxis: {
      range: xRange,
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      fixedrange: false,
    },
    yaxis: {
      range: yRange,
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      fixedrange: false,
      scaleanchor: "x",
      scaleratio: 1,
    },
    images: [
      {
        source: `/cache/${background.file}`,
        xref: "x",
        yref: "y",
        x: 0,
        y: state.meta.full_height,
        sizex: state.meta.full_width,
        sizey: state.meta.full_height,
        sizing: "stretch",
        yanchor: "bottom",
        layer: "below",
        opacity: 1,
      },
    ],
    paper_bgcolor: "#000",
    plot_bgcolor: "#000",
    dragmode: "pan",
    hovermode: "closest",
    showlegend: false,
    uirevision: "map-persist",
  };
}

function renderMap() {
  const colors = buildMapMarkerColors();
  const trace = {
    type: "scatter",
    mode: "markers",
    x: state.points.x,
    y: state.points.y,
    text: state.points.id.map((_, idx) => buildNeuronHoverText(idx)),
    hovertemplate: "%{text}<extra></extra>",
    marker: {
      color: colors.fill,
      size: colors.size,
      line: {
        color: colors.line,
        width: 0.8,
      },
      opacity: 1,
    },
  };
  const layout = buildMapLayout();
  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d"],
    displaylogo: false,
    scrollZoom: true,
  };
  const plotDiv = document.getElementById("map-plot");
  state.mapViewportKey = `${window.innerWidth}x${window.innerHeight}`;
  Plotly.react(plotDiv, [trace], layout, config).then(() => {
    if (!state.mapPlotReady) {
      plotDiv.on("plotly_click", (event) => {
        const pointIndex = event?.points?.[0]?.pointIndex;
        if (typeof pointIndex === "number") {
          handleNeuronToggle(pointIndex);
        }
      });
      state.mapPlotReady = true;
    }
  });
}

function renderSourceToggle(containerId, activeSourceKey, onSelect) {
  const container = document.getElementById(containerId);
  if (!container) {
    return;
  }
  container.innerHTML = "";
  const available = getAvailableTraceSourceKeys();
  if (available.length <= 1) {
    container.style.display = "none";
    return;
  }
  container.style.display = "inline-flex";
  for (const sourceKey of available) {
    const button = document.createElement("button");
    button.className = `trace-source-btn${sourceKey === activeSourceKey ? " active" : ""}`;
    button.innerHTML = TRACE_SOURCE_UI_LABELS[sourceKey] ?? sourceKey;
    button.addEventListener("click", () => {
      onSelect(sourceKey);
    });
    container.appendChild(button);
  }
}

function handleNeuronToggle(pointIndex) {
  const neuronId = state.points.id[pointIndex];
  const activeRoi = getRoiById(state.activeRoiId);
  if (!activeRoi) {
    return;
  }
  const currentRoiId = findAssignedRoiId(neuronId);
  if (currentRoiId === activeRoi.id) {
    activeRoi.neuronIds = activeRoi.neuronIds.filter((id) => id !== neuronId);
  } else {
    removeNeuronFromAllRois(neuronId);
    activeRoi.neuronIds = [...activeRoi.neuronIds, neuronId];
  }
  saveUiState();
  renderMap();
  renderRoiList();
  updatePlots();
}

function getTraceStats(sourceKey) {
  return state.points.trace_stats[sourceKey];
}

function getTraceSlice(sourceKey, neuronId) {
  const traceBuffer = state.tracesBySource[sourceKey];
  const offset = neuronId * state.meta.trace_length;
  return traceBuffer.subarray(offset, offset + state.meta.trace_length);
}

function getSortedNeuronIds(roi) {
  return [...roi.neuronIds].sort((a, b) => {
    const dx = state.points.x[a] - state.points.x[b];
    if (dx !== 0) {
      return dx;
    }
    return state.points.y[a] - state.points.y[b];
  });
}

function buildTracePlotData(sourceKey) {
  const frameRate = state.meta.frame_rate_hz;
  const nFrames = state.meta.trace_length;
  const traceStats = getTraceStats(sourceKey);
  const time = Array.from({ length: nFrames }, (_, idx) => idx / frameRate);
  const traces = [];
  const shapes = [];
  let rowCursor = 0;
  const rowGap = 0.9;
  const rowStep = 1.08;

  for (const roi of state.rois) {
    const neuronIds = getSortedNeuronIds(roi);
    if (neuronIds.length === 0) {
      continue;
    }
    const x = [];
    const y = [];
    const groupStart = rowCursor;
    neuronIds.forEach((neuronId, localIdx) => {
      const trace = getTraceSlice(sourceKey, neuronId);
      const p05 = Number.isFinite(traceStats.p05[neuronId]) ? traceStats.p05[neuronId] : 0;
      const p95 = Number.isFinite(traceStats.p95[neuronId]) ? traceStats.p95[neuronId] : 1;
      const scale = Math.max(p95 - p05, 1e-6);
      const baseline = -(groupStart + localIdx * rowStep);
      for (let t = 0; t < nFrames; t += 1) {
        x.push(time[t]);
        y.push(baseline + (trace[t] - p05) / scale);
      }
      x.push(NaN);
      y.push(NaN);
    });

    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y,
      line: { color: roi.color, width: 1 },
      hoverinfo: "skip",
      showlegend: false,
    });

    const groupEnd = groupStart + (neuronIds.length - 1) * rowStep + 1;
    shapes.push({
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: -(groupEnd + rowGap / 2),
      y1: -(groupEnd + rowGap / 2),
      line: { color: "rgba(80,70,60,0.18)", width: 1 },
    });
    rowCursor = groupEnd + rowGap + rowStep;
  }

  const height = Math.max(320, Math.min(900, rowCursor * 15 + 80));
  return { traces, shapes, height };
}

function renderTracePlot(plotId, sourceKey) {
  const plotDiv = document.getElementById(plotId);
  const { traces, shapes, height } = buildTracePlotData(sourceKey);
  if (traces.length === 0) {
    Plotly.react(
      plotDiv,
      [],
      {
        margin: { l: 18, r: 8, t: 12, b: 12 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        annotations: [{
          x: 0.5,
          y: 0.5,
          xref: "paper",
          yref: "paper",
          text: "Select neurons on the map to render traces.",
          showarrow: false,
          font: { size: 14, color: "#6f665c" },
        }],
        xaxis: { visible: false },
        yaxis: { visible: false },
        height: 320,
      },
      { responsive: true, displaylogo: false }
    );
    return;
  }

  Plotly.react(
    plotDiv,
    traces,
    {
      margin: { l: 18, r: 8, t: 8, b: 8 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        visible: false,
      },
      yaxis: {
        visible: false,
      },
      shapes,
      height,
      showlegend: false,
      hovermode: false,
    },
    { responsive: true, displaylogo: false }
  );
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function buildHeatmapData(sourceKey) {
  const frameRate = state.meta.frame_rate_hz;
  const nFrames = state.meta.trace_length;
  const traceStats = getTraceStats(sourceKey);
  const x = Array.from({ length: nFrames }, (_, idx) => idx / frameRate);
  const z = [];
  const shapes = [];
  let rowCursor = 0;

  for (const roi of state.rois) {
    const neuronIds = getSortedNeuronIds(roi);
    if (neuronIds.length === 0) {
      continue;
    }
    const startRow = rowCursor;
    for (const neuronId of neuronIds) {
      const trace = getTraceSlice(sourceKey, neuronId);
      const mean = Number.isFinite(traceStats.mean[neuronId]) ? traceStats.mean[neuronId] : 0;
      const stdValue = Number.isFinite(traceStats.std[neuronId]) ? traceStats.std[neuronId] : 1;
      const std = Math.max(stdValue, 1e-6);
      z.push(Array.from(trace, (value) => clamp((value - mean) / std, -2.5, 4.5)));
      rowCursor += 1;
    }
    const endRow = rowCursor - 1;
    shapes.push({
      type: "rect",
      xref: "paper",
      x0: -0.045,
      x1: -0.015,
      yref: "y",
      y0: startRow - 0.5,
      y1: endRow + 0.5,
      fillcolor: roi.color,
      line: { width: 0 },
    });
    shapes.push({
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: endRow + 0.5,
      y1: endRow + 0.5,
      line: { color: "rgba(255,255,255,0.65)", width: 1.2 },
    });
  }

  const height = Math.max(320, Math.min(900, z.length * 10 + 80));
  return { x, z, shapes, height };
}

function renderHeatmapPlot(plotId, sourceKey) {
  const plotDiv = document.getElementById(plotId);
  const { x, z, shapes, height } = buildHeatmapData(sourceKey);
  if (z.length === 0) {
    Plotly.react(
      plotDiv,
      [],
      {
        margin: { l: 18, r: 8, t: 12, b: 12 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        annotations: [{
          x: 0.5,
          y: 0.5,
          xref: "paper",
          yref: "paper",
          text: "Heatmap appears here once an ROI contains neurons.",
          showarrow: false,
          font: { size: 14, color: "#6f665c" },
        }],
        xaxis: { visible: false },
        yaxis: { visible: false },
        height: 320,
      },
      { responsive: true, displaylogo: false }
    );
    return;
  }

  Plotly.react(
    plotDiv,
    [{
      type: "heatmap",
      x,
      z,
      colorscale: "Viridis",
      zmin: -2.5,
      zmax: 4.5,
      showscale: false,
      hovertemplate: "Time %{x:.2f}s<br>z=%{z:.2f}<extra></extra>",
    }],
    {
      margin: { l: 18, r: 8, t: 8, b: 8 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        visible: false,
      },
      yaxis: {
        visible: false,
        autorange: "reversed",
      },
      shapes,
      height,
    },
    { responsive: true, displaylogo: false }
  );
}

function updatePlots() {
  ensureValidActiveTraceSource();
  renderSourceToggle("shared-source-toggle", state.activeSignalSource, (sourceKey) => {
    state.activeSignalSource = sourceKey;
    saveUiState();
    updatePlots();
  });

  if (state.meta.trace_sources[state.activeSignalSource]) {
    renderTracePlot("c-trace-plot", state.activeSignalSource);
    renderHeatmapPlot("c-heatmap-plot", state.activeSignalSource);
  }
}

function exportPlot(plotId, filename) {
  const node = document.getElementById(plotId);
  Plotly.downloadImage(node, {
    format: "svg",
    filename,
    width: Math.max(node.clientWidth, 1200),
    height: Math.max(node.clientHeight, 600),
    scale: 1,
  });
}

function wireButtons() {
  document.getElementById("add-roi-btn").addEventListener("click", () => {
    const roi = makeRoi();
    state.rois.push(roi);
    state.activeRoiId = roi.id;
    saveUiState();
    renderRoiList();
    updatePlots();
  });

  document.getElementById("download-c-traces-btn").addEventListener("click", () => {
    exportPlot("c-trace-plot", `roi_traces_${state.activeSignalSource}`);
  });
  document.getElementById("download-c-heatmap-btn").addEventListener("click", () => {
    exportPlot("c-heatmap-plot", `roi_heatmap_${state.activeSignalSource}`);
  });

  window.addEventListener("resize", () => {
    if (!state.meta) {
      return;
    }
    const nextKey = `${window.innerWidth}x${window.innerHeight}`;
    if (state.mapViewportKey !== nextKey) {
      renderMap();
    }
  });
}

async function loadCache() {
  const [metaResponse, pointsResponse] = await Promise.all([
    fetch("/cache/metadata.json"),
    fetch("/cache/points.json"),
  ]);
  if (!metaResponse.ok || !pointsResponse.ok) {
    throw new Error("Failed to load web cache files.");
  }
  const meta = await metaResponse.json();
  const points = await pointsResponse.json();
  const expected = meta.neuron_count * meta.trace_length;
  const traceSources = meta.trace_sources ?? {};
  const traceEntries = TRACE_SOURCE_ORDER
    .filter((sourceKey) => Boolean(traceSources[sourceKey]))
    .map((sourceKey) => [sourceKey, traceSources[sourceKey]]);
  const tracesBySource = {};
  const tracePayloads = await Promise.all(traceEntries.map(async ([sourceKey, traceSpec]) => {
    const traceResponse = await fetch(`/cache/${traceSpec.file}`);
    if (!traceResponse.ok) {
      throw new Error(`Failed to load trace cache for ${sourceKey}.`);
    }
    const traceBuffer = await traceResponse.arrayBuffer();
    const traces = new Float32Array(traceBuffer);
    if (traces.length !== expected) {
      throw new Error(`Trace cache size mismatch for ${sourceKey}: got ${traces.length}, expected ${expected}`);
    }
    return [sourceKey, traces];
  }));
  for (const [sourceKey, traces] of tracePayloads) {
    tracesBySource[sourceKey] = traces;
  }
  return { meta, points, tracesBySource };
}

async function init() {
  try {
    setStatus("Loading cache…");
    const { meta, points, tracesBySource } = await loadCache();
    state.meta = meta;
    state.points = points;
    state.tracesBySource = tracesBySource;
    if (!loadUiState()) {
      const roi = makeRoi();
      state.rois = [roi];
      state.activeRoiId = roi.id;
    }
    ensureValidActiveTraceSource();
    ensureAtLeastOneRoi();
    wireButtons();
    renderRoiList();
    renderMap();
    updatePlots();
    setStatus("");
  } catch (error) {
    console.error(error);
    setStatus(error.message ?? "Failed to initialize web app.", true);
  }
}

window.addEventListener("load", init);
