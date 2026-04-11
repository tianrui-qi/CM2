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
const TRACE_SOURCE_ORDER = ["c"];
const TRACE_SOURCE_VIEW_SPECS = {
  c: {
    tracePlotId: "c-trace-plot",
    heatmapPlotId: "c-heatmap-plot",
    traceDownloadButtonId: "download-c-traces-btn",
    heatmapDownloadButtonId: "download-c-heatmap-btn",
    traceDownloadName: "roi_traces_c",
    heatmapDownloadName: "roi_heatmap_c",
  },
};

const state = {
  meta: null,
  points: null,
  tracesBySource: {},
  rois: [],
  activeRoiId: null,
  mapPlotReady: false,
  mapLastWidth: null,
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
    return true;
  } catch (error) {
    console.error(error);
    return false;
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
  const plotDiv = document.getElementById("map-plot");
  const width = Math.max(plotDiv?.clientWidth ?? 0, 320);
  const aspectRatio = state.meta.full_height / state.meta.full_width;
  return Math.max(420, Math.round(width * aspectRatio) + 16);
}

function renderRoiList() {
  const container = document.getElementById("roi-list");
  container.innerHTML = "";
  for (const roi of state.rois) {
    const item = document.createElement("div");
    item.className = `roi-item${roi.id === state.activeRoiId ? " active" : ""}`;

    const main = document.createElement("div");
    main.className = "roi-main";

    const swatch = document.createElement("span");
    swatch.className = "roi-swatch";
    swatch.style.background = roi.color;

    const meta = document.createElement("div");
    meta.className = "roi-meta";

    const nameInput = document.createElement("input");
    nameInput.className = "roi-name-input";
    nameInput.value = roi.name;
    nameInput.addEventListener("input", () => {
      roi.name = nameInput.value || "Untitled ROI";
      saveUiState();
      updatePlots();
    });

    const subtext = document.createElement("div");
    subtext.className = "roi-subtext";
    subtext.textContent = `${roi.neuronIds.length} neurons`;

    meta.appendChild(nameInput);
    meta.appendChild(subtext);
    main.appendChild(swatch);
    main.appendChild(meta);

    const controls = document.createElement("div");
    controls.className = "roi-controls";

    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = roi.color;
    colorInput.className = "roi-color-input";
    colorInput.addEventListener("input", () => {
      roi.color = colorInput.value;
      swatch.style.background = roi.color;
      saveUiState();
      renderMap();
      updatePlots();
    });

    const selectBtn = document.createElement("button");
    selectBtn.className = "mini-btn";
    selectBtn.textContent = roi.id === state.activeRoiId ? "Active" : "Activate";
    selectBtn.addEventListener("click", () => {
      state.activeRoiId = roi.id;
      saveUiState();
      renderRoiList();
    });

    const clearBtn = document.createElement("button");
    clearBtn.className = "mini-btn";
    clearBtn.textContent = "Clear";
    clearBtn.addEventListener("click", () => {
      roi.neuronIds = [];
      saveUiState();
      renderMap();
      renderRoiList();
      updatePlots();
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "mini-btn";
    deleteBtn.textContent = "Delete";
    deleteBtn.disabled = state.rois.length <= 1;
    deleteBtn.addEventListener("click", () => {
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

    controls.appendChild(colorInput);
    controls.appendChild(selectBtn);
    controls.appendChild(clearBtn);
    controls.appendChild(deleteBtn);

    item.appendChild(main);
    item.appendChild(controls);
    container.appendChild(item);
  }
}

function buildMapLayout() {
  const background = state.meta.backgrounds.find((bg) => bg.key === state.meta.default_background_key) ?? state.meta.backgrounds[0];
  return {
    margin: { l: 8, r: 8, t: 8, b: 8 },
    height: computeMapHeight(),
    xaxis: {
      range: [0, state.meta.full_width],
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      fixedrange: false,
      constrain: "domain",
    },
    yaxis: {
      range: [state.meta.full_height, 0],
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      fixedrange: false,
      scaleanchor: "x",
      scaleratio: 1,
      constrain: "domain",
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
    paper_bgcolor: "#111",
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
  };
  const plotDiv = document.getElementById("map-plot");
  state.mapLastWidth = plotDiv.clientWidth;
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
  const annotations = [];
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
    const groupTop = -groupStart + 0.55;
    const groupBottom = -(groupStart + (neuronIds.length - 1) * rowStep) - 0.15;
    annotations.push({
      xref: "paper",
      x: 1.01,
      yref: "y",
      y: (groupTop + groupBottom) / 2,
      text: `${roi.name} (${neuronIds.length})`,
      showarrow: false,
      font: { color: roi.color, size: 12 },
      xanchor: "left",
    });
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
  return { traces, shapes, annotations, height };
}

function renderTracePlot(plotId, sourceKey) {
  const plotDiv = document.getElementById(plotId);
  const { traces, shapes, annotations, height } = buildTracePlotData(sourceKey);
  if (traces.length === 0) {
    Plotly.react(
      plotDiv,
      [],
      {
        margin: { l: 40, r: 40, t: 20, b: 40 },
        paper_bgcolor: "#fffdf8",
        plot_bgcolor: "#fffdf8",
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
      margin: { l: 28, r: 90, t: 14, b: 42 },
      paper_bgcolor: "#fffdf8",
      plot_bgcolor: "#fffdf8",
      xaxis: {
        title: "Time (s)",
        showgrid: false,
        zeroline: false,
      },
      yaxis: {
        visible: false,
      },
      shapes,
      annotations,
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
  const annotations = [];
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
    annotations.push({
      xref: "paper",
      x: 1.01,
      yref: "y",
      y: (startRow + endRow) / 2,
      text: `${roi.name} (${neuronIds.length})`,
      showarrow: false,
      font: { color: roi.color, size: 12 },
      xanchor: "left",
    });
  }

  const height = Math.max(320, Math.min(900, z.length * 10 + 80));
  return { x, z, shapes, annotations, height };
}

function renderHeatmapPlot(plotId, sourceKey) {
  const plotDiv = document.getElementById(plotId);
  const { x, z, shapes, annotations, height } = buildHeatmapData(sourceKey);
  if (z.length === 0) {
    Plotly.react(
      plotDiv,
      [],
      {
        margin: { l: 40, r: 40, t: 20, b: 40 },
        paper_bgcolor: "#fffdf8",
        plot_bgcolor: "#fffdf8",
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
      margin: { l: 48, r: 90, t: 14, b: 42 },
      paper_bgcolor: "#fffdf8",
      plot_bgcolor: "#fffdf8",
      xaxis: {
        title: "Time (s)",
        showgrid: false,
        zeroline: false,
      },
      yaxis: {
        visible: false,
        autorange: "reversed",
      },
      shapes,
      annotations,
      height,
    },
    { responsive: true, displaylogo: false }
  );
}

function updatePlots() {
  for (const sourceKey of TRACE_SOURCE_ORDER) {
    const viewSpec = TRACE_SOURCE_VIEW_SPECS[sourceKey];
    if (!state.meta.trace_sources[sourceKey] || !viewSpec) {
      continue;
    }
    renderTracePlot(viewSpec.tracePlotId, sourceKey);
    renderHeatmapPlot(viewSpec.heatmapPlotId, sourceKey);
  }
}

function exportPlot(plotId, filename) {
  const node = document.getElementById(plotId);
  Plotly.downloadImage(node, {
    format: "png",
    filename,
    width: Math.max(node.clientWidth, 1200),
    height: Math.max(node.clientHeight, 600),
    scale: 2,
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

  for (const sourceKey of TRACE_SOURCE_ORDER) {
    const viewSpec = TRACE_SOURCE_VIEW_SPECS[sourceKey];
    if (!viewSpec) {
      continue;
    }
    document.getElementById(viewSpec.traceDownloadButtonId).addEventListener("click", () => {
      exportPlot(viewSpec.tracePlotId, viewSpec.traceDownloadName);
    });
    document.getElementById(viewSpec.heatmapDownloadButtonId).addEventListener("click", () => {
      exportPlot(viewSpec.heatmapPlotId, viewSpec.heatmapDownloadName);
    });
  }

  window.addEventListener("resize", () => {
    const plotDiv = document.getElementById("map-plot");
    if (!plotDiv || !state.meta) {
      return;
    }
    const width = plotDiv.clientWidth;
    if (state.mapLastWidth === null || Math.abs(width - state.mapLastWidth) > 4) {
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
