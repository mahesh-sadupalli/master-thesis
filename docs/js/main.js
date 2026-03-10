/**
 * main.js — App initialization, UI controls, and animation loop.
 */
var APP = window.APP || {};

APP.Main = (function () {
  'use strict';

  var state = {
    mode: 'offline',
    model: 'base',
    fieldIdx: 0,
    timestep: 0,
    playing: false,
    speed: 1,
  };

  var manifest;
  var gridX, gridY, cylinderMask;
  var nx, ny;
  var currentOrigField = null;
  var currentPredField = null;
  var currentErrorField = null;
  var currentOrigData = null;
  var currentPredData = null;
  var playIntervalId = null;

  // ── Init ────────────────────────────────────────────────────────────────
  async function init() {
    try {
      manifest = await APP.DataLoader.loadManifest();
      nx = manifest.grid.nx;
      ny = manifest.grid.ny;

      var gridData = await APP.DataLoader.loadGrid();
      gridX = gridData.gridX;
      gridY = gridData.gridY;
      cylinderMask = await APP.DataLoader.loadCylinderMask();

      // 2D renderer
      APP.CanvasRenderer.init(nx, ny, cylinderMask);

      // 3D scene
      APP.ThreeScene.init(nx, ny, gridX, gridY, cylinderMask);
      APP.ThreeScene.show();

      // Coordinate system scatter plot
      APP.CoordSystem.init(nx, ny, gridX, gridY, cylinderMask, manifest);

      // Bit viz (How It Works)
      APP.BitViz.initHowItWorks();

      document.getElementById('timeline-slider').max = manifest.timesteps.count - 1;

      setupControls();
      setupCanvasHover();

      await loadAndDisplay();

      var overlay = document.getElementById('loading-overlay');
      if (overlay) overlay.classList.add('hidden');

    } catch (err) {
      console.error('Init error:', err, err.stack);
      var overlay = document.getElementById('loading-overlay');
      if (overlay) {
        overlay.querySelector('p').textContent = 'Error: ' + err.message;
        overlay.querySelector('.spinner').style.display = 'none';
      }
    }
  }

  // ── Data ────────────────────────────────────────────────────────────────
  function getSourceKey() {
    return state.mode + '_' + state.model;
  }

  async function loadAndDisplay() {
    var sourceKey = getSourceKey();
    var t = state.timestep;

    var results = await Promise.all([
      APP.DataLoader.loadTimestep('original', t),
      APP.DataLoader.loadTimestep(sourceKey, t),
    ]);
    currentOrigData = results[0];
    currentPredData = results[1];

    currentOrigField = APP.DataLoader.extractField(currentOrigData, state.fieldIdx, nx, ny);
    currentPredField = APP.DataLoader.extractField(currentPredData, state.fieldIdx, nx, ny);

    // 2D canvases
    var errResult = APP.CanvasRenderer.update(currentOrigField, currentPredField);
    currentErrorField = errResult.errorField;

    // 3D surface
    APP.ThreeScene.updateSurface(currentPredField);

    // Coordinate system coloring
    APP.CoordSystem.setFieldData(currentOrigField, state.timestep);

    updateMetrics();
  }

  // ── Controls ────────────────────────────────────────────────────────────
  function setupControls() {
    setupToggle('mode-toggle', function (val) {
      state.mode = val;
      loadAndDisplay();
    });

    setupToggle('model-toggle', function (val) {
      state.model = val;
      loadAndDisplay();
    });

    document.getElementById('field-select').addEventListener('change', function (e) {
      state.fieldIdx = parseInt(e.target.value);
      loadAndDisplay();
    });

    document.getElementById('play-btn').addEventListener('click', function () {
      state.playing = !state.playing;
      this.textContent = state.playing ? '\u23F8 Pause' : '\u25B6 Play';
      this.classList.toggle('playing', state.playing);
      if (state.playing) startPlayback(); else stopPlayback();
    });

    document.getElementById('timeline-slider').addEventListener('input', function (e) {
      state.timestep = parseInt(e.target.value);
      updateTimestepLabel();
      loadAndDisplay();
    });

    document.getElementById('speed-slider').addEventListener('input', function (e) {
      state.speed = parseFloat(e.target.value);
      if (state.playing) { stopPlayback(); startPlayback(); }
    });
  }

  function setupToggle(groupId, callback) {
    var group = document.getElementById(groupId);
    if (!group) return;
    var btns = group.querySelectorAll('.toggle-btn');
    btns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        btns.forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        callback(btn.dataset.value);
      });
    });
  }

  function updateTimestepLabel() {
    var el = document.getElementById('timestep-label');
    if (el) el.textContent = state.timestep + ' / ' + (manifest.timesteps.count - 1);
  }

  function updateMetrics() {
    var m = manifest.metrics[getSourceKey()];
    if (!m) return;
    document.getElementById('metric-psnr').textContent = m.psnr.toFixed(2);
    document.getElementById('metric-ssim').textContent = m.ssim.toFixed(4);
    document.getElementById('metric-error').textContent = m.rel_error.toFixed(2);
    document.getElementById('metric-params').textContent = m.params.toLocaleString();
    document.getElementById('metric-size').textContent = m.size_kb.toFixed(1);
    document.getElementById('metric-cr').textContent = m.cr.toLocaleString();
  }

  // ── Playback ────────────────────────────────────────────────────────────
  function startPlayback() {
    stopPlayback();
    playIntervalId = setInterval(function () {
      state.timestep = (state.timestep + 1) % manifest.timesteps.count;
      document.getElementById('timeline-slider').value = state.timestep;
      updateTimestepLabel();
      loadAndDisplay();
    }, 400 / state.speed);
  }

  function stopPlayback() {
    if (playIntervalId) { clearInterval(playIntervalId); playIntervalId = null; }
  }

  // ── Canvas hover → bits ─────────────────────────────────────────────────
  function setupCanvasHover() {
    ['canvas-original', 'canvas-predicted', 'canvas-error'].forEach(function (id) {
      var c = document.getElementById(id);
      if (!c) return;
      c.addEventListener('mousemove', onCanvasHover);
    });
  }

  function onCanvasHover(e) {
    if (!currentOrigField || !currentPredField) return;
    var grid = APP.CanvasRenderer.canvasToGrid(e.target, e.clientX, e.clientY);
    if (!grid) return;
    var idx = grid.ix * ny + grid.iy;
    if (cylinderMask[idx]) return;

    var origVal = currentOrigField[idx];
    var predVal = currentPredField[idx];
    var errVal = Math.abs(origVal - predVal);
    APP.BitViz.updateVizBits(state.fieldIdx, origVal, predVal, errVal);
  }

  // ── Boot ────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', init);
  return { state: state };
})();

window.APP = APP;
