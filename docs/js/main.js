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
  var lastMaxErr = 1;
  var currentOrigData = null;
  var currentPredData = null;
  var playIntervalId = null;
  var lastHoverIx = -1;
  var lastHoverIy = -1;
  var pinnedIx = -1;
  var pinnedIy = -1;

  async function init() {
    try {
      manifest = await APP.DataLoader.loadManifest();
      nx = manifest.grid.nx;
      ny = manifest.grid.ny;

      var gridData = await APP.DataLoader.loadGrid();
      gridX = gridData.gridX;
      gridY = gridData.gridY;
      cylinderMask = await APP.DataLoader.loadCylinderMask();

      APP.CanvasRenderer.init(nx, ny, cylinderMask);
      APP.CoordSystem.init(nx, ny, gridX, gridY, cylinderMask, manifest);
      APP.BitViz.initHowItWorks();

      document.getElementById('timeline-slider').max = manifest.timesteps.count - 1;

      setupControls();
      setupCanvasHover();
      setupAbstractToggle();

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

    var errResult = APP.CanvasRenderer.update(currentOrigField, currentPredField);
    currentErrorField = errResult.errorField;
    lastMaxErr = errResult.maxErr;

    APP.CoordSystem.setFieldData(currentOrigField, state.timestep);
    updateMetrics();

    if (pinnedIx >= 0) {
      APP.CanvasRenderer.drawMarker(pinnedIx, pinnedIy);
      updateBitsAtPoint(pinnedIx, pinnedIy);
    }
  }

  function setupControls() {
    // Approach dropdown
    var modeSelect = document.getElementById('mode-select');
    modeSelect.addEventListener('change', function (e) {
      var val = e.target.value;
      var comingSoon = ['lae_offline', 'lae_online', 'conv_offline', 'conv_online'];
      var csOverlay = document.getElementById('coming-soon-overlay');
      if (comingSoon.indexOf(val) !== -1) {
        csOverlay.style.display = 'flex';
        clearMetrics();
      } else {
        csOverlay.style.display = 'none';
        state.mode = val;
        loadAndDisplay();
      }
    });

    setupToggle('model-toggle', function (val) {
      state.model = val;
      var csOverlay = document.getElementById('coming-soon-overlay');
      if (csOverlay.style.display !== 'flex') {
        loadAndDisplay();
      }
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
    if (!m) { clearMetrics(); return; }
    document.getElementById('metric-psnr').textContent = m.psnr.toFixed(2);
    document.getElementById('metric-ssim').textContent = m.ssim.toFixed(4);
    document.getElementById('metric-error').textContent = m.rel_error.toFixed(2);
    document.getElementById('metric-params').textContent = m.params.toLocaleString();
    document.getElementById('metric-size').textContent = m.size_kb.toFixed(1);
    document.getElementById('metric-cr').textContent = m.cr.toLocaleString();
  }

  function clearMetrics() {
    ['metric-psnr', 'metric-ssim', 'metric-error', 'metric-params', 'metric-size', 'metric-cr']
      .forEach(function (id) {
        var el = document.getElementById(id);
        if (el) el.textContent = '--';
      });
  }

  function setupAbstractToggle() {
    var btn = document.getElementById('abstract-toggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var desc = document.getElementById('description');
      desc.classList.toggle('collapsed');
      btn.innerHTML = desc.classList.contains('collapsed')
        ? 'Abstract &darr;'
        : 'Abstract &uarr;';
    });
  }

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

  function setupCanvasHover() {
    ['canvas-original', 'canvas-predicted', 'canvas-error'].forEach(function (id) {
      var c = document.getElementById(id);
      if (!c) return;
      c.style.cursor = 'crosshair';
      c.addEventListener('mousemove', onCanvasHover);
      c.addEventListener('click', onCanvasClick);
    });
  }

  function onCanvasHover(e) {
    if (!currentOrigField || !currentPredField) return;
    var grid = APP.CanvasRenderer.canvasToGrid(e.target, e.clientX, e.clientY);
    if (!grid) return;
    if (cylinderMask[grid.ix * ny + grid.iy]) return;

    lastHoverIx = grid.ix;
    lastHoverIy = grid.iy;

    var ph = document.getElementById('bit-placeholder');
    if (ph) ph.style.display = 'none';

    if (pinnedIx < 0) {
      updateBitsAtPoint(lastHoverIx, lastHoverIy);
    }
  }

  function onCanvasClick(e) {
    if (!currentOrigField || !currentPredField) return;
    var grid = APP.CanvasRenderer.canvasToGrid(e.target, e.clientX, e.clientY);
    if (!grid) return;
    if (cylinderMask[grid.ix * ny + grid.iy]) return;

    if (pinnedIx === grid.ix && pinnedIy === grid.iy) {
      pinnedIx = -1;
      pinnedIy = -1;
      updatePinIndicator(false);
      var errResult = APP.CanvasRenderer.update(currentOrigField, currentPredField);
      currentErrorField = errResult.errorField;
    } else {
      pinnedIx = grid.ix;
      pinnedIy = grid.iy;
      updatePinIndicator(true);

      var ph = document.getElementById('bit-placeholder');
      if (ph) ph.style.display = 'none';

      APP.CanvasRenderer.drawMarker(pinnedIx, pinnedIy);
      updateBitsAtPoint(pinnedIx, pinnedIy);
    }
  }

  function updatePinIndicator(pinned) {
    var el = document.getElementById('pin-status');
    if (!el) return;
    if (pinned) {
      var x = gridX[pinnedIx].toFixed(4);
      var y = gridY[pinnedIy].toFixed(4);
      el.innerHTML = '&#128204; Pinned at (' + x + ', ' + y + ') &mdash; <a href="#" id="unpin-link">click point or here to unpin</a>';
      el.style.display = 'block';
      var link = document.getElementById('unpin-link');
      if (link) {
        link.addEventListener('click', function (ev) {
          ev.preventDefault();
          pinnedIx = -1;
          pinnedIy = -1;
          updatePinIndicator(false);
          var errResult = APP.CanvasRenderer.update(currentOrigField, currentPredField);
          currentErrorField = errResult.errorField;
        });
      }
    } else {
      el.innerHTML = 'Click on the flow field to pin a tracking point';
      el.style.display = 'block';
    }
  }

  function updateBitsAtPoint(ix, iy) {
    if (!currentOrigField || !currentPredField || !currentOrigData) return;
    var idx = ix * ny + iy;

    var origVal = currentOrigField[idx];
    var predVal = currentPredField[idx];
    var errVal = Math.abs(origVal - predVal);
    APP.BitViz.updateVizBits(state.fieldIdx, origVal, predVal, errVal);

    var x = gridX[ix];
    var y = gridY[iy];
    var z = 0.0;
    var t = manifest.timesteps.values[state.timestep];
    APP.BitViz.updateRowBits('how-inputs', ['x', 'y', 'z', 't'], [x, y, z, t]);

    var vx  = currentOrigData[idx * 4 + 0] / 255.0;
    var vy  = currentOrigData[idx * 4 + 1] / 255.0;
    var p   = currentOrigData[idx * 4 + 2] / 255.0;
    var tke = currentOrigData[idx * 4 + 3] / 255.0;
    APP.BitViz.updateRowBits('how-outputs', ['Vx', 'Vy', 'P', 'TKE'], [vx, vy, p, tke]);
  }

  document.addEventListener('DOMContentLoaded', init);
  return { state: state };
})();

window.APP = APP;
