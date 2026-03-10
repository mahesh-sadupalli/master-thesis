/**
 * data-loader.js — Fetch and decode binary data files for the visualization.
 */
var APP = window.APP || {};

APP.DataLoader = (function () {
  'use strict';

  let manifest = null;
  const cache = {};  // key: "source/t_NNN" -> Uint8Array

  async function loadManifest() {
    if (manifest) return manifest;
    const resp = await fetch('data/manifest.json');
    manifest = await resp.json();
    return manifest;
  }

  function getManifest() {
    return manifest;
  }

  /**
   * Load a single timestep file for a given source.
   * @param {string} source  e.g. "original", "offline_base"
   * @param {number} index   timestep index (0-29)
   * @returns {Uint8Array} shape conceptually (NX * NY * 4)
   */
  async function loadTimestep(source, index) {
    const key = source + '/t_' + String(index).padStart(3, '0');
    if (cache[key]) return cache[key];

    const url = 'data/' + key + '.bin';
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('Failed to load ' + url);
    const buf = await resp.arrayBuffer();
    const data = new Uint8Array(buf);
    cache[key] = data;
    return data;
  }

  /**
   * Load grid coordinates (Float32).
   * @returns {{ gridX: Float32Array, gridY: Float32Array }}
   */
  async function loadGrid() {
    const [rxBuf, ryBuf] = await Promise.all([
      fetch('data/grid_x.bin').then(r => r.arrayBuffer()),
      fetch('data/grid_y.bin').then(r => r.arrayBuffer()),
    ]);
    return {
      gridX: new Float32Array(rxBuf),
      gridY: new Float32Array(ryBuf),
    };
  }

  /**
   * Load cylinder mask (Uint8).
   * @returns {Uint8Array}  1 = inside cylinder, 0 = flow domain
   */
  async function loadCylinderMask() {
    const resp = await fetch('data/cylinder_mask.bin');
    const buf = await resp.arrayBuffer();
    return new Uint8Array(buf);
  }

  /**
   * Extract a single field variable from a timestep buffer.
   * Data layout: (NX, NY, 4) in row-major (C) order.
   * @param {Uint8Array} data  full timestep data
   * @param {number} fieldIdx  0=Vx, 1=Vy, 2=Pressure, 3=TKE
   * @param {number} nx
   * @param {number} ny
   * @returns {Float32Array}  values in [0, 1], length nx*ny
   */
  function extractField(data, fieldIdx, nx, ny) {
    const out = new Float32Array(nx * ny);
    for (let i = 0; i < nx * ny; i++) {
      out[i] = data[i * 4 + fieldIdx] / 255.0;
    }
    return out;
  }

  /**
   * Pre-load a range of timesteps for a source.
   */
  async function preloadSource(source, count) {
    const promises = [];
    for (let i = 0; i < count; i++) {
      promises.push(loadTimestep(source, i));
    }
    await Promise.all(promises);
  }

  return {
    loadManifest,
    getManifest,
    loadTimestep,
    loadGrid,
    loadCylinderMask,
    extractField,
    preloadSource,
  };
})();

window.APP = APP;
