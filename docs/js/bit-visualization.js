/**
 * bit-visualization.js — IEEE 754 bit breakdown display.
 *
 * Renders 64-bit double-precision layout as colored boxes:
 *   sign (1 bit, red) | exponent (11 bits, blue) | mantissa (52 bits, green)
 */
var APP = window.APP || {};

APP.BitViz = (function () {
  'use strict';

  var INPUT_LABELS  = ['x', 'y', 'z', 't'];
  var OUTPUT_LABELS = ['Vx', 'Vy', 'P', 'TKE'];

  /**
   * Convert a JS number to its IEEE 754 double-precision 64-bit string.
   * Returns a string of '0' and '1', length 64.
   */
  function float64ToBits(value) {
    var buf = new ArrayBuffer(8);
    new Float64Array(buf)[0] = value;
    var bytes = new Uint8Array(buf);
    var bits = '';
    for (var i = 7; i >= 0; i--) {
      bits += bytes[i].toString(2).padStart(8, '0');
    }
    return bits;
  }

  /**
   * Create a single bit-row element: label | value | bit boxes.
   */
  function createBitRow(label, value) {
    var row = document.createElement('div');
    row.className = 'bit-row';

    var lbl = document.createElement('span');
    lbl.className = 'bit-row-label';
    lbl.textContent = label;
    row.appendChild(lbl);

    var val = document.createElement('span');
    val.className = 'bit-row-value';
    val.textContent = (value !== undefined && value !== null)
      ? (typeof value === 'number' ? value.toFixed(6) : String(value))
      : '--';
    row.appendChild(val);

    var bits = float64ToBits(value || 0);
    var container = document.createElement('span');
    container.className = 'bit-boxes';
    for (var i = 0; i < 64; i++) {
      var box = document.createElement('span');
      box.className = 'bit-box';
      if (i === 0)      box.classList.add('sign');
      else if (i < 12)  box.classList.add('exp');
      else               box.classList.add('mant');
      box.title = (i === 0 ? 'Sign' : i < 12 ? 'Exponent' : 'Mantissa') +
                  ' [' + i + '] = ' + bits[i];
      container.appendChild(box);
    }
    row.appendChild(container);
    return row;
  }

  /**
   * Populate the "How It Works" section.
   */
  function initHowItWorks() {
    var inputEl = document.getElementById('how-inputs');
    var outputEl = document.getElementById('how-outputs');
    if (!inputEl || !outputEl) return;

    var sampleInputs  = [0.05, 0.02, 0.0, 0.013];
    var sampleOutputs = [0.72, 0.15, 0.48, 0.03];

    INPUT_LABELS.forEach(function (lbl, i) {
      inputEl.appendChild(createBitRow(lbl, sampleInputs[i]));
    });
    OUTPUT_LABELS.forEach(function (lbl, i) {
      outputEl.appendChild(createBitRow(lbl, sampleOutputs[i]));
    });
  }

  /**
   * Update the in-row bit display for a specific viz row.
   * @param {string} containerId  e.g. 'bits-original'
   * @param {string[]} labels     field labels
   * @param {number[]} values     field values (up to 4)
   */
  function updateRowBits(containerId, labels, values) {
    var el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '';
    for (var i = 0; i < labels.length; i++) {
      el.appendChild(createBitRow(labels[i], values[i]));
    }
  }

  /**
   * Update all three viz row bit displays for a clicked point.
   * @param {number} fieldIdx    which field is being displayed (0-3)
   * @param {number} origVal     original value
   * @param {number} predVal     predicted value
   * @param {number} errVal      absolute error value
   */
  function updateVizBits(fieldIdx, origVal, predVal, errVal) {
    var fieldName = OUTPUT_LABELS[fieldIdx];
    updateRowBits('bits-original',  [fieldName], [origVal]);
    updateRowBits('bits-predicted', [fieldName], [predVal]);
    updateRowBits('bits-error',     [fieldName + ' err'], [errVal]);
  }

  return {
    initHowItWorks: initHowItWorks,
    updateVizBits: updateVizBits,
    updateRowBits: updateRowBits,
    float64ToBits: float64ToBits,
    createBitRow: createBitRow,
  };
})();

window.APP = APP;
