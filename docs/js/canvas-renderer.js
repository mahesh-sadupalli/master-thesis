/**
 * canvas-renderer.js — High-quality 2D color-mapped heatmap rendering.
 *
 * Renders the 200x100 data grid as smooth jet-colormap images.
 * Uses a high-resolution backing canvas with bilinear interpolation
 * for smooth visuals (no visible pixels).
 */
var APP = window.APP || {};

APP.CanvasRenderer = (function () {
  'use strict';

  var canvasOriginal, ctxOriginal;
  var canvasPredicted, ctxPredicted;
  var canvasError, ctxError;
  var nx, ny;
  var cylinderMask;

  // Render at higher resolution for smooth output
  var SCALE = 4;  // 200*4=800 x 100*4=400
  var renderW, renderH;

  /**
   * Jet colormap: t in [0,1] -> [r, g, b] each 0-255.
   */
  function jetRGB(t) {
    var r, g, b;
    if (t < 0.125)      { r = 0;                 g = 0;                 b = 0.5 + t * 4; }
    else if (t < 0.375) { r = 0;                 g = (t - 0.125) * 4;  b = 1; }
    else if (t < 0.625) { r = (t - 0.375) * 4;  g = 1;                 b = 1 - (t - 0.375) * 4; }
    else if (t < 0.875) { r = 1;                 g = 1 - (t - 0.625) * 4; b = 0; }
    else                 { r = 1 - (t - 0.875) * 4; g = 0;             b = 0; }
    return [
      Math.max(0, Math.min(255, (r * 255) | 0)),
      Math.max(0, Math.min(255, (g * 255) | 0)),
      Math.max(0, Math.min(255, (b * 255) | 0)),
    ];
  }

  /** Hot colormap for error. */
  function errorRGB(t) {
    t = Math.max(0, Math.min(1, t));
    var r, g, b;
    if (t < 0.33)      { r = t / 0.33; g = 0; b = 0; }
    else if (t < 0.66) { r = 1; g = (t - 0.33) / 0.33; b = 0; }
    else               { r = 1; g = 1; b = (t - 0.66) / 0.34; }
    return [(r * 255) | 0, (g * 255) | 0, (b * 255) | 0];
  }

  /**
   * Bilinear sample from fieldData at fractional grid position (fx, fy).
   * fieldData is nx*ny, indexed as [ix*ny + iy].
   */
  function bilinearSample(fieldData, fx, fy) {
    var ix0 = Math.max(0, Math.min(nx - 2, fx | 0));
    var iy0 = Math.max(0, Math.min(ny - 2, fy | 0));
    var ix1 = ix0 + 1;
    var iy1 = iy0 + 1;
    var sx = fx - ix0;
    var sy = fy - iy0;

    var v00 = fieldData[ix0 * ny + iy0];
    var v10 = fieldData[ix1 * ny + iy0];
    var v01 = fieldData[ix0 * ny + iy1];
    var v11 = fieldData[ix1 * ny + iy1];

    return v00 * (1 - sx) * (1 - sy) +
           v10 * sx * (1 - sy) +
           v01 * (1 - sx) * sy +
           v11 * sx * sy;
  }

  /** Check if fractional grid position is inside cylinder. */
  function isCylinder(fx, fy) {
    var ix = Math.round(fx);
    var iy = Math.round(fy);
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) return false;
    return cylinderMask[ix * ny + iy] === 1;
  }

  function init(_nx, _ny, _cylinderMask) {
    nx = _nx;
    ny = _ny;
    cylinderMask = _cylinderMask;
    renderW = nx * SCALE;
    renderH = ny * SCALE;

    canvasOriginal  = document.getElementById('canvas-original');
    canvasPredicted = document.getElementById('canvas-predicted');
    canvasError     = document.getElementById('canvas-error');

    [canvasOriginal, canvasPredicted, canvasError].forEach(function (c) {
      c.width = renderW;
      c.height = renderH;
    });

    ctxOriginal  = canvasOriginal.getContext('2d');
    ctxPredicted = canvasPredicted.getContext('2d');
    ctxError     = canvasError.getContext('2d');

    drawColorbar('cbar-original', 'jet');
    drawColorbar('cbar-predicted', 'jet');
    drawColorbar('cbar-error', 'hot');
  }

  /**
   * Render a field onto a canvas with bilinear interpolation and colormapping.
   */
  function renderSmooth(ctx, fieldData, colormapFn, maxVal) {
    var imgData = ctx.createImageData(renderW, renderH);
    var data = imgData.data;
    maxVal = maxVal || 1.0;

    for (var py = 0; py < renderH; py++) {
      // Map pixel y to grid iy (flip: py=0 is top -> iy=ny-1)
      var fy = (ny - 1) - py / (renderH - 1) * (ny - 1);

      for (var px = 0; px < renderW; px++) {
        var fx = px / (renderW - 1) * (nx - 1);
        var pixelIdx = (py * renderW + px) * 4;

        if (isCylinder(fx, fy)) {
          data[pixelIdx]     = 50;
          data[pixelIdx + 1] = 50;
          data[pixelIdx + 2] = 58;
          data[pixelIdx + 3] = 255;
        } else {
          var val = bilinearSample(fieldData, fx, fy) / maxVal;
          val = Math.max(0, Math.min(1, val));
          var rgb = colormapFn(val);
          data[pixelIdx]     = rgb[0];
          data[pixelIdx + 1] = rgb[1];
          data[pixelIdx + 2] = rgb[2];
          data[pixelIdx + 3] = 255;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  /**
   * Update all three canvases.
   */
  function update(origField, predField) {
    renderSmooth(ctxOriginal, origField, jetRGB, 1.0);
    renderSmooth(ctxPredicted, predField, jetRGB, 1.0);

    // Compute error
    var errorField = new Float32Array(nx * ny);
    var maxErr = 0;
    for (var i = 0; i < nx * ny; i++) {
      var e = Math.abs(origField[i] - predField[i]);
      errorField[i] = e;
      if (e > maxErr && !cylinderMask[i]) maxErr = e;
    }
    if (maxErr < 0.001) maxErr = 0.1;

    renderSmooth(ctxError, errorField, errorRGB, maxErr);

    var errLabel = document.getElementById('err-max');
    if (errLabel) errLabel.textContent = maxErr.toFixed(3);

    return { errorField: errorField, maxErr: maxErr };
  }

  function drawColorbar(canvasId, type) {
    var c = document.getElementById(canvasId);
    if (!c) return;
    c.width = 14;
    c.height = 200;
    var ctx = c.getContext('2d');
    var fn = type === 'hot' ? errorRGB : jetRGB;
    for (var y = 0; y < 200; y++) {
      var t = 1 - y / 199;
      var rgb = fn(t);
      ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
      ctx.fillRect(0, y, 14, 1);
    }
  }

  /**
   * Get grid indices from canvas mouse coordinates.
   */
  function canvasToGrid(canvas, clientX, clientY) {
    var rect = canvas.getBoundingClientRect();
    var px = (clientX - rect.left) / rect.width;
    var py = (clientY - rect.top) / rect.height;
    if (px < 0 || px > 1 || py < 0 || py > 1) return null;
    var ix = Math.round(px * (nx - 1));
    var iy = Math.round((1 - py) * (ny - 1));  // flip Y
    ix = Math.max(0, Math.min(nx - 1, ix));
    iy = Math.max(0, Math.min(ny - 1, iy));
    return { ix: ix, iy: iy };
  }

  return {
    init: init,
    update: update,
    canvasToGrid: canvasToGrid,
    jetRGB: jetRGB,
  };
})();

window.APP = APP;
