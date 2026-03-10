/**
 * coord-system.js — Interactive coordinate system scatter plot.
 *
 * Visualizes how the 7.9M data points are distributed in (x,y,z,t) space.
 * Uses the exported grid data + timesteps to show 2D projections.
 * Points colored by field value or time.
 */
var APP = window.APP || {};

APP.CoordSystem = (function () {
  'use strict';

  var canvas, ctx;
  var gridX, gridY, cylinderMask;
  var manifest;
  var nx, ny;

  // Current state
  var projection = 'xy';  // 'xy', 'xt', 'yt'
  var colorMode = 'field'; // 'field' or 'time'

  // Current field data (for coloring)
  var fieldData = null;
  var currentTimestep = 0;

  // Axis labels
  var axisLabels = {
    xy: { x: 'x (spatial)', y: 'y (spatial)' },
    xt: { x: 'x (spatial)', y: 't (time)' },
    yt: { x: 'y (spatial)', y: 't (time)' },
  };

  function init(_nx, _ny, _gridX, _gridY, _cylinderMask, _manifest) {
    nx = _nx;
    ny = _ny;
    gridX = _gridX;
    gridY = _gridY;
    cylinderMask = _cylinderMask;
    manifest = _manifest;

    canvas = document.getElementById('coord-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');

    // Setup toggle handlers
    setupToggle('coord-proj-toggle', function (val) {
      projection = val;
      draw();
    });

    setupToggle('coord-color-toggle', function (val) {
      colorMode = val;
      draw();
    });

    draw();
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

  /**
   * Set the current field data for coloring points.
   */
  function setFieldData(_fieldData, _timestep) {
    fieldData = _fieldData;
    currentTimestep = _timestep;
    draw();
  }

  function jetRGB(t) {
    return APP.CanvasRenderer.jetRGB(t);
  }

  function draw() {
    if (!ctx || !gridX || !gridY) return;

    var w = canvas.width;
    var h = canvas.height;
    var pad = 48;  // padding for axes

    ctx.fillStyle = '#080810';
    ctx.fillRect(0, 0, w, h);

    var plotW = w - pad - 20;
    var plotH = h - pad - 20;
    var plotX = pad;
    var plotY = 10;

    // Determine data ranges
    var xMin, xMax, yMin, yMax;
    var tVals = manifest.timesteps.values;
    var tMin = tVals[0];
    var tMax = tVals[tVals.length - 1];

    if (projection === 'xy') {
      xMin = manifest.grid.x_min; xMax = manifest.grid.x_max;
      yMin = manifest.grid.y_min; yMax = manifest.grid.y_max;
    } else if (projection === 'xt') {
      xMin = manifest.grid.x_min; xMax = manifest.grid.x_max;
      yMin = tMin; yMax = tMax;
    } else {
      xMin = manifest.grid.y_min; xMax = manifest.grid.y_max;
      yMin = tMin; yMax = tMax;
    }

    // Map functions
    function mapX(val) { return plotX + (val - xMin) / (xMax - xMin) * plotW; }
    function mapY(val) { return plotY + plotH - (val - yMin) / (yMax - yMin) * plotH; }

    // Draw grid lines
    ctx.strokeStyle = '#1a1a2a';
    ctx.lineWidth = 0.5;
    for (var i = 0; i <= 5; i++) {
      var gx = plotX + (i / 5) * plotW;
      var gy = plotY + (i / 5) * plotH;
      ctx.beginPath(); ctx.moveTo(gx, plotY); ctx.lineTo(gx, plotY + plotH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(plotX, gy); ctx.lineTo(plotX + plotW, gy); ctx.stroke();
    }

    // Draw points
    var pointSize = 1.5;
    var numTimesteps = manifest.timesteps.count;

    if (projection === 'xy') {
      // Draw all grid points at current timestep
      for (var ix = 0; ix < nx; ix++) {
        for (var iy = 0; iy < ny; iy++) {
          var idx = ix * ny + iy;
          if (cylinderMask[idx]) continue;

          var px = mapX(gridX[ix]);
          var py = mapY(gridY[iy]);

          var rgb;
          if (colorMode === 'field' && fieldData) {
            rgb = jetRGB(fieldData[idx]);
          } else {
            // Color by time
            var tNorm = currentTimestep / Math.max(1, numTimesteps - 1);
            rgb = jetRGB(tNorm);
          }
          ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
          ctx.fillRect(px - pointSize, py - pointSize, pointSize * 2, pointSize * 2);
        }
      }

      // Draw cylinder outline
      drawCylinderOutline(mapX, mapY);

    } else if (projection === 'xt') {
      // Show points across timesteps (sample some)
      for (var ti = 0; ti < numTimesteps; ti++) {
        var tVal = tVals[ti];
        var tNorm2 = ti / Math.max(1, numTimesteps - 1);
        // Sample every 4th x point for clarity
        for (var ix2 = 0; ix2 < nx; ix2 += 2) {
          var px2 = mapX(gridX[ix2]);
          var py2 = mapY(tVal);
          var rgb2;
          if (colorMode === 'time') {
            rgb2 = jetRGB(tNorm2);
          } else {
            rgb2 = jetRGB(ix2 / nx);
          }
          ctx.fillStyle = 'rgba(' + rgb2[0] + ',' + rgb2[1] + ',' + rgb2[2] + ',0.6)';
          ctx.fillRect(px2 - 1, py2 - 1, 2, 2);
        }
      }

      // Highlight current timestep
      var currTVal = tVals[currentTimestep] || 0;
      var cy = mapY(currTVal);
      ctx.strokeStyle = 'rgba(79, 195, 247, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(plotX, cy); ctx.lineTo(plotX + plotW, cy); ctx.stroke();
      ctx.setLineDash([]);

    } else { // 'yt'
      for (var ti2 = 0; ti2 < numTimesteps; ti2++) {
        var tVal2 = tVals[ti2];
        var tNorm3 = ti2 / Math.max(1, numTimesteps - 1);
        for (var iy2 = 0; iy2 < ny; iy2 += 2) {
          var px3 = mapX(gridY[iy2]);
          var py3 = mapY(tVal2);
          var rgb3;
          if (colorMode === 'time') {
            rgb3 = jetRGB(tNorm3);
          } else {
            rgb3 = jetRGB(iy2 / ny);
          }
          ctx.fillStyle = 'rgba(' + rgb3[0] + ',' + rgb3[1] + ',' + rgb3[2] + ',0.6)';
          ctx.fillRect(px3 - 1, py3 - 1, 2, 2);
        }
      }

      var currTVal2 = tVals[currentTimestep] || 0;
      var cy2 = mapY(currTVal2);
      ctx.strokeStyle = 'rgba(79, 195, 247, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(plotX, cy2); ctx.lineTo(plotX + plotW, cy2); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Axes
    ctx.strokeStyle = '#444460';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotX, plotY);
    ctx.lineTo(plotX, plotY + plotH);
    ctx.lineTo(plotX + plotW, plotY + plotH);
    ctx.stroke();

    // Axis labels
    var labels = axisLabels[projection];
    ctx.fillStyle = '#8888a0';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(labels.x, plotX + plotW / 2, h - 4);

    ctx.save();
    ctx.translate(12, plotY + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(labels.y, 0, 0);
    ctx.restore();

    // Tick values
    ctx.font = '9px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
    ctx.fillStyle = '#666680';
    ctx.textAlign = 'center';
    ctx.fillText(xMin.toFixed(3), plotX, plotY + plotH + 14);
    ctx.fillText(xMax.toFixed(3), plotX + plotW, plotY + plotH + 14);
    ctx.textAlign = 'right';
    ctx.fillText(yMax.toFixed(3), plotX - 4, plotY + 10);
    ctx.fillText(yMin.toFixed(3), plotX - 4, plotY + plotH + 3);
  }

  function drawCylinderOutline(mapX, mapY) {
    // Find boundary of cylinder mask
    ctx.strokeStyle = 'rgba(200, 200, 220, 0.4)';
    ctx.lineWidth = 1;

    // Simple approach: draw a circle at the cylinder center
    var cx = manifest.cylinder.center_x;
    var cy = manifest.cylinder.center_y;
    var r = manifest.cylinder.radius;

    var screenCx = mapX(cx);
    var screenCy = mapY(cy);
    var screenR = Math.abs(mapX(cx + r) - screenCx);

    ctx.beginPath();
    ctx.arc(screenCx, screenCy, screenR, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = 'rgba(50, 50, 58, 0.8)';
    ctx.fill();
  }

  return {
    init: init,
    setFieldData: setFieldData,
    draw: draw,
  };
})();

window.APP = APP;
