/**
 * dashboard.js — Data distribution & statistics for the data dashboard.
 */
(function () {
  'use strict';

  var COLORS = {
    Vx: '#1B4F72',
    Vy: '#E74C3C',
    Pressure: '#27AE60',
    TKE: '#8E44AD',
  };

  var stats = null;

  async function init() {
    var resp = await fetch('data/dataset_stats.json');
    stats = await resp.json();
    renderDescriptiveStats();
    renderDistributions();
    renderBoxPlots();
    renderCorrelation();
    renderSpatial();
    renderTKEDetail();
    renderTemporal();
    renderCoordSummary();
  }

  // ── Descriptive Statistics Table ────────────────────────────────────
  function renderDescriptiveStats() {
    var tbody = document.getElementById('desc-stats-body');
    var fields = ['Vx', 'Vy', 'Pressure', 'TKE'];
    var html = '';

    fields.forEach(function (name) {
      var s = stats[name];
      html += '<tr>'
        + '<td class="var-name" style="color:' + COLORS[name] + '">' + name + '</td>'
        + '<td>' + s.count.toLocaleString() + '</td>'
        + '<td>' + s.mean.toFixed(4) + '</td>'
        + '<td>' + s.median.toFixed(4) + '</td>'
        + '<td>' + s.std.toFixed(4) + '</td>'
        + '<td>' + s.skewness.toFixed(3) + '</td>'
        + '<td>' + s.kurtosis.toFixed(3) + '</td>'
        + '<td>' + s.min.toFixed(4) + '</td>'
        + '<td>' + s.q1.toFixed(4) + '</td>'
        + '<td>' + s.q3.toFixed(4) + '</td>'
        + '<td>' + s.max.toFixed(4) + '</td>'
        + '<td>' + s.iqr.toFixed(4) + '</td>'
        + '</tr>';
    });

    tbody.innerHTML = html;
  }

  // ── Histogram Distributions ─────────────────────────────────────────
  function renderDistributions() {
    var canvas = document.getElementById('chart-distributions');
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.parentElement.clientWidth;
    var h = 220;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    var fields = ['Vx', 'Vy', 'Pressure', 'TKE'];
    var n = fields.length;
    var pad = { top: 24, bottom: 36, left: 55, right: 20 };
    var chartW = (w - pad.left - pad.right - (n - 1) * 20) / n;
    var chartH = h - pad.top - pad.bottom;

    fields.forEach(function (name, fi) {
      var hist = stats[name + '_hist'];
      var maxCount = Math.max.apply(null, hist.counts);
      var ox = pad.left + fi * (chartW + 20);
      var oy = pad.top;
      var color = COLORS[name];

      // Bars
      var barW = chartW / hist.counts.length;
      hist.counts.forEach(function (count, bi) {
        var barH = (count / maxCount) * chartH;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.75;
        ctx.fillRect(ox + bi * barW, oy + chartH - barH, Math.max(barW - 0.5, 1), barH);
      });
      ctx.globalAlpha = 1;

      // X axis
      ctx.strokeStyle = '#e0e4ed';
      ctx.beginPath();
      ctx.moveTo(ox, oy + chartH);
      ctx.lineTo(ox + chartW, oy + chartH);
      ctx.stroke();

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 13px "Source Sans 3", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(name, ox + chartW / 2, oy + chartH + 20);

      // Subtitle stats
      var s = stats[name];
      ctx.fillStyle = '#718096';
      ctx.font = '10px "Inter", sans-serif';
      ctx.fillText('skew=' + s.skewness.toFixed(2), ox + chartW / 2, oy + chartH + 32);

      // Y max tick
      ctx.fillStyle = '#718096';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText((maxCount / 1000).toFixed(0) + 'K', ox - 4, oy + 12);

      // X ticks
      ctx.textAlign = 'center';
      ctx.fillText('0', ox, oy + chartH + 12);
      ctx.fillText('1', ox + chartW, oy + chartH + 12);
    });
  }

  // ── Box Plots ───────────────────────────────────────────────────────
  function renderBoxPlots() {
    var canvas = document.getElementById('chart-boxplots');
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.parentElement.clientWidth;
    var h = 300;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    var fields = ['Vx', 'Vy', 'Pressure', 'TKE'];
    var n = fields.length;
    var pad = { top: 30, bottom: 40, left: 50, right: 20 };
    var cw = w - pad.left - pad.right;
    var ch = h - pad.top - pad.bottom;
    var boxW = cw / n;

    // Y axis (0 to 1)
    ctx.strokeStyle = '#e0e4ed';
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + ch);
    ctx.lineTo(pad.left + cw, pad.top + ch);
    ctx.stroke();

    // Grid lines
    for (var g = 0; g <= 4; g++) {
      var gy = pad.top + ch - (g / 4) * ch;
      ctx.strokeStyle = '#f0f2f5';
      ctx.beginPath();
      ctx.moveTo(pad.left, gy);
      ctx.lineTo(pad.left + cw, gy);
      ctx.stroke();
      ctx.fillStyle = '#718096';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText((g * 0.25).toFixed(2), pad.left - 6, gy + 4);
    }

    fields.forEach(function (name, fi) {
      var s = stats[name];
      var cx = pad.left + fi * boxW + boxW / 2;
      var bw = boxW * 0.5;
      var color = COLORS[name];

      function yPos(val) { return pad.top + ch - val * ch; }

      // Whiskers (p5 to p95)
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, yPos(s.p5));
      ctx.lineTo(cx, yPos(s.q1));
      ctx.moveTo(cx, yPos(s.q3));
      ctx.lineTo(cx, yPos(s.p95));
      ctx.stroke();

      // Whisker caps
      ctx.beginPath();
      ctx.moveTo(cx - bw * 0.3, yPos(s.p5));
      ctx.lineTo(cx + bw * 0.3, yPos(s.p5));
      ctx.moveTo(cx - bw * 0.3, yPos(s.p95));
      ctx.lineTo(cx + bw * 0.3, yPos(s.p95));
      ctx.stroke();

      // Box (Q1 to Q3)
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.2;
      ctx.fillRect(cx - bw / 2, yPos(s.q3), bw, yPos(s.q1) - yPos(s.q3));
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(cx - bw / 2, yPos(s.q3), bw, yPos(s.q1) - yPos(s.q3));

      // Median line
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx - bw / 2, yPos(s.median));
      ctx.lineTo(cx + bw / 2, yPos(s.median));
      ctx.stroke();

      // Mean dot
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, yPos(s.mean), 4, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 13px "Source Sans 3", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(name, cx, pad.top + ch + 24);

      ctx.lineWidth = 1;
    });

    // Legend
    ctx.fillStyle = '#4a5568';
    ctx.font = '10px "Inter", sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Box: Q1-Q3 | Line: Median | Dot: Mean | Whiskers: P5-P95', pad.left, pad.top - 10);
  }

  // ── Correlation Matrix ──────────────────────────────────────────────
  function renderCorrelation() {
    var canvas = document.getElementById('chart-correlation');
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.parentElement.clientWidth;
    var h = 300;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    var matrix = stats.correlation.matrix;
    var labels = stats.correlation.labels;
    var n = labels.length;

    var pad = { top: 50, bottom: 20, left: 70, right: 50 };
    var size = Math.min(w - pad.left - pad.right, h - pad.top - pad.bottom);
    var cellSize = size / n;
    var ox = pad.left;
    var oy = pad.top;

    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var val = matrix[i][j];
        var cx = ox + j * cellSize;
        var cy = oy + i * cellSize;

        if (val >= 0) {
          ctx.fillStyle = 'rgba(27, 79, 114, ' + (val * 0.8 + 0.1) + ')';
        } else {
          ctx.fillStyle = 'rgba(231, 76, 60, ' + (-val * 0.8 + 0.1) + ')';
        }

        ctx.beginPath();
        ctx.roundRect(cx + 2, cy + 2, cellSize - 4, cellSize - 4, 4);
        ctx.fill();

        ctx.fillStyle = Math.abs(val) > 0.4 ? '#fff' : '#1a1a2e';
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val.toFixed(2), cx + cellSize / 2, cy + cellSize / 2);
      }
    }

    // Labels
    ctx.fillStyle = '#1a1a2e';
    ctx.font = '12px "Source Sans 3", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (var i = 0; i < n; i++) {
      ctx.fillStyle = COLORS[labels[i]];
      ctx.font = 'bold 12px "Source Sans 3", sans-serif';
      ctx.fillText(labels[i], ox - 10, oy + i * cellSize + cellSize / 2);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    for (var j = 0; j < n; j++) {
      ctx.fillStyle = COLORS[labels[j]];
      ctx.font = 'bold 12px "Source Sans 3", sans-serif';
      ctx.fillText(labels[j], ox + j * cellSize + cellSize / 2, oy - 10);
    }
  }

  // ── Spatial Mesh ────────────────────────────────────────────────────
  function renderSpatial() {
    var canvas = document.getElementById('chart-spatial');
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.parentElement.clientWidth;
    var h = 260;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    var xs = stats.spatial_x;
    var ys = stats.spatial_y;
    var xRange = stats.coords.x_range;
    var yRange = stats.coords.y_range;

    var pad = { top: 20, bottom: 35, left: 55, right: 20 };
    var cw = w - pad.left - pad.right;
    var ch = h - pad.top - pad.bottom;

    // Background
    ctx.fillStyle = '#fafbfc';
    ctx.fillRect(pad.left, pad.top, cw, ch);

    // Points
    ctx.fillStyle = '#1B4F72';
    ctx.globalAlpha = 0.35;
    for (var i = 0; i < xs.length; i++) {
      var px = pad.left + ((xs[i] - xRange[0]) / (xRange[1] - xRange[0])) * cw;
      var py = pad.top + ch - ((ys[i] - yRange[0]) / (yRange[1] - yRange[0])) * ch;
      ctx.beginPath();
      ctx.arc(px, py, 1.8, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Axes
    ctx.strokeStyle = '#cbd5e0';
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + ch);
    ctx.lineTo(pad.left + cw, pad.top + ch);
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + ch);
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#4a5568';
    ctx.font = '11px "Source Sans 3", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('x coordinate', pad.left + cw / 2, h - 4);
    ctx.save();
    ctx.translate(14, pad.top + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('y coordinate', 0, 0);
    ctx.restore();

    // Ticks
    ctx.font = '9px monospace';
    ctx.fillStyle = '#718096';
    ctx.textAlign = 'center';
    ctx.fillText(xRange[0].toFixed(3), pad.left, h - 18);
    ctx.fillText(xRange[1].toFixed(3), pad.left + cw, h - 18);
    ctx.textAlign = 'right';
    ctx.fillText(yRange[0].toFixed(3), pad.left - 6, pad.top + ch);
    ctx.fillText(yRange[1].toFixed(3), pad.left - 6, pad.top + 8);
  }

  // ── TKE Detail ─────────────────────────────────────────────────────
  function renderTKEDetail() {
    var container = document.getElementById('tke-detail');
    var d = stats.tke_detail;
    var s = stats.TKE;

    container.innerHTML = ''
      + '<div class="tke-stats-grid">'
      + '<div class="tke-stat"><span class="tke-stat-value">' + d.below_001.toFixed(1) + '%</span><span class="tke-stat-label">TKE &lt; 0.01</span></div>'
      + '<div class="tke-stat"><span class="tke-stat-value">' + d.below_005.toFixed(1) + '%</span><span class="tke-stat-label">TKE &lt; 0.05</span></div>'
      + '<div class="tke-stat"><span class="tke-stat-value">' + d.below_010.toFixed(1) + '%</span><span class="tke-stat-label">TKE &lt; 0.10</span></div>'
      + '<div class="tke-stat"><span class="tke-stat-value">' + d.above_050.toFixed(1) + '%</span><span class="tke-stat-label">TKE &gt; 0.50</span></div>'
      + '</div>'
      + '<p class="tke-note">TKE is heavily right-skewed (skewness = <strong>' + s.skewness.toFixed(2) + '</strong>, kurtosis = <strong>' + s.kurtosis.toFixed(2) + '</strong>). '
      + 'Standard MSE optimization is dominated by the ' + d.below_001.toFixed(0) + '% of near-zero samples, '
      + 'making rare high-TKE events harder to reconstruct accurately.</p>';
  }

  // ── Temporal Trends ─────────────────────────────────────────────────
  function renderTemporal() {
    var canvas = document.getElementById('chart-temporal');
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.parentElement.clientWidth;
    var h = 220;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    var temporal = stats.temporal;
    var fields = ['Vx', 'Vy', 'Pressure', 'TKE'];
    var indices = stats.temporal_indices;
    var numPts = indices.length;

    var pad = { top: 20, bottom: 36, left: 50, right: 120 };
    var cw = w - pad.left - pad.right;
    var ch = h - pad.top - pad.bottom;

    // Grid
    ctx.strokeStyle = '#f0f2f5';
    for (var g = 0; g <= 4; g++) {
      var gy = pad.top + ch - (g / 4) * ch;
      ctx.beginPath();
      ctx.moveTo(pad.left, gy);
      ctx.lineTo(pad.left + cw, gy);
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#e0e4ed';
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + ch);
    ctx.lineTo(pad.left + cw, pad.top + ch);
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + ch);
    ctx.stroke();

    fields.forEach(function (name) {
      var means = temporal[name].means;
      var stds = temporal[name].stds;
      var color = COLORS[name];

      // Std band
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.08;
      ctx.beginPath();
      for (var i = 0; i < numPts; i++) {
        var px = pad.left + (i / (numPts - 1)) * cw;
        var py = pad.top + ch - (means[i] + stds[i]) * ch;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      for (var i = numPts - 1; i >= 0; i--) {
        var px = pad.left + (i / (numPts - 1)) * cw;
        var py = pad.top + ch - (means[i] - stds[i]) * ch;
        ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;

      // Mean line
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (var i = 0; i < numPts; i++) {
        var px = pad.left + (i / (numPts - 1)) * cw;
        var py = pad.top + ch - means[i] * ch;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.lineWidth = 1;
    });

    // Legend
    var legendX = w - pad.right + 16;
    fields.forEach(function (name, i) {
      var ly = pad.top + 8 + i * 22;
      ctx.fillStyle = COLORS[name];
      ctx.fillRect(legendX, ly, 14, 14);
      ctx.fillStyle = '#1a1a2e';
      ctx.font = '12px "Source Sans 3", sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(name, legendX + 20, ly + 7);
    });

    // Axis labels
    ctx.fillStyle = '#4a5568';
    ctx.font = '11px "Source Sans 3", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Timestep Index', pad.left + cw / 2, h - 4);

    ctx.font = '9px monospace';
    ctx.fillStyle = '#718096';
    ctx.textAlign = 'center';
    ctx.fillText('0', pad.left, h - 18);
    ctx.fillText(String(indices[numPts - 1]), pad.left + cw, h - 18);
    ctx.textAlign = 'right';
    ctx.fillText('0.0', pad.left - 6, pad.top + ch + 4);
    ctx.fillText('1.0', pad.left - 6, pad.top + 4);
  }

  // ── Coordinate Space Summary ────────────────────────────────────────
  function renderCoordSummary() {
    var container = document.getElementById('coord-summary');
    var c = stats.coords;

    container.innerHTML = ''
      + '<table class="stats-table">'
      + '<thead><tr><th>Coordinate</th><th>Min</th><th>Max</th><th>Range</th><th>Notes</th></tr></thead>'
      + '<tbody>'
      + '<tr><td class="var-name">x</td><td>' + c.x_range[0].toFixed(5) + '</td><td>' + c.x_range[1].toFixed(5) + '</td><td>' + (c.x_range[1] - c.x_range[0]).toFixed(5) + '</td><td>Streamwise direction</td></tr>'
      + '<tr><td class="var-name">y</td><td>' + c.y_range[0].toFixed(5) + '</td><td>' + c.y_range[1].toFixed(5) + '</td><td>' + (c.y_range[1] - c.y_range[0]).toFixed(5) + '</td><td>Cross-stream direction</td></tr>'
      + '<tr><td class="var-name">z</td><td>0.0</td><td>0.0</td><td>0.0</td><td>Single plane (2D slice)</td></tr>'
      + '<tr><td class="var-name">t</td><td>' + c.time_range[0].toFixed(5) + '</td><td>' + c.time_range[1].toFixed(5) + '</td><td>' + (c.time_range[1] - c.time_range[0]).toFixed(5) + '</td><td>' + c.n_timesteps + ' timesteps</td></tr>'
      + '</tbody></table>';
  }

  document.addEventListener('DOMContentLoaded', init);
})();
