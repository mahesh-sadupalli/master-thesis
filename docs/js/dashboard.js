/**
 * dashboard.js — Plotly-based interactive data dashboard.
 * MedSat-style layout with page navigation.
 */
(function () {
  'use strict';

  var COLORS = {
    Vx: '#1B4F72',
    Vy: '#E74C3C',
    Pressure: '#27AE60',
    TKE: '#8E44AD',
  };

  var MODEL_COLORS = {
    Base: '#636EFA',
    Medium: '#EF553B',
    Large: '#00CC96',
  };

  var APPROACH_COLORS = {
    'Batch (Offline)': '#636EFA',
    'Online (Naive)': '#EF553B',
    'CL Boosted': '#00CC96',
  };

  var PLOTLY_LAYOUT = {
    font: { family: "'Source Sans 3', 'Inter', sans-serif", size: 14 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 50, r: 20, t: 10, b: 50 },
  };

  var stats = null;
  var manifest = null;
  var compressedStats = null;
  var fields = ['Vx', 'Vy', 'Pressure', 'TKE'];

  var COMPRESSED_LABELS = {
    'offline_base': 'Batch Base',
    'offline_medium': 'Batch Medium',
    'offline_large': 'Batch Large',
    'online_naive_base': 'Online Naive Base',
    'boosted_base': 'CL Boosted Base',
    'boosted_medium': 'CL Boosted Medium',
    'boosted_large': 'CL Boosted Large',
  };

  var COMPRESSED_COLORS = {
    'offline_base': '#85C1E9',
    'offline_medium': '#5DADE2',
    'offline_large': '#2E86C1',
    'online_naive_base': '#EF553B',
    'boosted_base': '#82E0AA',
    'boosted_medium': '#27AE60',
    'boosted_large': '#1E8449',
  };

  // ── Page Navigation ─────────────────────────────────────────────────
  window.switchPage = function (pageId) {
    document.querySelectorAll('.dash-page').forEach(function (el) {
      el.classList.remove('active');
    });
    var target = document.getElementById('page-' + pageId);
    if (target) {
      target.classList.add('active');
      // Render on first visit, then resize Plotly charts
      if (!target.dataset.rendered) {
        target.dataset.rendered = '1';
        // Use setTimeout so the container is visible before Plotly measures it
        setTimeout(function () { renderPage(pageId); }, 50);
      } else {
        // Resize existing charts in case container was hidden
        target.querySelectorAll('.chart-container').forEach(function (el) {
          if (el.querySelector('.plotly')) Plotly.Plots.resize(el);
        });
      }
    }
  };

  function renderPage(pageId) {
    switch (pageId) {
      case 'inputdist': renderInputDistPage(); break;
      case 'distributions': renderDistributionsPage(); break;
      case 'correlations': renderCorrelationsPage(); break;
      case 'temporal': renderTemporalPage(); break;
      case 'compression': renderCompressionPage(); break;
      case 'comparison': renderComparisonPage(); break;
      case 'distcompare': renderDistComparePage(); break;
      case 'training': renderTrainingPage(); break;
    }
  }

  // ── Init ──────────────────────────────────────────────────────────────
  async function init() {
    var [statsResp, manifestResp, compResp] = await Promise.all([
      fetch('data/dataset_stats.json'),
      fetch('data/manifest.json'),
      fetch('data/compressed_stats.json'),
    ]);
    stats = await statsResp.json();
    manifest = await manifestResp.json();
    try { compressedStats = await compResp.json(); } catch (e) { compressedStats = null; }

    renderOverviewPage();
    // Mark overview as rendered
    document.getElementById('page-overview').dataset.rendered = '1';
  }

  // ══════════════════════════════════════════════════════════════════════
  // OVERVIEW PAGE
  // ══════════════════════════════════════════════════════════════════════
  function renderOverviewPage() {
    renderOverviewMetrics();
    renderRadarCheckboxes();
    renderFieldRadar();
    renderQualityBar();
    renderDescriptiveStats();
    renderOverviewCorrelation();
  }

  function renderOverviewMetrics() {
    var container = document.getElementById('overview-metrics');
    var items = [
      { value: '7,919,100', label: 'Total Samples' },
      { value: '26,397', label: 'Points / Timestep' },
      { value: '120.8 MB', label: 'Dataset Size' },
      { value: manifest.metrics.offline_large.psnr.toFixed(2) + ' dB', label: 'Best PSNR (Offline)' },
      { value: manifest.metrics.offline_base.cr.toLocaleString() + ':1', label: 'Best Compression Ratio' },
    ];
    container.innerHTML = items.map(function (item) {
      return '<div class="metric-card"><span class="metric-value">' + item.value +
             '</span><span class="metric-label">' + item.label + '</span></div>';
    }).join('');
  }

  // ── Radar Chart ─────────────────────────────────────────────────────
  var radarActive = { Vx: true, Vy: true, Pressure: false, TKE: true };

  function renderRadarCheckboxes() {
    var container = document.getElementById('radar-checks');
    container.innerHTML = fields.map(function (f) {
      return '<button class="check-btn' + (radarActive[f] ? ' active' : '') +
             '" data-field="' + f + '">' + f + '</button>';
    }).join('');
    container.querySelectorAll('.check-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var f = btn.dataset.field;
        radarActive[f] = !radarActive[f];
        btn.classList.toggle('active');
        renderFieldRadar();
      });
    });
  }

  function renderFieldRadar() {
    var radarMetrics = ['mean', 'median', 'std', 'skewness', 'iqr'];
    var radarLabels = ['Mean', 'Median', 'Std Dev', 'Skewness', 'IQR'];

    // Normalize each metric across fields to [0, 1]
    var normalized = {};
    radarMetrics.forEach(function (m) {
      var vals = fields.map(function (f) { return Math.abs(stats[f][m]); });
      var mn = Math.min.apply(null, vals);
      var mx = Math.max.apply(null, vals);
      var rng = mx - mn || 1;
      normalized[m] = vals.map(function (v) { return (v - mn) / rng; });
    });

    var traces = [];
    fields.forEach(function (f, fi) {
      if (!radarActive[f]) return;
      var r = radarMetrics.map(function (m) { return normalized[m][fi]; });
      r.push(r[0]);
      traces.push({
        type: 'scatterpolar',
        r: r,
        theta: radarLabels.concat([radarLabels[0]]),
        fill: 'toself',
        name: f,
        line: { color: COLORS[f] },
        opacity: 0.7,
      });
    });

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      polar: { radialaxis: { visible: true, range: [0, 1], showticklabels: false } },
      height: 400,
      legend: { orientation: 'h', y: -0.1 },
      margin: { l: 40, r: 40, t: 20, b: 40 },
    });

    Plotly.react('chart-radar', traces, layout, { responsive: true, displayModeBar: false });
  }

  // ── Quality Bar ─────────────────────────────────────────────────────
  window.renderQualityBar = function () {
    var metric = document.getElementById('quality-metric-select').value;
    var metricLabels = { psnr: 'PSNR (dB)', ssim: 'SSIM', rel_error: 'Relative Error (%)' };
    var models = ['Base', 'Medium', 'Large'];
    var approaches = [
      { label: 'Batch (Offline)', prefix: 'offline' },
      { label: 'Online (Naive)', prefix: 'online' },
      { label: 'CL Boosted', prefix: 'boosted' },
    ];

    var traces = approaches.map(function (a) {
      return {
        type: 'bar',
        name: a.label,
        y: models,
        x: models.map(function (m) {
          return manifest.metrics[a.prefix + '_' + m.toLowerCase()][metric];
        }),
        orientation: 'h',
        marker: { color: APPROACH_COLORS[a.label] },
      };
    });

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 400,
      barmode: 'group',
      xaxis: { title: metricLabels[metric] },
      legend: { orientation: 'h', y: -0.15 },
      margin: { l: 70, r: 20, t: 10, b: 50 },
    });

    Plotly.react('chart-quality-bar', traces, layout, { responsive: true, displayModeBar: false });
  };

  // ── Descriptive Stats Table ─────────────────────────────────────────
  function renderDescriptiveStats() {
    var tbody = document.getElementById('desc-stats-body');
    tbody.innerHTML = fields.map(function (f) {
      var s = stats[f];
      return '<tr>'
        + '<td class="var-name" style="color:' + COLORS[f] + '">' + f + '</td>'
        + '<td>' + s.count.toLocaleString() + '</td>'
        + '<td>' + s.mean.toFixed(4) + '</td>'
        + '<td>' + s.median.toFixed(4) + '</td>'
        + '<td>' + s.std.toFixed(4) + '</td>'
        + '<td>' + s.skewness.toFixed(3) + '</td>'
        + '<td>' + s.kurtosis.toFixed(3) + '</td>'
        + '<td>' + s.q1.toFixed(4) + '</td>'
        + '<td>' + s.q3.toFixed(4) + '</td>'
        + '<td>' + s.iqr.toFixed(4) + '</td>'
        + '</tr>';
    }).join('');
  }

  // ── Overview Correlation Heatmap ────────────────────────────────────
  function renderOverviewCorrelation() {
    renderCorrelationHeatmap('chart-corr-overview', '.2f');
  }

  function renderCorrelationHeatmap(containerId, textFmt) {
    var matrix = stats.correlation.matrix;
    var labels = stats.correlation.labels;
    var trace = {
      type: 'heatmap',
      z: matrix,
      x: labels,
      y: labels,
      colorscale: 'RdBu',
      reversescale: true,
      zmin: -1,
      zmax: 1,
      text: matrix.map(function (row) {
        return row.map(function (v) { return v.toFixed(textFmt === '.3f' ? 3 : 2); });
      }),
      texttemplate: '%{text}',
      hovertemplate: '%{y} vs %{x}: %{z:.3f}<extra></extra>',
    };

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 400,
      margin: { l: 80, r: 20, t: 10, b: 50 },
      xaxis: { side: 'bottom' },
    });

    Plotly.react(containerId, [trace], layout, { responsive: true, displayModeBar: false });
  }

  // ══════════════════════════════════════════════════════════════════════
  // INPUT DISTRIBUTIONS PAGE
  // ══════════════════════════════════════════════════════════════════════
  var inputCoords = ['x', 'y', 't']; // skip z (constant)
  var inputColors = { x: '#0A2540', y: '#065A82', z: '#999999', t: '#1C7293' };
  var inputLabels = { x: 'x \u2014 Streamwise', y: 'y \u2014 Cross-stream', z: 'z \u2014 Out-of-plane', t: 't \u2014 Time' };

  function renderInputDistPage() {
    renderInputHistogram();
    renderAllInputDistributions();
    renderInputStatsTable();
    renderTimeInfo();
  }

  function histCenters(hist) {
    var c = [];
    for (var i = 0; i < hist.counts.length; i++) {
      c.push((hist.edges[i] + hist.edges[i + 1]) / 2);
    }
    return c;
  }

  window.renderInputHistogram = function () {
    var coord = document.getElementById('input-coord-select').value;

    // Raw histogram (before normalization)
    var rawHist = stats[coord + '_raw_hist'];
    if (rawHist && rawHist.edges[0] !== rawHist.edges[rawHist.edges.length - 1]) {
      var rawCenters = histCenters(rawHist);
      Plotly.react('chart-input-raw', [{
        type: 'bar', x: rawCenters, y: rawHist.counts,
        marker: { color: inputColors[coord], opacity: 0.75 }, name: coord,
      }], Object.assign({}, PLOTLY_LAYOUT, {
        height: 380,
        xaxis: { title: 'Physical Value' },
        yaxis: { title: 'Count' },
        bargap: 0.02,
      }), { responsive: true, displayModeBar: false });
    }

    // Normalized histogram (after normalization)
    var normHist = stats[coord + '_hist'];
    if (normHist) {
      var normCenters = histCenters(normHist);
      Plotly.react('chart-input-norm', [{
        type: 'bar', x: normCenters, y: normHist.counts,
        marker: { color: inputColors[coord], opacity: 0.75 }, name: coord,
      }], Object.assign({}, PLOTLY_LAYOUT, {
        height: 380,
        xaxis: { title: 'Normalized [0, 1]' },
        yaxis: { title: 'Count' },
        bargap: 0.02,
      }), { responsive: true, displayModeBar: false });
    }

    // Metrics
    var s = stats[coord];
    if (!s) return;
    var container = document.getElementById('input-hist-metrics');
    var rawRange = s.raw_min !== undefined ? '[' + s.raw_min.toFixed(4) + ', ' + s.raw_max.toFixed(4) + ']' : '--';
    var rawMean = s.raw_mean !== undefined ? s.raw_mean.toFixed(6) : '--';
    var items = [
      { value: rawRange, label: 'Raw Physical Range' },
      { value: rawMean, label: 'Raw Mean' },
      { value: s.mean.toFixed(4), label: 'Normalized Mean' },
      { value: s.std.toFixed(4), label: 'Normalized Std' },
      { value: s.skewness.toFixed(3), label: 'Skewness' },
    ];
    container.innerHTML = items.map(function (item) {
      return '<div class="metric-card"><span class="metric-value">' + item.value +
             '</span><span class="metric-label">' + item.label + '</span></div>';
    }).join('');
  };

  function renderAllInputDistributions() {
    // Raw overlay
    var rawTraces = inputCoords.map(function (c) {
      var hist = stats[c + '_raw_hist'];
      if (!hist || hist.edges[0] === hist.edges[hist.edges.length - 1]) return null;
      var total = hist.counts.reduce(function (a, b) { return a + b; }, 0);
      var centers = histCenters(hist);
      var freq = hist.counts.map(function (v) { return v / total; });
      return {
        type: 'scatter', x: centers, y: freq,
        mode: 'lines', name: inputLabels[c],
        line: { color: inputColors[c], width: 2 },
        fill: 'tozeroy', fillcolor: inputColors[c] + '18',
      };
    }).filter(Boolean);

    Plotly.react('chart-all-inputs-raw', rawTraces, Object.assign({}, PLOTLY_LAYOUT, {
      height: 380,
      xaxis: { title: 'Physical Value' },
      yaxis: { title: 'Relative Frequency' },
      legend: { orientation: 'h', y: -0.18 },
    }), { responsive: true, displayModeBar: false });

    // Normalized overlay
    var normTraces = inputCoords.map(function (c) {
      var hist = stats[c + '_hist'];
      if (!hist) return null;
      var total = hist.counts.reduce(function (a, b) { return a + b; }, 0);
      var centers = histCenters(hist);
      var freq = hist.counts.map(function (v) { return v / total; });
      return {
        type: 'scatter', x: centers, y: freq,
        mode: 'lines', name: inputLabels[c],
        line: { color: inputColors[c], width: 2 },
        fill: 'tozeroy', fillcolor: inputColors[c] + '18',
      };
    }).filter(Boolean);

    Plotly.react('chart-all-inputs-norm', normTraces, Object.assign({}, PLOTLY_LAYOUT, {
      height: 380,
      xaxis: { title: 'Normalized [0, 1]' },
      yaxis: { title: 'Relative Frequency' },
      legend: { orientation: 'h', y: -0.18 },
    }), { responsive: true, displayModeBar: false });
  }

  function renderInputStatsTable() {
    var tbody = document.getElementById('input-stats-body');
    var allCoords = ['x', 'y', 'z', 't'];
    var notes = {
      x: 'Mesh denser near cylinder (right-skewed)',
      y: 'Symmetric around 0',
      z: 'Constant = 0 (2D simulation slice)',
      t: 't\u2080=0, t\u2081=0.010, then \u0394t\u22480.0001',
    };
    tbody.innerHTML = allCoords.map(function (c) {
      var s = stats[c];
      if (!s) return '';
      var rawRange = s.constant ? '0.000' : '[' + s.raw_min.toFixed(4) + ', ' + s.raw_max.toFixed(4) + ']';
      var rawMean = s.raw_mean !== undefined ? s.raw_mean.toFixed(6) : '--';
      return '<tr>'
        + '<td class="var-name" style="color:' + inputColors[c] + '">' + c + '</td>'
        + '<td>' + rawRange + '</td>'
        + '<td>' + rawMean + '</td>'
        + '<td>' + s.mean.toFixed(4) + '</td>'
        + '<td>' + s.median.toFixed(4) + '</td>'
        + '<td>' + s.std.toFixed(4) + '</td>'
        + '<td>' + s.skewness.toFixed(3) + '</td>'
        + '<td>' + (notes[c] || '') + '</td>'
        + '</tr>';
    }).join('');
  }

  function renderTimeInfo() {
    var info = stats.t_info;
    if (!info) return;
    document.getElementById('time-info').innerHTML =
      '<strong>300 timesteps</strong> in the dataset. ' +
      'First timestep t\u2080 = 0.000, second t\u2081 = 0.0101 ' +
      '(<strong>initial gap of ' + info.first_gap + ' s</strong>), ' +
      'then \u0394t \u2248 ' + info.regular_gap + ' for the remaining 298 timesteps. ' +
      'The large first gap means the normalized time coordinate is not uniformly distributed \u2014 ' +
      'the first 26,397 samples (t=0) map to normalized t=0, while the remaining 299 timesteps ' +
      'are compressed into the [0.253, 1.0] range.';
  }

  // ══════════════════════════════════════════════════════════════════════
  // OUTPUT DISTRIBUTIONS PAGE
  // ══════════════════════════════════════════════════════════════════════
  function renderDistributionsPage() {
    renderHistogram();
    renderAllDistributions();
    renderBoxPlots();
    renderTKEDetail();
  }

  window.renderHistogram = function () {
    var field = document.getElementById('hist-field-select').value;
    var hist = stats[field + '_hist'];
    var centers = [];
    for (var i = 0; i < hist.counts.length; i++) {
      centers.push((hist.edges[i] + hist.edges[i + 1]) / 2);
    }

    var trace = {
      type: 'bar',
      x: centers,
      y: hist.counts,
      marker: { color: COLORS[field], opacity: 0.75 },
      name: field,
    };

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 400,
      xaxis: { title: 'Normalized Value' },
      yaxis: { title: 'Count' },
      bargap: 0.02,
    });

    Plotly.react('chart-histogram', [trace], layout, { responsive: true, displayModeBar: false });

    // Metrics
    var s = stats[field];
    var container = document.getElementById('hist-metrics');
    var items = [
      { value: s.mean.toFixed(4), label: 'Mean' },
      { value: s.median.toFixed(4), label: 'Median' },
      { value: s.skewness.toFixed(3), label: 'Skewness' },
      { value: s.kurtosis.toFixed(3), label: 'Kurtosis' },
    ];
    container.innerHTML = items.map(function (item) {
      return '<div class="metric-card"><span class="metric-value">' + item.value +
             '</span><span class="metric-label">' + item.label + '</span></div>';
    }).join('');
  };

  function renderAllDistributions() {
    var traces = fields.map(function (f) {
      var hist = stats[f + '_hist'];
      var total = hist.counts.reduce(function (a, b) { return a + b; }, 0);
      var centers = [];
      var freq = [];
      for (var i = 0; i < hist.counts.length; i++) {
        centers.push((hist.edges[i] + hist.edges[i + 1]) / 2);
        freq.push(hist.counts[i] / total);
      }
      return {
        type: 'scatter',
        x: centers,
        y: freq,
        mode: 'lines',
        name: f,
        line: { color: COLORS[f], width: 2 },
        fill: 'tozeroy',
        fillcolor: COLORS[f] + '18',
      };
    });

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 400,
      xaxis: { title: 'Normalized Value [0, 1]' },
      yaxis: { title: 'Relative Frequency' },
      legend: { orientation: 'h', y: -0.15 },
    });

    Plotly.react('chart-all-distributions', traces, layout, { responsive: true, displayModeBar: false });
  }

  function renderBoxPlots() {
    var traces = fields.map(function (f) {
      var s = stats[f];
      return {
        type: 'box',
        name: f,
        lowerfence: [s.p5],
        q1: [s.q1],
        median: [s.median],
        q3: [s.q3],
        upperfence: [s.p95],
        mean: [s.mean],
        boxmean: true,
        marker: { color: COLORS[f] },
        line: { color: COLORS[f] },
        fillcolor: COLORS[f] + '40',
      };
    });

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 420,
      showlegend: false,
      yaxis: { title: 'Normalized Value [0, 1]' },
    });

    Plotly.react('chart-boxplots', traces, layout, { responsive: true, displayModeBar: false });
  }

  function renderTKEDetail() {
    var tke = stats.tke_detail;
    var s = stats.TKE;

    var container = document.getElementById('tke-metrics');
    var items = [
      { value: tke.below_001.toFixed(1) + '%', label: 'TKE < 0.01' },
      { value: tke.below_005.toFixed(1) + '%', label: 'TKE < 0.05' },
      { value: tke.below_010.toFixed(1) + '%', label: 'TKE < 0.10' },
      { value: tke.above_050.toFixed(1) + '%', label: 'TKE > 0.50' },
    ];
    container.innerHTML = items.map(function (item) {
      return '<div class="metric-card"><span class="metric-value" style="color:#8E44AD">' + item.value +
             '</span><span class="metric-label">' + item.label + '</span></div>';
    }).join('');

    document.getElementById('tke-info').innerHTML =
      'TKE is heavily right-skewed (skewness = <strong>' + s.skewness.toFixed(2) + '</strong>, ' +
      'kurtosis = <strong>' + s.kurtosis.toFixed(2) + '</strong>). Standard MSE optimization is dominated ' +
      'by the ' + tke.below_001.toFixed(0) + '% of near-zero samples, making rare high-TKE events ' +
      'harder to reconstruct accurately.';
  }

  // ══════════════════════════════════════════════════════════════════════
  // CORRELATIONS PAGE
  // ══════════════════════════════════════════════════════════════════════
  function renderCorrelationsPage() {
    renderCorrelationHeatmap('chart-corr-full', '.3f');
    renderPairwiseCorrelations();
  }

  function renderPairwiseCorrelations() {
    var matrix = stats.correlation.matrix;
    var labels = stats.correlation.labels;
    var pairs = [];
    for (var i = 0; i < labels.length; i++) {
      for (var j = i + 1; j < labels.length; j++) {
        pairs.push({ pair: labels[i] + ' vs ' + labels[j], r: matrix[i][j] });
      }
    }
    pairs.sort(function (a, b) { return a.r - b.r; });

    var trace = {
      type: 'bar',
      y: pairs.map(function (p) { return p.pair; }),
      x: pairs.map(function (p) { return p.r; }),
      orientation: 'h',
      marker: {
        color: pairs.map(function (p) { return p.r < 0 ? '#E74C3C' : '#3498db'; }),
      },
      text: pairs.map(function (p) { return (p.r >= 0 ? '+' : '') + p.r.toFixed(3); }),
      textposition: 'outside',
    };

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 350,
      xaxis: { title: 'Pearson Correlation' },
      margin: { l: 130, r: 80, t: 10, b: 50 },
      shapes: [{ type: 'line', x0: 0, x1: 0, y0: -0.5, y1: pairs.length - 0.5,
                 line: { dash: 'dash', color: 'gray' } }],
    });

    Plotly.react('chart-corr-pairs', [trace], layout, { responsive: true, displayModeBar: false });
  }

  // ══════════════════════════════════════════════════════════════════════
  // TEMPORAL PAGE
  // ══════════════════════════════════════════════════════════════════════
  var temporalActive = { Vx: true, Vy: true, Pressure: true, TKE: true };

  function renderTemporalPage() {
    renderTemporalCheckboxes();
    renderTemporalChart();
    renderTemporalStats();
    renderSpatialCloud();
  }

  function renderTemporalCheckboxes() {
    var container = document.getElementById('temporal-checks');
    container.innerHTML = fields.map(function (f) {
      return '<button class="check-btn' + (temporalActive[f] ? ' active' : '') +
             '" data-field="' + f + '">' + f + '</button>';
    }).join('');
    container.querySelectorAll('.check-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var f = btn.dataset.field;
        temporalActive[f] = !temporalActive[f];
        btn.classList.toggle('active');
        renderTemporalChart();
      });
    });
  }

  function renderTemporalChart() {
    var temporal = stats.temporal;
    var indices = stats.temporal_indices;
    var traces = [];

    fields.forEach(function (f) {
      if (!temporalActive[f]) return;
      var means = temporal[f].means;
      var stds = temporal[f].stds;
      var upper = means.map(function (m, i) { return m + stds[i]; });
      var lower = means.map(function (m, i) { return m - stds[i]; });

      // Std band
      traces.push({
        type: 'scatter',
        x: indices.concat(indices.slice().reverse()),
        y: upper.concat(lower.slice().reverse()),
        fill: 'toself',
        fillcolor: COLORS[f] + '15',
        line: { color: 'rgba(0,0,0,0)' },
        showlegend: false,
        hoverinfo: 'skip',
      });
      // Mean line
      traces.push({
        type: 'scatter',
        x: indices,
        y: means,
        mode: 'lines',
        name: f,
        line: { color: COLORS[f], width: 2.5 },
      });
    });

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 480,
      xaxis: { title: 'Timestep Index' },
      yaxis: { title: 'Normalized Value', range: [0, 1] },
      legend: { orientation: 'h', y: -0.12 },
    });

    Plotly.react('chart-temporal', traces, layout, { responsive: true, displayModeBar: false });
  }

  function renderTemporalStats() {
    var temporal = stats.temporal;
    var tbody = document.getElementById('temporal-stats-body');
    tbody.innerHTML = fields.map(function (f) {
      var means = temporal[f].means;
      var stds = temporal[f].stds;
      var avgMean = means.reduce(function (a, b) { return a + b; }, 0) / means.length;
      var minMean = Math.min.apply(null, means);
      var maxMean = Math.max.apply(null, means);
      var avgStd = stds.reduce(function (a, b) { return a + b; }, 0) / stds.length;
      return '<tr>'
        + '<td class="var-name" style="color:' + COLORS[f] + '">' + f + '</td>'
        + '<td>' + avgMean.toFixed(4) + '</td>'
        + '<td>' + minMean.toFixed(4) + '</td>'
        + '<td>' + maxMean.toFixed(4) + '</td>'
        + '<td>' + avgStd.toFixed(4) + '</td>'
        + '<td>' + (maxMean - minMean).toFixed(4) + '</td>'
        + '</tr>';
    }).join('');
  }

  function renderSpatialCloud() {
    var trace = {
      type: 'scatter',
      x: stats.spatial_x,
      y: stats.spatial_y,
      mode: 'markers',
      marker: { size: 3, color: '#1B4F72', opacity: 0.4 },
      hoverinfo: 'x+y',
    };

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 350,
      xaxis: { title: 'x coordinate', scaleanchor: 'y' },
      yaxis: { title: 'y coordinate' },
    });

    Plotly.react('chart-spatial', [trace], layout, { responsive: true, displayModeBar: false });
  }

  // ══════════════════════════════════════════════════════════════════════
  // COMPRESSION RESULTS PAGE
  // ══════════════════════════════════════════════════════════════════════
  function renderCompressionPage() {
    renderCompressionTable();
    renderDumbbellChart();
    renderGapChart();
  }

  function renderCompressionTable() {
    var approaches = [
      { label: 'Batch (Offline)', prefix: 'offline' },
      { label: 'Online (Naive)', prefix: 'online' },
      { label: 'CL Boosted (ER Aggressive)', prefix: 'boosted' },
    ];
    var models = ['base', 'medium', 'large'];
    var modelLabels = { base: 'Base', medium: 'Medium', large: 'Large' };
    var tbody = document.getElementById('compression-body');

    var rows = [];
    approaches.forEach(function (a) {
      models.forEach(function (mod) {
        var m = manifest.metrics[a.prefix + '_' + mod];
        rows.push('<tr>'
          + '<td class="approach-cell">' + a.label + '</td>'
          + '<td class="var-name">' + modelLabels[mod] + '</td>'
          + '<td>' + m.psnr.toFixed(2) + '</td>'
          + '<td>' + m.ssim.toFixed(4) + '</td>'
          + '<td>' + m.rel_error.toFixed(2) + '</td>'
          + '<td>' + m.params.toLocaleString() + '</td>'
          + '<td>' + m.size_kb.toFixed(1) + '</td>'
          + '<td>' + m.cr.toLocaleString() + ':1</td>'
          + '</tr>');
      });
    });
    tbody.innerHTML = rows.join('');
  }

  function renderDumbbellChart() {
    var models = ['Base', 'Medium', 'Large'];
    var traces = [];

    // Lines connecting naive to offline
    models.forEach(function (mod) {
      var naive = manifest.metrics['online_' + mod.toLowerCase()].psnr;
      var offline = manifest.metrics['offline_' + mod.toLowerCase()].psnr;
      traces.push({
        type: 'scatter',
        x: [naive, offline],
        y: [mod, mod],
        mode: 'lines',
        line: { color: '#bbb', width: 3 },
        showlegend: false,
        hoverinfo: 'skip',
      });
    });

    // Markers
    var addMarkers = function (prefix, name, color, symbol) {
      traces.push({
        type: 'scatter',
        x: models.map(function (m) { return manifest.metrics[prefix + '_' + m.toLowerCase()].psnr; }),
        y: models,
        mode: 'markers',
        name: name,
        marker: { size: 14, color: color, symbol: symbol },
      });
    };

    addMarkers('online', 'Online (Naive)', '#EF553B', 'circle');
    addMarkers('boosted', 'CL Boosted (ER Aggressive)', '#00CC96', 'diamond');
    addMarkers('offline', 'Batch (Offline)', '#636EFA', 'square');

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 350,
      xaxis: { title: 'PSNR (dB)' },
      legend: { orientation: 'h', y: 1.08 },
      margin: { l: 70, r: 20, t: 40, b: 50 },
    });

    Plotly.react('chart-dumbbell', traces, layout, { responsive: true, displayModeBar: false });
  }

  function renderGapChart() {
    var models = ['Base', 'Medium', 'Large'];

    var naiveGaps = models.map(function (m) {
      return manifest.metrics['online_' + m.toLowerCase()].psnr - manifest.metrics['offline_' + m.toLowerCase()].psnr;
    });
    var boostedGaps = models.map(function (m) {
      return manifest.metrics['boosted_' + m.toLowerCase()].psnr - manifest.metrics['offline_' + m.toLowerCase()].psnr;
    });

    var traces = [
      {
        type: 'bar', y: models, x: naiveGaps, orientation: 'h',
        name: 'Naive', marker: { color: '#EF553B' },
        text: naiveGaps.map(function (v) { return v.toFixed(1) + ' dB'; }),
        textposition: 'outside',
      },
      {
        type: 'bar', y: models, x: boostedGaps, orientation: 'h',
        name: 'CL Boosted', marker: { color: '#00CC96' },
        text: boostedGaps.map(function (v) { return v.toFixed(1) + ' dB'; }),
        textposition: 'outside',
      },
    ];

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 300,
      barmode: 'group',
      xaxis: { title: 'PSNR Gap from Offline (dB)' },
      legend: { orientation: 'h', y: 1.08 },
      margin: { l: 70, r: 80, t: 40, b: 50 },
      shapes: [{ type: 'line', x0: 0, x1: 0, y0: -0.5, y1: 2.5,
                 line: { dash: 'dash', color: 'gray' } }],
    });

    Plotly.react('chart-gap', traces, layout, { responsive: true, displayModeBar: false });
  }

  // ══════════════════════════════════════════════════════════════════════
  // MODEL COMPARISON PAGE
  // ══════════════════════════════════════════════════════════════════════
  function renderComparisonPage() {
    renderModelRadar();
    renderSSIMBar();
    renderCLEffectiveness();
    renderSizeQualityScatter();
  }

  window.renderModelRadar = function () {
    var approach = document.getElementById('model-approach-select').value;
    var models = ['base', 'medium', 'large'];
    var modelLabels = { base: 'Base', medium: 'Medium', large: 'Large' };
    var radarMetrics = ['psnr', 'ssim', 'cr'];
    var radarDisplay = ['PSNR', 'SSIM', 'Compression Ratio'];

    // Normalize
    var normalized = {};
    radarMetrics.forEach(function (m) {
      var vals = models.map(function (mod) { return manifest.metrics[approach + '_' + mod][m]; });
      var mn = Math.min.apply(null, vals);
      var mx = Math.max.apply(null, vals);
      var rng = mx - mn || 1;
      normalized[m] = vals.map(function (v) { return (v - mn) / rng; });
    });

    var traces = models.map(function (mod, mi) {
      var r = radarMetrics.map(function (m) { return normalized[m][mi]; });
      r.push(r[0]);
      return {
        type: 'scatterpolar',
        r: r,
        theta: radarDisplay.concat([radarDisplay[0]]),
        fill: 'toself',
        name: modelLabels[mod],
        line: { color: MODEL_COLORS[modelLabels[mod]] },
        opacity: 0.7,
      };
    });

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      polar: { radialaxis: { visible: true, range: [0, 1], showticklabels: false } },
      height: 400,
      legend: { orientation: 'h', y: -0.1 },
      margin: { l: 40, r: 40, t: 20, b: 40 },
    });

    Plotly.react('chart-model-radar', traces, layout, { responsive: true, displayModeBar: false });
  };

  function renderSSIMBar() {
    var models = ['Base', 'Medium', 'Large'];
    var approaches = [
      { label: 'Batch (Offline)', prefix: 'offline' },
      { label: 'Online (Naive)', prefix: 'online' },
      { label: 'CL Boosted', prefix: 'boosted' },
    ];

    var traces = approaches.map(function (a) {
      return {
        type: 'bar',
        name: a.label,
        y: models,
        x: models.map(function (m) {
          return manifest.metrics[a.prefix + '_' + m.toLowerCase()].ssim;
        }),
        orientation: 'h',
        marker: { color: APPROACH_COLORS[a.label] },
      };
    });

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 400,
      barmode: 'group',
      xaxis: { title: 'SSIM' },
      legend: { orientation: 'h', y: -0.15 },
      margin: { l: 70, r: 20, t: 10, b: 50 },
    });

    Plotly.react('chart-ssim-bar', traces, layout, { responsive: true, displayModeBar: false });
  }

  function renderCLEffectiveness() {
    var models = ['base', 'medium', 'large'];
    var modelLabels = { base: 'Base', medium: 'Medium', large: 'Large' };

    // Metrics cards
    var container = document.getElementById('cl-metrics');
    var html = '';
    models.forEach(function (mod) {
      var naive = manifest.metrics['online_' + mod].psnr;
      var boosted = manifest.metrics['boosted_' + mod].psnr;
      var offline = manifest.metrics['offline_' + mod].psnr;
      var recovery = boosted - naive;
      var pct = ((recovery) / (offline - naive) * 100).toFixed(1);
      html += '<div class="metric-card">'
        + '<span class="metric-value">+' + recovery.toFixed(1) + ' dB</span>'
        + '<span class="metric-label">' + modelLabels[mod] + ' Recovery</span>'
        + '<span class="metric-delta positive">' + pct + '% of gap recovered</span>'
        + '</div>';
    });
    container.innerHTML = html;

    // Table
    var tbody = document.getElementById('cl-body');
    tbody.innerHTML = models.map(function (mod) {
      var naive = manifest.metrics['online_' + mod].psnr;
      var boosted = manifest.metrics['boosted_' + mod].psnr;
      var offline = manifest.metrics['offline_' + mod].psnr;
      var recovery = boosted - naive;
      var pct = ((recovery) / (offline - naive) * 100).toFixed(1);
      return '<tr>'
        + '<td class="var-name">' + modelLabels[mod] + '</td>'
        + '<td>' + naive.toFixed(2) + '</td>'
        + '<td>' + boosted.toFixed(2) + '</td>'
        + '<td>' + offline.toFixed(2) + '</td>'
        + '<td style="color:#27AE60;font-weight:700">+' + recovery.toFixed(2) + '</td>'
        + '<td>' + pct + '%</td>'
        + '</tr>';
    }).join('');
  }

  function renderSizeQualityScatter() {
    var approaches = [
      { label: 'Batch (Offline)', prefix: 'offline', color: '#636EFA' },
      { label: 'Online (Naive)', prefix: 'online', color: '#EF553B' },
      { label: 'CL Boosted', prefix: 'boosted', color: '#00CC96' },
    ];
    var models = ['base', 'medium', 'large'];
    var modelLabels = { base: 'Base', medium: 'Medium', large: 'Large' };
    var symbols = { base: 'circle', medium: 'diamond', large: 'square' };

    var traces = [];
    approaches.forEach(function (a) {
      models.forEach(function (mod) {
        var m = manifest.metrics[a.prefix + '_' + mod];
        traces.push({
          type: 'scatter',
          x: [m.size_kb],
          y: [m.psnr],
          mode: 'markers',
          name: a.label + ' (' + modelLabels[mod] + ')',
          marker: {
            size: Math.sqrt(m.params) / 5,
            color: a.color,
            symbol: symbols[mod],
            line: { width: 1, color: '#fff' },
          },
          text: [modelLabels[mod] + '<br>SSIM: ' + m.ssim.toFixed(4) + '<br>Params: ' + m.params.toLocaleString()],
          hoverinfo: 'text+x+y',
        });
      });
    });

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 480,
      xaxis: { title: 'Model Size (KB)' },
      yaxis: { title: 'PSNR (dB)' },
      legend: { orientation: 'h', y: -0.15, font: { size: 10 } },
      margin: { l: 60, r: 20, t: 20, b: 80 },
    });

    Plotly.react('chart-size-quality', traces, layout, { responsive: true, displayModeBar: false });
  }

  // ══════════════════════════════════════════════════════════════════════
  // DISTRIBUTION BEFORE vs AFTER PAGE
  // ══════════════════════════════════════════════════════════════════════
  function renderDistComparePage() {
    if (!compressedStats) return;
    renderDistComparison();
    renderDistCompareAll();
    renderDist4Panel();
  }

  window.renderDistComparison = function () {
    if (!compressedStats) return;
    var field = document.getElementById('distcomp-field').value;
    var model = document.getElementById('distcomp-model').value;

    var orig = compressedStats.original[field];
    var comp = compressedStats[model] ? compressedStats[model][field] : null;
    if (!orig) return;

    var centers = [];
    for (var i = 0; i < orig.counts.length; i++) {
      centers.push((orig.edges[i] + orig.edges[i + 1]) / 2);
    }

    // Normalize to relative frequency
    var origTotal = orig.counts.reduce(function (a, b) { return a + b; }, 0);
    var origFreq = orig.counts.map(function (c) { return c / origTotal; });

    var traces = [{
      type: 'scatter',
      x: centers,
      y: origFreq,
      mode: 'lines',
      name: 'Original',
      line: { color: COLORS[field], width: 2.5 },
      fill: 'tozeroy',
      fillcolor: COLORS[field] + '20',
    }];

    if (comp) {
      var compTotal = comp.counts.reduce(function (a, b) { return a + b; }, 0);
      var compFreq = comp.counts.map(function (c) { return c / compTotal; });
      traces.push({
        type: 'scatter',
        x: centers,
        y: compFreq,
        mode: 'lines',
        name: COMPRESSED_LABELS[model] || model,
        line: { color: COMPRESSED_COLORS[model] || '#FF6692', width: 2.5, dash: 'dot' },
        fill: 'tozeroy',
        fillcolor: (COMPRESSED_COLORS[model] || '#FF6692') + '15',
      });
    }

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 460,
      xaxis: { title: 'Value' },
      yaxis: { title: 'Relative Frequency' },
      legend: { orientation: 'h', y: -0.12, font: { size: 13 } },
      margin: { l: 60, r: 20, t: 10, b: 50 },
    });

    Plotly.react('chart-dist-overlay', traces, layout, { responsive: true, displayModeBar: false });

    // Metrics
    var container = document.getElementById('distcomp-metrics');
    if (comp) {
      var meanShift = Math.abs(comp.mean - orig.mean);
      var stdShift = Math.abs(comp.std - orig.std);
      // KL-like divergence approximation
      var distortion = 0;
      for (var i = 0; i < origFreq.length; i++) {
        var p = origFreq[i] + 1e-10;
        var q = (comp ? compFreq[i] : origFreq[i]) + 1e-10;
        distortion += p * Math.log(p / q);
      }
      var compMin = comp.min !== undefined ? comp.min.toFixed(4) : '--';
      var compMax = comp.max !== undefined ? comp.max.toFixed(4) : '--';
      container.innerHTML =
        '<div class="metric-card"><span class="metric-value">' + orig.mean.toFixed(4) +
        '</span><span class="metric-label">Original Mean</span></div>' +
        '<div class="metric-card"><span class="metric-value">' + comp.mean.toFixed(4) +
        '</span><span class="metric-label">Reconstructed Mean</span></div>' +
        '<div class="metric-card"><span class="metric-value">' + meanShift.toFixed(4) +
        '</span><span class="metric-label">Mean Shift</span></div>' +
        '<div class="metric-card"><span class="metric-value">' + orig.std.toFixed(4) +
        '</span><span class="metric-label">Original Std</span></div>' +
        '<div class="metric-card"><span class="metric-value">' + comp.std.toFixed(4) +
        '</span><span class="metric-label">Reconstructed Std</span></div>' +
        '<div class="metric-card"><span class="metric-value">' + compMin + ' / ' + compMax +
        '</span><span class="metric-label">Reconstructed Min / Max</span></div>';
    } else {
      container.innerHTML = '<div class="info-box">No compressed data available for this model.</div>';
    }
  };

  window.renderDistCompareAll = function () {
    if (!compressedStats) return;
    var field = document.getElementById('distcomp-field-all').value;
    var orig = compressedStats.original[field];
    if (!orig) return;

    var skipKeys = ['original', 'normalization', 'histogram_ranges'];
    var modelKeys = Object.keys(compressedStats).filter(function (k) { return skipKeys.indexOf(k) === -1; });

    var barData = [{ name: 'Original', mean: orig.mean, std: orig.std }];
    modelKeys.forEach(function (k) {
      if (compressedStats[k][field]) {
        barData.push({
          name: COMPRESSED_LABELS[k] || k,
          mean: compressedStats[k][field].mean,
          std: compressedStats[k][field].std,
        });
      }
    });

    var traces = [
      {
        type: 'bar',
        name: 'Mean',
        x: barData.map(function (d) { return d.name; }),
        y: barData.map(function (d) { return d.mean; }),
        marker: { color: barData.map(function (d, i) { return i === 0 ? COLORS[field] : '#85C1E9'; }) },
        error_y: {
          type: 'data',
          array: barData.map(function (d) { return d.std; }),
          visible: true,
          color: '#666',
        },
      },
    ];

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 420,
      yaxis: { title: 'Value (mean +/- std)' },
      showlegend: false,
      margin: { l: 60, r: 20, t: 10, b: 100 },
      xaxis: { tickangle: -25 },
    });

    Plotly.react('chart-dist-all', traces, layout, { responsive: true, displayModeBar: false });
  };

  function renderDist4Panel() {
    if (!compressedStats || !compressedStats['offline_large']) return;

    var traces = [];
    var fieldList = ['Vx', 'Vy', 'Pressure', 'TKE'];

    fieldList.forEach(function (field, fi) {
      var orig = compressedStats.original[field];
      var comp = compressedStats['offline_large'][field];
      if (!orig || !comp) return;

      var centers = [];
      for (var i = 0; i < orig.counts.length; i++) {
        centers.push((orig.edges[i] + orig.edges[i + 1]) / 2);
      }
      var origTotal = orig.counts.reduce(function (a, b) { return a + b; }, 0);
      var compTotal = comp.counts.reduce(function (a, b) { return a + b; }, 0);

      traces.push({
        type: 'scatter',
        x: centers,
        y: orig.counts.map(function (c) { return c / origTotal; }),
        mode: 'lines',
        name: fi === 0 ? 'Original' : undefined,
        showlegend: fi === 0,
        legendgroup: 'original',
        line: { color: COLORS[field], width: 2 },
        xaxis: 'x' + (fi + 1),
        yaxis: 'y' + (fi + 1),
      });
      traces.push({
        type: 'scatter',
        x: centers,
        y: comp.counts.map(function (c) { return c / compTotal; }),
        mode: 'lines',
        name: fi === 0 ? 'Batch Large' : undefined,
        showlegend: fi === 0,
        legendgroup: 'compressed',
        line: { color: COMPRESSED_COLORS['offline_large'], width: 2, dash: 'dot' },
        xaxis: 'x' + (fi + 1),
        yaxis: 'y' + (fi + 1),
      });
    });

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
      height: 500,
      grid: { rows: 2, columns: 2, pattern: 'independent', xgap: 0.08, ygap: 0.12 },
      legend: { orientation: 'h', y: -0.08, font: { size: 13 } },
      margin: { l: 50, r: 20, t: 30, b: 50 },
      annotations: fieldList.map(function (f, i) {
        var row = Math.floor(i / 2);
        var col = i % 2;
        return {
          text: '<b>' + f + '</b>',
          xref: 'x' + (i + 1) + ' domain',
          yref: 'y' + (i + 1) + ' domain',
          x: 0.5, y: 1.12,
          showarrow: false,
          font: { size: 14, color: COLORS[f] },
        };
      }),
    });

    Plotly.react('chart-dist-4panel', traces, layout, { responsive: true, displayModeBar: false });
  }

  // ══════════════════════════════════════════════════════════════════════
  // PAGE: Training & Optimization (merged)
  // ══════════════════════════════════════════════════════════════════════

  var OPT_NAMES = { gd: 'Gradient Descent', sgd: 'SGD', adam: 'Adam' };
  var OPT_COLORS = { gd: '#636EFA', sgd: '#EF553B', adam: '#00CC96' };
  var ACT_COLORS = { relu: '#E74C3C', leaky: '#2874A6', sin: '#27AE60' };
  var WINDOW_COLORS = ['#E74C3C', '#2874A6', '#27AE60', '#8E44AD', '#E67E22'];
  var LIGHT_CONTOUR = [[0, '#f7fbff'], [0.15, '#deebf7'], [0.3, '#c6dbef'], [0.5, '#9ecae1'], [0.7, '#6baed6'], [0.85, '#3182bd'], [1, '#08519c']];

  // Animation state
  var optAnim = { timer: null, step: 0, totalSteps: 0, paths: {}, losses: {}, surface: null, show: [], scenario: '' };

  function renderTrainingPage() {
    renderTrainingOpt();
    renderActivations();
    renderDeadNeuronChart();
  }

  function linspace(a, b, n) {
    var arr = [], step = (b - a) / (n - 1);
    for (var i = 0; i < n; i++) arr.push(a + step * i);
    return arr;
  }

  // ── Loss surface definitions per scenario ──────────────────────────

  function offlineLoss(w1, w2) { return 0.5 * (w1 * w1 + 5 * w2 * w2); }
  function offlineGrad(w1, w2) { return [w1, 5 * w2]; }

  var WINDOW_MINS = [[2.0, 1.5], [0.5, -1.0], [-1.5, 0.8], [-0.5, -0.5]];

  function windowLoss(w1, w2, cx, cy) {
    var d1 = w1 - cx, d2 = w2 - cy;
    return 0.5 * (d1 * d1 + 3 * d2 * d2);
  }
  function windowGrad(w1, w2, cx, cy) { return [w1 - cx, 3 * (w2 - cy)]; }

  function erLoss(w1, w2) {
    var total = 0;
    for (var i = 0; i < WINDOW_MINS.length; i++) total += windowLoss(w1, w2, WINDOW_MINS[i][0], WINDOW_MINS[i][1]);
    return total / WINDOW_MINS.length;
  }
  function erGrad(w1, w2) {
    var g = [0, 0];
    for (var i = 0; i < WINDOW_MINS.length; i++) {
      var wg = windowGrad(w1, w2, WINDOW_MINS[i][0], WINDOW_MINS[i][1]);
      g[0] += wg[0]; g[1] += wg[1];
    }
    return [g[0] / WINDOW_MINS.length, g[1] / WINDOW_MINS.length];
  }

  // ── Optimizer simulators ───────────────────────────────────────────

  function simGD(lr, steps, lossFn, gradFn, w0) {
    var w = w0.slice(), path = [w.slice()], losses = [lossFn(w[0], w[1])];
    for (var i = 0; i < steps; i++) {
      var g = gradFn(w[0], w[1]);
      w = [w[0] - lr * g[0], w[1] - lr * g[1]];
      path.push(w.slice()); losses.push(lossFn(w[0], w[1]));
    }
    return { path: path, losses: losses };
  }

  function simSGD(lr, steps, lossFn, gradFn, w0) {
    var w = w0.slice(), path = [w.slice()], losses = [lossFn(w[0], w[1])];
    for (var i = 0; i < steps; i++) {
      var eta = lr / (1 + 0.02 * i);
      var g = gradFn(w[0], w[1]);
      var noise = 0.4 * Math.max(0.2, 1 - i / steps);
      var n1 = (Math.sin(i * 7.3 + 1.7) * 0.5 + Math.sin(i * 13.1) * 0.3) * noise;
      var n2 = (Math.sin(i * 11.1 + 2.3) * 0.5 + Math.sin(i * 17.7) * 0.3) * noise;
      w = [w[0] - eta * (g[0] + n1), w[1] - eta * (g[1] + n2)];
      path.push(w.slice()); losses.push(lossFn(w[0], w[1]));
    }
    return { path: path, losses: losses };
  }

  function simAdam(lr, steps, lossFn, gradFn, w0) {
    var w = w0.slice(), path = [w.slice()], losses = [lossFn(w[0], w[1])];
    var b1 = 0.9, b2 = 0.999, eps = 1e-7, m = [0, 0], s = [0, 0];
    for (var i = 0; i < steps; i++) {
      var t = i + 1, g = gradFn(w[0], w[1]);
      m = [b1 * m[0] + (1 - b1) * g[0], b1 * m[1] + (1 - b1) * g[1]];
      s = [b2 * s[0] + (1 - b2) * g[0] * g[0], b2 * s[1] + (1 - b2) * g[1] * g[1]];
      var mh = [m[0] / (1 - Math.pow(b1, t)), m[1] / (1 - Math.pow(b1, t))];
      var sh = [s[0] / (1 - Math.pow(b2, t)), s[1] / (1 - Math.pow(b2, t))];
      w = [w[0] - lr * mh[0] / (Math.sqrt(sh[0]) + eps), w[1] - lr * mh[1] / (Math.sqrt(sh[1]) + eps)];
      path.push(w.slice()); losses.push(lossFn(w[0], w[1]));
    }
    return { path: path, losses: losses };
  }

  function simOnlineNaive(optKey, lr, stepsPerWindow) {
    var w = [3.0, 2.5], fullPath = [w.slice()], windowPaths = [];
    for (var wi = 0; wi < WINDOW_MINS.length; wi++) {
      var cx = WINDOW_MINS[wi][0], cy = WINDOW_MINS[wi][1];
      var lf = function (w1, w2) { return windowLoss(w1, w2, cx, cy); };
      var gf = function (w1, w2) { return windowGrad(w1, w2, cx, cy); };
      var sim = optKey === 'adam' ? simAdam : optKey === 'sgd' ? simSGD : simGD;
      var res = sim(lr, stepsPerWindow, lf, gf, w);
      w = res.path[res.path.length - 1];
      windowPaths.push(res.path);
      fullPath = fullPath.concat(res.path.slice(1));
    }
    var globalLosses = fullPath.map(function (p) {
      var total = 0;
      for (var i = 0; i < WINDOW_MINS.length; i++) total += windowLoss(p[0], p[1], WINDOW_MINS[i][0], WINDOW_MINS[i][1]);
      return total / WINDOW_MINS.length;
    });
    return { path: fullPath, losses: globalLosses, windowPaths: windowPaths };
  }

  // ── Render contour surface (static background) ─────────────────────

  function buildContourSurface(scenario) {
    var range = 4.5, contN = 80;
    var contX = linspace(-range, range, contN);
    var contY = linspace(-range, range, contN);
    var lossFn;
    if (scenario === 'offline') lossFn = offlineLoss;
    else if (scenario === 'online_er') lossFn = erLoss;
    else {
      var lw = WINDOW_MINS[WINDOW_MINS.length - 1];
      lossFn = function (w1, w2) { return windowLoss(w1, w2, lw[0], lw[1]); };
    }
    var contZ = [];
    for (var i = 0; i < contN; i++) {
      var row = [];
      for (var j = 0; j < contN; j++) row.push(lossFn(contX[j], contY[i]));
      contZ.push(row);
    }
    return { contX: contX, contY: contY, contZ: contZ, lossFn: lossFn, range: range };
  }

  // ── Draw frame at step N ───────────────────────────────────────────

  function drawOptFrame(step) {
    var a = optAnim;
    var surf = a.surface;
    var range = surf.range;
    var scenario = a.scenario;

    // Update slider + label
    var slider = document.getElementById('opt-step-slider');
    var label = document.getElementById('opt-step-label');
    if (slider) slider.value = step;
    if (label) label.textContent = 'Step ' + step + ' / ' + a.totalSteps;

    // ── 2D contour ───────────────────────────────────────────────
    var traces2d = [{
      type: 'contour', x: surf.contX, y: surf.contY, z: surf.contZ,
      colorscale: LIGHT_CONTOUR, ncontours: 25, showscale: false,
      line: { width: 0.5, color: 'rgba(100,100,100,0.3)' },
    }];

    // Stars for minima
    if (scenario === 'online') {
      WINDOW_MINS.forEach(function (wm, wi) {
        traces2d.push({
          type: 'scatter', mode: 'markers',
          x: [wm[0]], y: [wm[1]],
          marker: { size: 14, color: WINDOW_COLORS[wi], symbol: 'star', line: { width: 1, color: '#333' } },
          name: 'W' + (wi + 1) + ' min', showlegend: true,
        });
      });
    } else {
      var minPt = scenario === 'offline' ? [0, 0] :
        [WINDOW_MINS.reduce(function (s, w) { return s + w[0]; }, 0) / WINDOW_MINS.length,
         WINDOW_MINS.reduce(function (s, w) { return s + w[1]; }, 0) / WINDOW_MINS.length];
      traces2d.push({
        type: 'scatter', mode: 'markers', x: [minPt[0]], y: [minPt[1]],
        marker: { size: 14, color: '#FFD700', symbol: 'star', line: { width: 1, color: '#333' } },
        name: 'Global Min', showlegend: true,
      });
    }

    // Optimizer paths up to current step
    a.show.forEach(function (ok) {
      var fullPath = a.paths[ok];
      var n = Math.min(step + 1, fullPath.length);
      var px = [], py = [];
      for (var i = 0; i < n; i++) { px.push(fullPath[i][0]); py.push(fullPath[i][1]); }

      // Trail (fading line)
      traces2d.push({
        type: 'scatter', mode: 'lines',
        x: px, y: py,
        line: { color: OPT_COLORS[ok], width: 2 },
        name: OPT_NAMES[ok], showlegend: true, legendgroup: ok,
      });
      // Dots along trail
      traces2d.push({
        type: 'scatter', mode: 'markers',
        x: px, y: py,
        marker: { size: 4, color: OPT_COLORS[ok], opacity: 0.5 },
        showlegend: false, legendgroup: ok,
      });
      // Current position (big dot)
      if (n > 0) {
        traces2d.push({
          type: 'scatter', mode: 'markers',
          x: [px[n - 1]], y: [py[n - 1]],
          marker: { size: 10, color: OPT_COLORS[ok], line: { width: 2, color: '#fff' } },
          showlegend: false, legendgroup: ok,
        });
      }
      // Start marker
      traces2d.push({
        type: 'scatter', mode: 'markers',
        x: [fullPath[0][0]], y: [fullPath[0][1]],
        marker: { size: 8, color: '#333', symbol: 'diamond', line: { width: 1, color: '#fff' } },
        name: ok === a.show[0] ? 'Start' : undefined,
        showlegend: ok === a.show[0], legendgroup: 'start',
      });
    });

    Plotly.react('chart-opt-contour', traces2d, Object.assign({}, PLOTLY_LAYOUT, {
      height: 480,
      xaxis: { title: 'w₁', range: [-range, range], autorange: false, gridcolor: 'rgba(0,0,0,0.06)' },
      yaxis: { title: 'w₂', range: [-range, range], autorange: false, gridcolor: 'rgba(0,0,0,0.06)' },
      legend: { orientation: 'h', y: -0.12, font: { size: 11 } },
    }), { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'hoverCompareCartesian', 'hoverClosestCartesian'], displaylogo: false });

    // ── 3D surface ───────────────────────────────────────────────
    var surfN = 50, surfX = [], surfY = [], surfZ = [];
    for (var i = 0; i < surfN; i++) {
      var rx = [], ry = [], rz = [];
      for (var j = 0; j < surfN; j++) {
        var x = -range + 2 * range * i / (surfN - 1);
        var y = -range + 2 * range * j / (surfN - 1);
        rx.push(x); ry.push(y); rz.push(surf.lossFn(x, y));
      }
      surfX.push(rx); surfY.push(ry); surfZ.push(rz);
    }

    var t3d = [{
      type: 'surface', x: surfX, y: surfY, z: surfZ,
      colorscale: LIGHT_CONTOUR, opacity: 0.75, showscale: false,
    }];

    a.show.forEach(function (ok) {
      var fullPath = a.paths[ok];
      var n = Math.min(step + 1, fullPath.length);
      t3d.push({
        type: 'scatter3d', mode: 'lines+markers',
        x: fullPath.slice(0, n).map(function (p) { return p[0]; }),
        y: fullPath.slice(0, n).map(function (p) { return p[1]; }),
        z: fullPath.slice(0, n).map(function (p) { return surf.lossFn(p[0], p[1]); }),
        name: OPT_NAMES[ok], line: { color: OPT_COLORS[ok], width: 5 },
        marker: { size: 3, color: OPT_COLORS[ok] },
      });
    });

    Plotly.react('chart-opt-3d', t3d, Object.assign({}, PLOTLY_LAYOUT, {
      height: 480,
      scene: { xaxis: { title: 'w₁' }, yaxis: { title: 'w₂' }, zaxis: { title: 'Loss' },
               camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } } },
      margin: { l: 0, r: 0, t: 10, b: 0 },
      legend: { orientation: 'h', y: -0.05 },
    }), { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['orbitRotation', 'hoverClosest3d'], displaylogo: false });

    // ── Loss curve ───────────────────────────────────────────────
    var tracesLoss = [];
    a.show.forEach(function (ok) {
      var allLosses = a.losses[ok];
      var n = Math.min(step + 1, allLosses.length);
      tracesLoss.push({
        type: 'scatter', mode: 'lines',
        x: allLosses.slice(0, n).map(function (_, i) { return i; }),
        y: allLosses.slice(0, n),
        name: OPT_NAMES[ok], line: { color: OPT_COLORS[ok], width: 2.5 },
      });
      // Current step marker on loss curve
      if (n > 0) {
        tracesLoss.push({
          type: 'scatter', mode: 'markers',
          x: [n - 1], y: [allLosses[n - 1]],
          marker: { size: 8, color: OPT_COLORS[ok], line: { width: 2, color: '#fff' } },
          showlegend: false,
        });
      }
    });
    // Window boundary lines for online
    if (scenario === 'online') {
      for (var wi = 1; wi < WINDOW_MINS.length; wi++) {
        tracesLoss.push({
          type: 'scatter', mode: 'lines',
          x: [wi * 41, wi * 41], y: [0.001, 30],
          line: { color: WINDOW_COLORS[wi], width: 1, dash: 'dot' },
          name: wi === 1 ? 'Window boundary' : undefined,
          showlegend: wi === 1, legendgroup: 'wbound',
        });
      }
    }

    Plotly.react('chart-opt-loss', tracesLoss, Object.assign({}, PLOTLY_LAYOUT, {
      height: 400,
      xaxis: { title: scenario === 'online' ? 'Step (across windows)' : 'Iteration' },
      yaxis: { title: 'MSE Loss', type: 'log' },
      legend: { orientation: 'h', y: -0.15 },
    }), { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'], displaylogo: false });
  }

  // ── Animation controls ─────────────────────────────────────────────

  window.toggleOptPlay = function () {
    var btn = document.getElementById('opt-play-btn');
    if (optAnim.timer) {
      clearInterval(optAnim.timer);
      optAnim.timer = null;
      btn.innerHTML = '&#9654; Play';
    } else {
      if (optAnim.step >= optAnim.totalSteps) optAnim.step = 0;
      var speed = parseInt(document.getElementById('opt-speed').value);
      btn.innerHTML = '&#9646;&#9646; Pause';
      optAnim.timer = setInterval(function () {
        if (optAnim.step >= optAnim.totalSteps) {
          clearInterval(optAnim.timer);
          optAnim.timer = null;
          btn.innerHTML = '&#9654; Play';
          return;
        }
        optAnim.step++;
        drawOptFrame(optAnim.step);
      }, speed);
    }
  };

  window.resetOptAnim = function () {
    if (optAnim.timer) { clearInterval(optAnim.timer); optAnim.timer = null; }
    document.getElementById('opt-play-btn').innerHTML = '&#9654; Play';
    optAnim.step = 0;
    drawOptFrame(0);
  };

  window.seekOptStep = function (step) {
    optAnim.step = step;
    drawOptFrame(step);
  };

  // ── Main setup (precompute all paths, draw frame 0) ────────────────

  window.renderTrainingOpt = function () {
    // Stop any running animation
    if (optAnim.timer) { clearInterval(optAnim.timer); optAnim.timer = null; }
    document.getElementById('opt-play-btn').innerHTML = '&#9654; Play';

    var scenario = document.getElementById('opt-scenario').value;
    var optKey = document.getElementById('opt-select').value;
    var lr = parseFloat(document.getElementById('opt-lr-select').value);
    var steps = 150;
    var stepsPerWindow = 40;
    var w0 = [3.0, 2.5];

    // Scenario info
    var infoDiv = document.getElementById('opt-scenario-info');
    var infos = {
      offline: '<strong>Offline (Batch):</strong> The model sees the entire dataset every epoch. MSE loss forms a single convex bowl with one global minimum. Adam converges smoothly.',
      online: '<strong>Online (Naive):</strong> The model trains on one temporal window at a time. Each window defines a different loss surface with its own minimum (coloured stars). The optimizer chases each new minimum, overwriting previous knowledge &mdash; this is <strong>catastrophic forgetting</strong>.',
      online_er: '<strong>Online + Experience Replay:</strong> Replay mixes past samples into each window&rsquo;s training batch. The effective loss surface becomes a weighted average of all windows, and Adam converges toward a <strong>compromise minimum</strong> (gold star) that serves all windows.',
    };
    infoDiv.innerHTML = infos[scenario];

    var show = optKey === 'all' ? ['gd', 'sgd', 'adam'] : [optKey];

    // Precompute all paths and losses
    var surface = buildContourSurface(scenario);
    var paths = {}, losses = {};
    var maxLen = 0;

    if (scenario === 'online') {
      show.forEach(function (ok) {
        var res = simOnlineNaive(ok, lr, stepsPerWindow);
        paths[ok] = res.path;
        losses[ok] = res.losses;
        maxLen = Math.max(maxLen, res.path.length);
      });
    } else {
      var lf = scenario === 'offline' ? offlineLoss : erLoss;
      var gf = scenario === 'offline' ? offlineGrad : erGrad;
      show.forEach(function (ok) {
        var sim = ok === 'adam' ? simAdam : ok === 'sgd' ? simSGD : simGD;
        var res = sim(lr, steps, lf, gf, w0);
        paths[ok] = res.path;
        losses[ok] = res.losses;
        maxLen = Math.max(maxLen, res.path.length);
      });
    }

    // Store animation state
    optAnim.paths = paths;
    optAnim.losses = losses;
    optAnim.surface = surface;
    optAnim.show = show;
    optAnim.scenario = scenario;
    optAnim.totalSteps = maxLen - 1;
    optAnim.step = 0;

    // Update slider max
    var slider = document.getElementById('opt-step-slider');
    if (slider) slider.max = optAnim.totalSteps;

    // Draw initial frame (step 0 = starting position only)
    drawOptFrame(0);

    // Update rules panel
    var propsDiv = document.getElementById('opt-properties');
    if (!propsDiv) return;
    var rules = [
      { name: 'GD', color: OPT_COLORS.gd, eq: 'θ ← θ − η ∇<sub>θ</sub> L(θ)', desc: 'Full gradient, convex MSE → guaranteed global min (Geron Eq 4-7)' },
      { name: 'SGD', color: OPT_COLORS.sgd, eq: 'θ ← θ − η(t) ∇<sub>θ</sub> L(θ; x<sup>(i)</sup>)', desc: 'Mini-batch gradient + learning schedule η(t)=t₀/(t+t₁) (Geron Fig 4-9)' },
      { name: 'Adam', color: OPT_COLORS.adam, eq: 'θ ← θ − η m̂ / (√ŝ + ε)', desc: 'Momentum (β₁=0.9) + RMSProp (β₂=0.999) with bias correction. Thesis default. (Kingma & Ba 2014, Geron Eq 11-8)' },
    ];
    var html = '';
    rules.forEach(function (r) {
      html += '<div style="background:#f8f9fc; border:1px solid var(--border); border-left:4px solid ' + r.color + '; border-radius:6px; padding:12px 16px; margin-bottom:10px;">';
      html += '<strong style="color:' + r.color + '">' + r.name + '</strong>';
      html += '<div style="font-family:var(--font-mono); font-size:0.9rem; margin:6px 0;">' + r.eq + '</div>';
      html += '<div style="font-size:0.82rem; color:var(--text-dim);">' + r.desc + '</div>';
      html += '</div>';
    });
    propsDiv.innerHTML = html;
  };

  // ── Activation function renderers ──────────────────────────────────

  function relu(x) { return Math.max(0, x); }
  function reluGrad(x) { return x > 0 ? 1 : 0; }
  function leakyRelu(x, alpha) { return x >= 0 ? x : alpha * x; }
  function leakyReluGrad(x, alpha) { return x >= 0 ? 1 : alpha; }
  function sinAct(x, omega) { return Math.sin(omega * x); }
  function sinActGrad(x, omega) { return omega * Math.cos(omega * x); }

  window.renderActivations = function () {
    var alpha = parseFloat(document.getElementById('act-alpha').value);
    var omega = parseFloat(document.getElementById('act-omega').value);
    var xs = linspace(-3, 3, 600);

    Plotly.react('chart-act-forward', [
      { x: xs, y: xs.map(relu), name: 'ReLU', line: { color: ACT_COLORS.relu, width: 2.5 } },
      { x: xs, y: xs.map(function (x) { return leakyRelu(x, alpha); }), name: 'Leaky ReLU (α=' + alpha + ')', line: { color: ACT_COLORS.leaky, width: 2.5 } },
      { x: xs, y: xs.map(function (x) { return sinAct(x, omega); }), name: 'Sin (ω=' + omega + ')', line: { color: ACT_COLORS.sin, width: 2.5 } },
    ], Object.assign({}, PLOTLY_LAYOUT, {
      height: 450,
      xaxis: { title: 'x', zeroline: true, zerolinecolor: '#ccc' },
      yaxis: { title: 'f(x)', zeroline: true, zerolinecolor: '#ccc' },
      legend: { orientation: 'h', y: -0.15 },
    }), { responsive: true, displayModeBar: false });

    Plotly.react('chart-act-gradient', [
      { x: xs, y: xs.map(reluGrad), name: "ReLU'", line: { color: ACT_COLORS.relu, width: 2.5 } },
      { x: xs, y: xs.map(function (x) { return leakyReluGrad(x, alpha); }), name: "Leaky ReLU'", line: { color: ACT_COLORS.leaky, width: 2.5 } },
      { x: xs, y: xs.map(function (x) { return sinActGrad(x, omega); }), name: "Sin'", line: { color: ACT_COLORS.sin, width: 2.5 } },
    ], Object.assign({}, PLOTLY_LAYOUT, {
      height: 450,
      xaxis: { title: 'x', zeroline: true, zerolinecolor: '#ccc' },
      yaxis: { title: "f'(x)", zeroline: true, zerolinecolor: '#ccc' },
      legend: { orientation: 'h', y: -0.15 },
    }), { responsive: true, displayModeBar: false });
  };

  function renderDeadNeuronChart() {
    var alpha = parseFloat(document.getElementById('act-alpha').value);
    var xs = linspace(-3, 3, 600);
    Plotly.react('chart-act-dead', [
      { x: xs, y: xs.map(function (x) { return Math.abs(reluGrad(x)); }), name: 'ReLU |gradient|', fill: 'tozeroy', line: { color: ACT_COLORS.relu, width: 2 }, fillcolor: 'rgba(231,76,60,0.15)' },
      { x: xs, y: xs.map(function (x) { return Math.abs(leakyReluGrad(x, alpha)); }), name: 'Leaky ReLU |gradient|', fill: 'tozeroy', line: { color: ACT_COLORS.leaky, width: 2 }, fillcolor: 'rgba(40,116,166,0.15)' },
    ], Object.assign({}, PLOTLY_LAYOUT, {
      height: 380,
      xaxis: { title: 'Pre-activation value (x)' },
      yaxis: { title: '|Gradient|', range: [-0.05, 1.15] },
      legend: { orientation: 'h', y: -0.15 },
      annotations: [
        { x: -1.5, y: 0.05, text: 'Dead zone (gradient=0)', showarrow: true, arrowhead: 2, ax: 0, ay: -40, font: { size: 12, color: ACT_COLORS.relu } },
        { x: -1.5, y: alpha + 0.05, text: 'Leaky: gradient=α', showarrow: true, arrowhead: 2, ax: 0, ay: -40, font: { size: 12, color: ACT_COLORS.leaky } },
      ],
    }), { responsive: true, displayModeBar: false });
  }

  // ── Bootstrap ─────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', init);
})();
