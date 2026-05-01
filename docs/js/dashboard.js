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
      // Trigger render on first visit
      if (!target.dataset.rendered) {
        target.dataset.rendered = '1';
        renderPage(pageId);
      }
    }
  };

  function renderPage(pageId) {
    switch (pageId) {
      case 'distributions': renderDistributionsPage(); break;
      case 'correlations': renderCorrelationsPage(); break;
      case 'temporal': renderTemporalPage(); break;
      case 'compression': renderCompressionPage(); break;
      case 'comparison': renderComparisonPage(); break;
      case 'distcompare': renderDistComparePage(); break;
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
  // DISTRIBUTIONS PAGE
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
      xaxis: { title: 'Normalized Value [0, 1]' },
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
      container.innerHTML =
        '<div class="metric-card"><span class="metric-value">' + orig.mean.toFixed(4) +
        '</span><span class="metric-label">Original Mean</span></div>' +
        '<div class="metric-card"><span class="metric-value">' + comp.mean.toFixed(4) +
        '</span><span class="metric-label">Compressed Mean</span></div>' +
        '<div class="metric-card"><span class="metric-value">' + meanShift.toFixed(4) +
        '</span><span class="metric-label">Mean Shift</span></div>' +
        '<div class="metric-card"><span class="metric-value">' + orig.std.toFixed(4) +
        '</span><span class="metric-label">Original Std</span></div>' +
        '<div class="metric-card"><span class="metric-value">' + comp.std.toFixed(4) +
        '</span><span class="metric-label">Compressed Std</span></div>' +
        '<div class="metric-card"><span class="metric-value">' + distortion.toFixed(4) +
        '</span><span class="metric-label">KL Divergence</span></div>';
    } else {
      container.innerHTML = '<div class="info-box">No compressed data available for this model.</div>';
    }
  };

  window.renderDistCompareAll = function () {
    if (!compressedStats) return;
    var field = document.getElementById('distcomp-field-all').value;
    var orig = compressedStats.original[field];
    if (!orig) return;

    var modelKeys = Object.keys(compressedStats).filter(function (k) { return k !== 'original'; });

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

  // ── Bootstrap ─────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', init);
})();
