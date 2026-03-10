/**
 * three-scene.js — Optional 3D view using Three.js.
 *
 * Renders a single 3D surface with vertex colors (jet colormap).
 * Only active when user selects "3D" view mode.
 */
var APP = window.APP || {};

APP.ThreeScene = (function () {
  'use strict';

  var renderer, scene, camera, controls;
  var surfaceMesh;
  var container, canvas;
  var nx, ny, gridX, gridY, cylinderMask;
  var animationId = null;
  var active = false;
  var heightScale = 0.12;

  function jetColor(t) {
    var r, g, b;
    if (t < 0.125) { r = 0; g = 0; b = 0.5 + t * 4; }
    else if (t < 0.375) { r = 0; g = (t - 0.125) * 4; b = 1; }
    else if (t < 0.625) { r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4; }
    else if (t < 0.875) { r = 1; g = 1 - (t - 0.625) * 4; b = 0; }
    else { r = 1 - (t - 0.875) * 4; g = 0; b = 0; }
    return [Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b))];
  }

  function init(_nx, _ny, _gridX, _gridY, _cylinderMask) {
    nx = _nx; ny = _ny; gridX = _gridX; gridY = _gridY; cylinderMask = _cylinderMask;

    container = document.getElementById('three-container');
    canvas = document.getElementById('three-canvas');
    if (!container || !canvas) return;

    var w = container.clientWidth;
    var h = container.clientHeight || 500;

    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.setClearColor(0x0a0a0f);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
    camera.position.set(0.09, 0.20, 0.20);
    camera.lookAt(0.09, 0.03, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    var d = new THREE.DirectionalLight(0xffffff, 0.7);
    d.position.set(1, 2, 1);
    scene.add(d);

    var OC = THREE.OrbitControls || window.OrbitControls;
    if (OC) {
      controls = new OC(camera, renderer.domElement);
      controls.target.set(0.09, 0.03, 0);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.update();
    }

    // Build surface mesh
    buildSurface();

    window.addEventListener('resize', onResize);
  }

  function buildSurface() {
    var vertexCount = nx * ny;
    var positions = new Float32Array(vertexCount * 3);
    var colors = new Float32Array(vertexCount * 3);

    for (var ix = 0; ix < nx; ix++) {
      for (var iy = 0; iy < ny; iy++) {
        var idx = ix * ny + iy;
        positions[idx * 3]     = gridX[ix];
        positions[idx * 3 + 1] = 0;
        positions[idx * 3 + 2] = gridY[iy];
        colors[idx * 3] = 0.1; colors[idx * 3 + 1] = 0.1; colors[idx * 3 + 2] = 0.15;
      }
    }

    var indices = [];
    for (var ix = 0; ix < nx - 1; ix++) {
      for (var iy = 0; iy < ny - 1; iy++) {
        var a = ix * ny + iy, b = (ix+1) * ny + iy;
        var c = (ix+1) * ny + (iy+1), d2 = ix * ny + (iy+1);
        if (cylinderMask[a] && cylinderMask[b] && cylinderMask[c] && cylinderMask[d2]) continue;
        indices.push(a, b, c, a, c, d2);
      }
    }

    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    var mat = new THREE.MeshPhongMaterial({
      vertexColors: true, side: THREE.DoubleSide, shininess: 30
    });
    surfaceMesh = new THREE.Mesh(geo, mat);
    scene.add(surfaceMesh);
  }

  function updateSurface(fieldData) {
    if (!surfaceMesh) return;
    var positions = surfaceMesh.geometry.attributes.position.array;
    var colors = surfaceMesh.geometry.attributes.color.array;

    for (var i = 0; i < nx * ny; i++) {
      var val = fieldData[i];
      positions[i * 3 + 1] = cylinderMask[i] ? 0 : val * heightScale;
      var c = jetColor(val);
      colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
    }
    surfaceMesh.geometry.attributes.position.needsUpdate = true;
    surfaceMesh.geometry.attributes.color.needsUpdate = true;
    surfaceMesh.geometry.computeVertexNormals();
  }

  function startLoop() {
    active = true;
    function animate() {
      if (!active) return;
      animationId = requestAnimationFrame(animate);
      if (controls) controls.update();
      renderer.render(scene, camera);
    }
    animate();
  }

  function stopLoop() {
    active = false;
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }

  function onResize() {
    if (!active || !container) return;
    var w = container.clientWidth;
    var h = container.clientHeight || 500;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  function show() {
    container.classList.remove('hidden');
    onResize();
    startLoop();
  }

  function hide() {
    container.classList.add('hidden');
    stopLoop();
  }

  return {
    init: init,
    updateSurface: updateSurface,
    show: show,
    hide: hide,
    startLoop: startLoop,
    stopLoop: stopLoop,
  };
})();

window.APP = APP;
