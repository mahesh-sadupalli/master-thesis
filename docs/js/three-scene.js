/**
 * three-scene.js — 3D view with three surfaces: Original, Predicted, Absolute Error.
 *
 * Each surface lives in its own renderer/scene/camera inside a .viz-panel.
 * OrbitControls on the first surface are linked to the other two so all
 * three rotate together.
 */
var APP = window.APP || {};

APP.ThreeScene = (function () {
  'use strict';

  var surfaces = [];  // [{renderer, scene, camera, controls, mesh, canvas, container}]
  var nx, ny, gridX, gridY, cylinderMask;
  var animationId = null;
  var active = false;
  var heightScale = 0.12;

  // ── Colormaps ──────────────────────────────────────────────────────

  function jetColor(t) {
    var r, g, b;
    if (t < 0.125)      { r = 0; g = 0; b = 0.5 + t * 4; }
    else if (t < 0.375) { r = 0; g = (t - 0.125) * 4; b = 1; }
    else if (t < 0.625) { r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4; }
    else if (t < 0.875) { r = 1; g = 1 - (t - 0.625) * 4; b = 0; }
    else                 { r = 1 - (t - 0.875) * 4; g = 0; b = 0; }
    return [Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b))];
  }

  function hotColor(t) {
    t = Math.max(0, Math.min(1, t));
    var r, g, b;
    if (t < 0.33)      { r = t / 0.33; g = 0; b = 0; }
    else if (t < 0.66) { r = 1; g = (t - 0.33) / 0.33; b = 0; }
    else               { r = 1; g = 1; b = (t - 0.66) / 0.34; }
    return [r, g, b];
  }

  // ── Surface creation ──────────────────────────────────────────────

  var CANVAS_IDS = ['three-canvas-original', 'three-canvas-predicted', 'three-canvas-error'];

  function init(_nx, _ny, _gridX, _gridY, _cylinderMask) {
    nx = _nx; ny = _ny; gridX = _gridX; gridY = _gridY; cylinderMask = _cylinderMask;

    for (var s = 0; s < 3; s++) {
      var canvas = document.getElementById(CANVAS_IDS[s]);
      if (!canvas) continue;
      var container = canvas.parentElement; // .three-canvas-wrap

      var w = container.clientWidth || 600;
      var h = container.clientHeight || 300;

      var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(w, h);
      renderer.setClearColor(0x0a0a0f);

      var scene = new THREE.Scene();
      var camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
      camera.position.set(0.09, 0.20, 0.20);
      camera.lookAt(0.09, 0.03, 0);

      scene.add(new THREE.AmbientLight(0xffffff, 0.6));
      var d = new THREE.DirectionalLight(0xffffff, 0.7);
      d.position.set(1, 2, 1);
      scene.add(d);

      var OC = THREE.OrbitControls || window.OrbitControls;
      var controls = null;
      if (OC) {
        controls = new OC(camera, renderer.domElement);
        controls.target.set(0.09, 0.03, 0);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        // Only the first surface gets interactive controls
        if (s > 0) controls.enabled = false;
        controls.update();
      }

      var mesh = buildSurface(scene);

      surfaces.push({
        renderer: renderer,
        scene: scene,
        camera: camera,
        controls: controls,
        mesh: mesh,
        canvas: canvas,
        container: container
      });
    }

    // Link controls: when the first surface's controls change, sync all others
    if (surfaces.length > 0 && surfaces[0].controls) {
      surfaces[0].controls.addEventListener('change', syncCameras);
    }

    window.addEventListener('resize', onResize);
  }

  function syncCameras() {
    var src = surfaces[0];
    for (var i = 1; i < surfaces.length; i++) {
      var dst = surfaces[i];
      dst.camera.position.copy(src.camera.position);
      dst.camera.quaternion.copy(src.camera.quaternion);
      if (dst.controls) {
        dst.controls.target.copy(src.controls.target);
        dst.controls.update();
      }
    }
  }

  function buildSurface(scene) {
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
        var a = ix * ny + iy, b = (ix + 1) * ny + iy;
        var c = (ix + 1) * ny + (iy + 1), d2 = ix * ny + (iy + 1);
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
    var mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
    return mesh;
  }

  // ── Update all three surfaces ─────────────────────────────────────

  function updateSurfaces(origField, predField, errorField, maxErr) {
    if (surfaces.length < 3) return;
    var n = nx * ny;
    maxErr = maxErr || 1;

    // Original
    updateMesh(surfaces[0].mesh, origField, function (val) {
      return val * heightScale;
    }, function (val) {
      return jetColor(val);
    });

    // Predicted
    updateMesh(surfaces[1].mesh, predField, function (val) {
      return val * heightScale;
    }, function (val) {
      return jetColor(val);
    });

    // Absolute Error
    updateMesh(surfaces[2].mesh, errorField, function (val) {
      return (val / maxErr) * heightScale;
    }, function (val) {
      return hotColor(val / maxErr);
    });
  }

  function updateMesh(mesh, field, heightFn, colorFn) {
    if (!mesh) return;
    var positions = mesh.geometry.attributes.position.array;
    var colors = mesh.geometry.attributes.color.array;
    var n = nx * ny;

    for (var i = 0; i < n; i++) {
      var val = field[i];
      positions[i * 3 + 1] = cylinderMask[i] ? 0 : heightFn(val);
      var c = colorFn(val);
      colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
    }
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.attributes.color.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
  }

  // ── Render loop ───────────────────────────────────────────────────

  function startLoop() {
    active = true;
    function animate() {
      if (!active) return;
      animationId = requestAnimationFrame(animate);
      if (surfaces[0] && surfaces[0].controls) surfaces[0].controls.update();
      for (var i = 0; i < surfaces.length; i++) {
        surfaces[i].renderer.render(surfaces[i].scene, surfaces[i].camera);
      }
    }
    animate();
  }

  function stopLoop() {
    active = false;
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }

  // ── Resize ────────────────────────────────────────────────────────

  function onResize() {
    for (var i = 0; i < surfaces.length; i++) {
      var s = surfaces[i];
      var w = s.container.clientWidth;
      var h = s.container.clientHeight || 300;
      s.camera.aspect = w / h;
      s.camera.updateProjectionMatrix();
      s.renderer.setSize(w, h);
    }
  }

  function show() {
    onResize();
    startLoop();
  }

  function hide() {
    stopLoop();
  }

  return {
    init: init,
    updateSurfaces: updateSurfaces,
    show: show,
    hide: hide,
    startLoop: startLoop,
    stopLoop: stopLoop,
  };
})();

window.APP = APP;
