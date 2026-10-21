import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

const LABEL_COLORS = {
  chair: 0x00f3ff,
  tvmonitor: 0x9d00ff,
  diningtable: 0xff6200,
  laptop: 0x00ff73,
  person: 0xffd000,
  keyboard: 0xffffff,
  mouse: 0xa0a0a0,
  book: 0xff90c0,
  'cell phone': 0xff2070,
};

const Viewport3D = ({
  mapData, currentCamera, objects, person, audio,
  cameraMode, twinMode, denseMesh, meshVisible,
}) => {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const labelRendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);

  const pointCloudRef = useRef(null);
  const trajectoryLineRef = useRef(null);
  const cameraModelRef = useRef(null);
  const boxesGroupRef = useRef(new THREE.Group());
  const objectMeshesRef = useRef(new Map());

  // Person tracking box + label, and the audio-source direction ray.
  const personBoxRef = useRef(null);
  const personLabelRef = useRef(null);
  const audioRayRef = useRef(null);

  // Digital twin
  const adaptiveTwinRef = useRef(null);
  const fixedTwinRef = useRef(null);

  // Dense TSDF mesh (Path B)
  const denseMeshRef = useRef(null);
  const denseMeshWireRef = useRef(null);
  const lastMeshExtractedAtRef = useRef(0);

  // 1. Initialise scene
  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x06070b);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.05, 100);
    camera.position.set(0, 1.5, -3);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // CSS2D label overlay
    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(width, height);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0';
    labelRenderer.domElement.style.left = '0';
    labelRenderer.domElement.style.pointerEvents = 'none';
    containerRef.current.appendChild(labelRenderer.domElement);
    labelRendererRef.current = labelRenderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 20;
    controls.minDistance = 0.2;
    controlsRef.current = controls;

    const gridHelper = new THREE.GridHelper(10, 20, 0x00f3ff, 0x161c2d);
    gridHelper.position.y = -1.0;
    scene.add(gridHelper);

    const axesHelper = new THREE.AxesHelper(1);
    axesHelper.position.set(0, -0.99, 0);
    scene.add(axesHelper);

    scene.add(new THREE.AmbientLight(0xffffff, 0.3));
    const pointLight = new THREE.PointLight(0xffffff, 0.8, 20);
    pointLight.position.set(0, 5, 0);
    scene.add(pointLight);

    // Point cloud
    const ptGeo = new THREE.BufferGeometry();
    const ptMat = new THREE.PointsMaterial({
      size: 0.03, vertexColors: true, sizeAttenuation: true,
    });
    const pointCloud = new THREE.Points(ptGeo, ptMat);
    scene.add(pointCloud);
    pointCloudRef.current = pointCloud;

    // Trajectory ribbon: per-segment colour with time gradient
    const trajGeo = new THREE.BufferGeometry();
    const trajMat = new THREE.LineBasicMaterial({ vertexColors: true });
    const trajLine = new THREE.Line(trajGeo, trajMat);
    scene.add(trajLine);
    trajectoryLineRef.current = trajLine;

    scene.add(boxesGroupRef.current);

    const camGeo = new THREE.ConeGeometry(0.15, 0.3, 4);
    camGeo.rotateX(Math.PI / 2);
    const camMat = new THREE.MeshBasicMaterial({ color: 0x00f3ff, wireframe: true });
    const cameraModel = new THREE.Mesh(camGeo, camMat);
    scene.add(cameraModel);
    cameraModelRef.current = cameraModel;

    // --- Person tracking box (human-sized wireframe) -------------------
    const personGeo = new THREE.BoxGeometry(0.6, 1.7, 0.6);
    const personMat = new THREE.MeshBasicMaterial({
      color: 0xffd000, transparent: true, opacity: 0.12, depthWrite: false,
    });
    const personBox = new THREE.Mesh(personGeo, personMat);
    personBox.add(new THREE.LineSegments(
      new THREE.EdgesGeometry(personGeo),
      new THREE.LineBasicMaterial({ color: 0xffd000 }),
    ));
    const personLabelDiv = document.createElement('div');
    personLabelDiv.style.cssText =
      'background: rgba(10,14,23,0.85); border: 1px solid rgba(255,208,0,0.7);' +
      'color: #ffd000; font-family: var(--font-mono, monospace); font-size: 10px;' +
      'padding: 3px 7px; border-radius: 3px; pointer-events: none;' +
      'white-space: nowrap; transform: translateY(-46px);';
    personLabelDiv.textContent = 'PERSON';
    const personLabel = new CSS2DObject(personLabelDiv);
    personBox.add(personLabel);
    personBox.visible = false;
    scene.add(personBox);
    personBoxRef.current = personBox;
    personLabelRef.current = personLabel;

    // --- Audio source direction ray (camera → sound source) ------------
    const audioRayMat = new THREE.LineBasicMaterial({ color: 0xff2a2a });
    const audioRay = new THREE.Line(new THREE.BufferGeometry(), audioRayMat);
    audioRay.visible = false;
    scene.add(audioRay);
    audioRayRef.current = audioRay;

    // --- Digital Twin: adaptive (live bbox of point cloud) -------------
    const adaptiveGeo = new THREE.BoxGeometry(1, 1, 1);
    const adaptiveMat = new THREE.MeshBasicMaterial({
      color: 0x9d00ff, transparent: true, opacity: 0.04,
      depthWrite: false, wireframe: false,
    });
    const adaptiveMesh = new THREE.Mesh(adaptiveGeo, adaptiveMat);
    const adaptiveEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(adaptiveGeo),
      new THREE.LineBasicMaterial({ color: 0x9d00ff }),
    );
    adaptiveMesh.add(adaptiveEdges);
    adaptiveMesh.visible = false;
    scene.add(adaptiveMesh);
    adaptiveTwinRef.current = adaptiveMesh;

    // --- Dense TSDF mesh (Path B) -------------------------------------
    // A real surface mesh streamed from Open3D's marching-cubes extraction.
    // Starts empty; geometry is populated each time a new mesh arrives.
    const denseGeo = new THREE.BufferGeometry();
    const denseMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      flatShading: true,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
      metalness: 0.0,
      roughness: 0.9,
    });
    const denseMeshObj = new THREE.Mesh(denseGeo, denseMat);
    denseMeshObj.visible = false;
    scene.add(denseMeshObj);
    denseMeshRef.current = denseMeshObj;

    // Subtle wireframe overlay built lazily; we don't want to call
    // WireframeGeometry on an empty BufferGeometry during initial mount
    // (some Three.js builds are picky about that). Start with an empty
    // LineSegments and populate when the first real mesh arrives.
    const denseWireMat = new THREE.LineBasicMaterial({
      color: 0x00f3ff, transparent: true, opacity: 0.12,
    });
    const denseWireObj = new THREE.LineSegments(
      new THREE.BufferGeometry(), denseWireMat,
    );
    denseWireObj.visible = false;
    scene.add(denseWireObj);
    denseMeshWireRef.current = denseWireObj;

    // --- Digital Twin: fixed reference room (4 × 3 × 4 m) --------------
    const fixedGroup = new THREE.Group();
    const roomW = 4, roomH = 3, roomD = 4;
    const wallMat = new THREE.MeshBasicMaterial({
      color: 0x00f3ff, transparent: true, opacity: 0.05,
      side: THREE.BackSide, depthWrite: false,
    });
    const wallMesh = new THREE.Mesh(new THREE.BoxGeometry(roomW, roomH, roomD), wallMat);
    wallMesh.position.set(0, roomH / 2 - 1.0, 2.0);
    const wallEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(roomW, roomH, roomD)),
      new THREE.LineBasicMaterial({ color: 0x00f3ff }),
    );
    wallEdges.position.copy(wallMesh.position);
    fixedGroup.add(wallMesh);
    fixedGroup.add(wallEdges);
    fixedGroup.visible = false;
    scene.add(fixedGroup);
    fixedTwinRef.current = fixedGroup;

    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
      labelRendererRef.current.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      if (controlsRef.current) controlsRef.current.update();
      rendererRef.current.render(sceneRef.current, cameraRef.current);
      labelRendererRef.current.render(sceneRef.current, cameraRef.current);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', handleResize);
      objectMeshesRef.current.forEach(({ mesh, outline, label }) => {
        mesh.geometry.dispose();
        mesh.material.dispose();
        outline.geometry.dispose();
        outline.material.dispose();
        if (label && label.element && label.element.parentNode) {
          label.element.parentNode.removeChild(label.element);
        }
      });
      objectMeshesRef.current.clear();
      // Dense mesh disposal
      if (denseMeshRef.current) {
        denseMeshRef.current.geometry.dispose();
        denseMeshRef.current.material.dispose();
      }
      if (denseMeshWireRef.current) {
        denseMeshWireRef.current.geometry.dispose();
        denseMeshWireRef.current.material.dispose();
      }
      if (personLabelRef.current?.element?.parentNode) {
        personLabelRef.current.element.parentNode.removeChild(personLabelRef.current.element);
      }
      labelRendererRef.current?.domElement?.remove();
      rendererRef.current?.domElement?.remove();
      rendererRef.current?.dispose();
    };
  }, []);

  // Dense TSDF mesh updates: replace the BufferGeometry whenever a new
  // snapshot arrives. The mesh broadcaster only fires when it's actually
  // new (extracted_at changes), so we still gate on that here as a safety.
  useEffect(() => {
    if (!denseMeshRef.current || !denseMesh) return;
    if (!denseMesh.extracted_at || denseMesh.extracted_at === lastMeshExtractedAtRef.current) return;
    lastMeshExtractedAtRef.current = denseMesh.extracted_at;

    const verts = denseMesh.vertices || [];
    const tris = denseMesh.triangles || [];
    const cols = denseMesh.colors || [];

    if (verts.length < 3 || tris.length < 3) {
      denseMeshRef.current.visible = false;
      if (denseMeshWireRef.current) denseMeshWireRef.current.visible = false;
      return;
    }

    const posArr = new Float32Array(verts);
    const idxArr = (verts.length / 3) > 65535
      ? new Uint32Array(tris)
      : new Uint16Array(tris);

    const geo = denseMeshRef.current.geometry;
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setIndex(new THREE.BufferAttribute(idxArr, 1));
    if (cols.length === verts.length) {
      geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(cols), 3));
    }
    geo.computeVertexNormals();
    geo.computeBoundingSphere();

    // Refresh the wireframe overlay to match
    if (denseMeshWireRef.current) {
      const oldWire = denseMeshWireRef.current.geometry;
      denseMeshWireRef.current.geometry = new THREE.WireframeGeometry(geo);
      oldWire.dispose();
    }
  }, [denseMesh]);

  // Mesh visibility toggle
  useEffect(() => {
    if (denseMeshRef.current) denseMeshRef.current.visible = !!meshVisible;
    if (denseMeshWireRef.current) denseMeshWireRef.current.visible = !!meshVisible;
  }, [meshVisible]);

  // 2. Point cloud + heatmap (we colour points by proximity to objects)
  useEffect(() => {
    if (!pointCloudRef.current || !mapData.points) return;
    const n = mapData.points.length;
    const ptsArr = new Float32Array(n * 3);
    const colArr = new Float32Array(n * 3);

    // Pre-flatten object centroids into a typed array so the hot loop
    // does no property lookups. Also pre-compute a squared cutoff so the
    // inner loop can early-out without calling Math.exp on far points.
    const m = objects.length;
    const cArr = new Float32Array(m * 3);
    for (let j = 0; j < m; j++) {
      const c = objects[j].centroid_3d;
      cArr[j * 3] = c[0];
      cArr[j * 3 + 1] = c[1];
      cArr[j * 3 + 2] = c[2];
    }
    // exp(-d2 * 2.5) < 0.01 implies d2 > ln(100)/2.5 ≈ 1.84
    const CUTOFF_D2 = 1.85;

    for (let i = 0; i < n; i++) {
      const p = mapData.points[i];
      const px = p[0], py = p[1], pz = p[2];
      ptsArr[i * 3] = px;
      ptsArr[i * 3 + 1] = py;
      ptsArr[i * 3 + 2] = pz;

      let cR = 1.0, cG = 1.0, cB = 1.0;
      const baseCol = mapData.colors[i];
      if (baseCol) {
        cR = baseCol[0] / 255;
        cG = baseCol[1] / 255;
        cB = baseCol[2] / 255;
      }

      let heat = 0.0;
      for (let j = 0; j < m; j++) {
        const j3 = j * 3;
        const dx = cArr[j3] - px;
        const dy = cArr[j3 + 1] - py;
        const dz = cArr[j3 + 2] - pz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > CUTOFF_D2) continue;             // negligible influence
        const h = Math.exp(-d2 * 2.5);
        if (h > heat) heat = h;
      }
      if (heat > 0.01) {
        const inv = 1 - heat;
        cR = cR * inv + heat;
        cG = cG * inv + heat * 0.35;
        cB = cB * inv + heat * 0.1;
        if (cR > 1) cR = 1;
        if (cG > 1) cG = 1;
        if (cB > 1) cB = 1;
      }
      colArr[i * 3] = cR;
      colArr[i * 3 + 1] = cG;
      colArr[i * 3 + 2] = cB;
    }

    const geom = pointCloudRef.current.geometry;
    geom.setAttribute('position', new THREE.BufferAttribute(ptsArr, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
    geom.computeBoundingSphere();

    // Trajectory ribbon: gradient by index (old=dim, new=bright green)
    if (mapData.trajectory && mapData.trajectory.length > 0) {
      const m = mapData.trajectory.length;
      const lp = new Float32Array(m * 3);
      const lc = new Float32Array(m * 3);
      for (let i = 0; i < m; i++) {
        lp[i * 3] = mapData.trajectory[i][0];
        lp[i * 3 + 1] = mapData.trajectory[i][1];
        lp[i * 3 + 2] = mapData.trajectory[i][2];
        const t = i / Math.max(1, m - 1);
        lc[i * 3] = 0.0;
        lc[i * 3 + 1] = 0.4 + 0.6 * t;
        lc[i * 3 + 2] = 0.45 * (1 - t);
      }
      trajectoryLineRef.current.geometry.setAttribute('position',
        new THREE.BufferAttribute(lp, 3));
      trajectoryLineRef.current.geometry.setAttribute('color',
        new THREE.BufferAttribute(lc, 3));
      trajectoryLineRef.current.material.vertexColors = true;
    }

    // Adaptive digital twin: AABB of the *inlier* point cloud.
    // We use the 5th–95th percentile per axis so a handful of drift
    // outliers (which the SLAM map collects when tracking flickers) don't
    // blow the twin up to absurd size.
    if (adaptiveTwinRef.current && n > 50) {
      const xs = new Float32Array(n);
      const ys = new Float32Array(n);
      const zs = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        xs[i] = ptsArr[i * 3];
        ys[i] = ptsArr[i * 3 + 1];
        zs[i] = ptsArr[i * 3 + 2];
      }
      const pct = (arr, q) => {
        const a = arr.slice().sort();
        const idx = Math.max(0, Math.min(a.length - 1, Math.floor(q * (a.length - 1))));
        return a[idx];
      };
      const minX = pct(xs, 0.05), maxX = pct(xs, 0.95);
      const minY = pct(ys, 0.05), maxY = pct(ys, 0.95);
      const minZ = pct(zs, 0.05), maxZ = pct(zs, 0.95);
      const sx = Math.max(0.5, maxX - minX);
      const sy = Math.max(0.5, maxY - minY);
      const sz = Math.max(0.5, maxZ - minZ);
      adaptiveTwinRef.current.scale.set(sx, sy, sz);
      adaptiveTwinRef.current.position.set(
        (minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2,
      );
    }
  }, [mapData, objects]);

  // 3. Camera pose
  useEffect(() => {
    if (!cameraModelRef.current || !currentCamera) return;
    const m = currentCamera.matrix;
    const mat = new THREE.Matrix4().set(
      m[0][0], m[0][1], m[0][2], m[0][3],
      m[1][0], m[1][1], m[1][2], m[1][3],
      m[2][0], m[2][1], m[2][2], m[2][3],
      m[3][0], m[3][1], m[3][2], m[3][3],
    );
    cameraModelRef.current.position.setFromMatrixPosition(mat);
    cameraModelRef.current.quaternion.setFromRotationMatrix(mat);

    if (cameraMode === 'follow') {
      controlsRef.current.enabled = false;
      const tp = new THREE.Vector3().setFromMatrixPosition(mat);
      const dir = new THREE.Vector3(0, 0, 1).applyMatrix4(mat).sub(tp).normalize();
      const fp = tp.clone().sub(dir.multiplyScalar(2.0)).add(new THREE.Vector3(0, 0.8, 0));
      cameraRef.current.position.lerp(fp, 0.1);
      controlsRef.current.target.lerp(tp, 0.1);
    } else if (cameraMode === 'topdown') {
      controlsRef.current.enabled = false;
      const tp = new THREE.Vector3().setFromMatrixPosition(mat);
      cameraRef.current.position.set(tp.x, tp.y + 6.0, tp.z);
      controlsRef.current.target.copy(tp);
    } else {
      controlsRef.current.enabled = true;
    }
  }, [currentCamera, cameraMode]);

  // 4. Object meshes + persistent CSS labels
  useEffect(() => {
    if (!boxesGroupRef.current) return;

    const map = objectMeshesRef.current;
    const seen = new Set();

    objects.forEach((obj) => {
      seen.add(obj.id);
      const [dx, dy, dz] = obj.bbox_3d;
      const [x, y, z] = obj.centroid_3d;
      let color = LABEL_COLORS[obj.label] ?? 0x00f3ff;

      let entry = map.get(obj.id);
      if (!entry) {
        const geo = new THREE.BoxGeometry(dx, dy, dz);
        const mat = new THREE.MeshBasicMaterial({
          color, transparent: true,
          opacity: 0.08, depthWrite: false,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x, y, z);
        const edges = new THREE.EdgesGeometry(geo);
        const lineMat = new THREE.LineBasicMaterial({ color });
        const outline = new THREE.LineSegments(edges, lineMat);
        mesh.add(outline);

        const labelDiv = document.createElement('div');
        labelDiv.className = 'object-label-3d';
        labelDiv.style.cssText =
          'background: rgba(10,14,23,0.85); border: 1px solid rgba(255,255,255,0.15);' +
          'color: #fff; font-family: var(--font-mono, monospace); font-size: 10px;' +
          'padding: 3px 7px; border-radius: 3px; pointer-events: none;' +
          'white-space: nowrap; transform: translateY(-40px);';
        labelDiv.textContent = `${obj.label} #${obj.id}`;
        const label = new CSS2DObject(labelDiv);
        mesh.add(label);

        boxesGroupRef.current.add(mesh);
        entry = { mesh, outline, label, currentColor: color, dims: [dx, dy, dz] };
        map.set(obj.id, entry);
      } else {
        const { mesh, outline, label, dims } = entry;
        mesh.position.set(x, y, z);
        if (
          Math.abs(dims[0] - dx) > 1e-3 ||
          Math.abs(dims[1] - dy) > 1e-3 ||
          Math.abs(dims[2] - dz) > 1e-3
        ) {
          mesh.geometry.dispose();
          mesh.geometry = new THREE.BoxGeometry(dx, dy, dz);
          outline.geometry.dispose();
          outline.geometry = new THREE.EdgesGeometry(mesh.geometry);
          entry.dims = [dx, dy, dz];
        }
        if (entry.currentColor !== color) {
          mesh.material.color.setHex(color);
          outline.material.color.setHex(color);
          entry.currentColor = color;
        }
        mesh.material.opacity = 0.08;
        if (label && label.element) {
          label.element.textContent = `${obj.label} #${obj.id}`;
        }
      }
    });

    for (const [id, entry] of map) {
      if (!seen.has(id)) {
        boxesGroupRef.current.remove(entry.mesh);
        entry.mesh.geometry.dispose();
        entry.mesh.material.dispose();
        entry.outline.geometry.dispose();
        entry.outline.material.dispose();
        if (entry.label && entry.label.element && entry.label.element.parentNode) {
          entry.label.element.parentNode.removeChild(entry.label.element);
        }
        map.delete(id);
      }
    }
  }, [objects]);

  // 5. Digital twin toggle
  useEffect(() => {
    if (!adaptiveTwinRef.current || !fixedTwinRef.current) return;
    adaptiveTwinRef.current.visible = (twinMode === 'adaptive' || twinMode === 'both');
    fixedTwinRef.current.visible = (twinMode === 'fixed' || twinMode === 'both');
  }, [twinMode]);

  // 6. Person tracking box + audio source-direction ray
  useEffect(() => {
    const box = personBoxRef.current;
    const ray = audioRayRef.current;
    if (!box || !ray) return;

    const pos = person && person.present ? person.position_3d : null;
    if (pos && pos.length === 3) {
      // Box centre is at the person's mid-height; raise it so the 1.7 m box
      // straddles them rather than sinking into the floor.
      box.position.set(pos[0], pos[1], pos[2]);
      box.visible = true;
      if (personLabelRef.current?.element) {
        personLabelRef.current.element.textContent =
          `PERSON ${Number(person.distance_m || 0).toFixed(2)}m`;
      }
      // Audio ray: camera → person, shown only while sound is detected.
      const speaking = audio && audio.vad_active && audio.source === 'person';
      if (speaking && currentCamera) {
        const c = currentCamera.position;
        ray.geometry.setAttribute('position', new THREE.BufferAttribute(
          new Float32Array([c[0], c[1], c[2], pos[0], pos[1], pos[2]]), 3));
        ray.geometry.computeBoundingSphere();
        ray.visible = true;
      } else {
        ray.visible = false;
      }
    } else {
      box.visible = false;
      ray.visible = false;
    }
  }, [person, audio, currentCamera]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }} />;
};

export default Viewport3D;
