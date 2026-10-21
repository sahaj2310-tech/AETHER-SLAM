import React, { useEffect, useRef } from 'react';
import {
  Camera, Brain, Box, Network, Cpu, Layers,
  Zap, ChevronRight,
} from 'lucide-react';

/**
 * Aether-SLAM landing page.
 *
 * Pure-CSS animated background: layered gradient mesh + SVG grid +
 * floating particles generated procedurally so we don't pull in any
 * extra runtime dependency for this view.
 */
const LandingPage = ({ onStart }) => {
  const particleHostRef = useRef(null);

  useEffect(() => {
    if (!particleHostRef.current) return;
    const host = particleHostRef.current;
    host.innerHTML = '';
    const N = 28;
    for (let i = 0; i < N; i++) {
      const dot = document.createElement('span');
      dot.className = 'landing-particle';
      const size = 1 + Math.random() * 3;
      const left = Math.random() * 100;
      const duration = 9 + Math.random() * 14;
      const delay = -Math.random() * duration;
      const hue = [
        '#00f3ff', '#9d00ff', '#00ff73', '#ff6200', '#ffd000',
      ][Math.floor(Math.random() * 5)];
      dot.style.cssText = `
        width:${size}px; height:${size}px;
        left:${left}%;
        background:${hue};
        box-shadow:0 0 ${size * 4}px ${hue};
        animation-duration:${duration}s;
        animation-delay:${delay}s;
      `;
      host.appendChild(dot);
    }
    return () => { host.innerHTML = ''; };
  }, []);

  const stages = [
    { num: '01', label: 'CAPTURE',  desc: 'One monocular camera at 30 Hz: a webcam or a recorded sequence replayed live.' },
    { num: '02', label: 'TRACK',    desc: 'DISK learned features + LightGlue matching → PnP-RANSAC + LM pose.' },
    { num: '03', label: 'TRIANGULATE', desc: 'Map points with enough parallax are re-triangulated: real geometry, not network guesses.' },
    { num: '04', label: 'PERCEIVE', desc: 'YOLOv8n-seg person boxes mask people out of tracking and fusion.' },
    { num: '05', label: 'DEPTH',    desc: 'DPT-Hybrid relative depth, aligned by confidence-gated scale-shift filtering (CG-SSF).' },
    { num: '06', label: 'FUSE',     desc: 'Uncertainty-weighted TSDF on the GPU → live marching-cubes mesh.' },
    { num: '07', label: 'EXPORT',   desc: 'Full-resolution vertex-coloured mesh as .ply.' },
  ];

  const features = [
    { icon: <Camera size={20} />,  title: 'Visual Odometry',
      text: 'DISK + LightGlue (FP16 detector) with PnP-RANSAC, parallax-gated triangulation of the depth anchors.' },
    { icon: <Layers size={20} />,  title: 'CG-SSF Depth Alignment',
      text: 'A robust closed-form fit with covariance, an observability test, rank-one updates and a χ² gate keep the network depth at one consistent scale.' },
    { icon: <Box size={20} />,     title: 'Weighted Dense Fusion',
      text: 'Per-pixel inverse-variance weights from the alignment drive a sparse voxel-block TSDF on the GPU.' },
    { icon: <Brain size={20} />,   title: 'People Out of the Map',
      text: 'Person boxes are removed from features, anchors and fused depth, so you reconstruct the room, not yourself.' },
    { icon: <Network size={20} />, title: 'Bounded Staleness',
      text: 'Capacity-one overwrite buffers between stages: every output is at most 2·ΣLᵢ old, measured live at /api/perf.' },
    { icon: <Cpu size={20} />,     title: 'Fits in 4 GB',
      text: 'Three networks + dense fusion in one process, one CUDA context, a stream per stage. No NeRF, no splatting.' },
  ];

  return (
    <div className="landing-root">
      <div className="landing-bg-mesh" />
      <div className="landing-bg-grid" />
      <div ref={particleHostRef} className="landing-particles" />

      {/* HERO */}
      <section className="landing-hero">
        <div className="landing-eyebrow">
          <span className="landing-pulse-dot" />
          v3.0 · CG-SSF · CUDA FP16
        </div>

        <h1 className="landing-title">
          <span className="landing-title-word">AETHER</span>
          <span className="landing-title-dash">-</span>
          <span className="landing-title-word landing-title-accent">SLAM</span>
        </h1>

        <p className="landing-tagline">
          A Live 3D Room Scanner from a Single Webcam
        </p>

        <p className="landing-subtagline">
          Point a normal webcam at a room and move it around, and AETHER-SLAM builds
          a textured 3D model of the real space <em>live</em>, then lets
          you export it as a <code>.ply</code> mesh for Blender, MeshLab, or any
          3D tool. No depth sensor, no LiDAR: just one camera.
        </p>

        <div className="landing-pills">
          <span className="landing-pill">DISK + LightGlue</span>
          <span className="landing-pill">CG-SSF Depth Alignment</span>
          <span className="landing-pill">Weighted GPU TSDF</span>
          <span className="landing-pill">Person Tracking</span>
          <span className="landing-pill">.PLY Export</span>
        </div>

        <button className="landing-cta" onClick={onStart}>
          <span>LAUNCH DASHBOARD</span>
          <ChevronRight size={20} />
          <div className="landing-cta-glow" />
        </button>

        <div className="landing-scroll-hint">
          ↓ scroll to see what it does
        </div>
      </section>

      {/* WHAT IT DOES */}
      <section className="landing-section">
        <div className="landing-section-head">
          <Box size={18} />
          <h2>What it does</h2>
          <div className="landing-section-rule" />
        </div>

        <div className="landing-features">
          {[
            { t: '🏠 Reconstructs your room in 3D',
              d: 'As you pan the webcam, the live dense surface mesh of the actual room builds up in the viewport, walls and furniture included.' },
            { t: '📐 Measures the space',
              d: 'Auto-estimates the scanned floor area and the room’s width / depth / height from the mesh, with a top-down floor-plan view.' },
            { t: '🧍 Tracks people & distance',
              d: 'Reports the live distance from the camera to a person, draws a 3D tracking box, and masks them out of the reconstruction so you map the room, not yourself.' },
            { t: '🔊 Detects sound & its direction',
              d: 'Microphone voice-activity detection, with a source-direction indicator pointing at whoever is speaking.' },
            { t: '💾 Exports the model',
              d: 'One click downloads room_mesh.ply, a real, vertex-coloured 3D model you can open in any 3D software.' },
            { t: '⚡ Runs on consumer hardware',
              d: 'A lean, multi-threaded GPU pipeline whose whole footprint fits in a 4 GB GPU. No cloud, no special hardware.' },
          ].map((f, i) => (
            <div key={f.t} className="landing-feature" style={{ animationDelay: `${i * 60}ms` }}>
              <h3>{f.t}</h3>
              <p>{f.d}</p>
            </div>
          ))}
        </div>

        <p style={{
          textAlign: 'center', marginTop: 28, color: 'var(--color-text-secondary, #9aa)',
          fontSize: 14, maxWidth: 640, marginInline: 'auto', lineHeight: 1.6,
        }}>
          <strong>How to use it:</strong> launch the dashboard, then move the
          webcam <strong>slowly sideways and forward</strong> around the room
          (don&rsquo;t just rotate it in place). Watch the mesh fill in, then hit
          <strong> EXPORT .PLY</strong>.
        </p>
      </section>

      {/* HOW IT WORKS */}
      <section className="landing-section">
        <div className="landing-section-head">
          <Zap size={18} />
          <h2>How it works</h2>
          <div className="landing-section-rule" />
        </div>

        <div className="landing-stages">
          {stages.map((s, i) => (
            <div
              key={s.num}
              className="landing-stage"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className="landing-stage-num">{s.num}</div>
              <div className="landing-stage-body">
                <h3>{s.label}</h3>
                <p>{s.desc}</p>
              </div>
              {i < stages.length - 1 && <div className="landing-stage-link" />}
            </div>
          ))}
        </div>
      </section>

      {/* FEATURE GRID */}
      <section className="landing-section">
        <div className="landing-section-head">
          <Layers size={18} />
          <h2>Inside the engine</h2>
          <div className="landing-section-rule" />
        </div>

        <div className="landing-features">
          {features.map((f, i) => (
            <div
              key={f.title}
              className="landing-feature"
              style={{ animationDelay: `${i * 60}ms` }}
            >
              <div className="landing-feature-icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA REPEAT */}
      <section className="landing-section landing-section-cta">
        <h2 className="landing-end-headline">
          Ready when you are.
        </h2>
        <button className="landing-cta" onClick={onStart}>
          <span>LAUNCH DASHBOARD</span>
          <ChevronRight size={20} />
          <div className="landing-cta-glow" />
        </button>
      </section>

      {/* FOOTER */}
      <footer className="landing-footer">
        <div className="landing-footer-row">
          <div className="landing-footer-stack">
            <span>FastAPI</span><span>·</span>
            <span>React</span><span>·</span>
            <span>Three.js</span><span>·</span>
            <span>PyTorch</span><span>·</span>
            <span>CUDA</span>
          </div>
        </div>
        <div className="landing-credit">
          Built by <span className="landing-credit-name">Sahaj Sinha</span>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
