import React, { useState, useEffect, useRef } from 'react';
import {
  Activity, Camera, Database, Mic, MicOff,
  Terminal as ConsoleIcon, MapPin, Search, Boxes, GitBranch, Box,
  MessageSquare, Save,
} from 'lucide-react';
import Viewport3D from './components/Viewport3D';
import LandingPage from './components/LandingPage';
import ErrorBoundary from './components/ErrorBoundary';

const BACKEND_HOST =
  import.meta.env.VITE_BACKEND_HOST || `${window.location.hostname}:8000`;
const WS_URL = `ws://${BACKEND_HOST}/ws`;
const HTTP_URL = `http://${BACKEND_HOST}`;

const App = () => {
  const [started, setStarted] = useState(false);
  const [connected, setConnected] = useState(false);
  const [camera, setCamera] = useState({
    position: [0, 0, 0], orientation: [1, 0, 0, 0],
    matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  });
  const [mapData, setMapData] = useState({ points: [], colors: [], trajectory: [] });
  const [objects, setObjects] = useState([]);
  const [sceneGraph, setSceneGraph] = useState({ nodes: [], edges: [] });
  const [sceneDescription, setSceneDescription] = useState([]);
  const [loopEvents, setLoopEvents] = useState([]);

  const [telemetry, setTelemetry] = useState({
    fps: 0, latency_ms: 0, slam_ms: 0, semantic_ms: 0,
    tracked_points_count: 0, keyframes_count: 0,
    ram_pct: 0, vram_mb: 0, status: 'INIT', feature_backend: '?',
  });
  const [trackingStatus, setTrackingStatus] = useState('INIT');
  const [videoFrame, setVideoFrame] = useState(null);
  const [videoSize, setVideoSize] = useState('640×480');
  const [audio, setAudio] = useState({
    available: false, energy: 0, level: 0, vad_active: false, type: 'no_mic',
    azimuth_deg: 0, source: 'none',
  });
  const [person, setPerson] = useState({
    present: false, distance_m: 0, azimuth_deg: 0, conf: 0,
  });

  const [cameraMode, setCameraMode] = useState('free');
  // Default OFF: the decorative twin boxes just clutter the real dense mesh.
  const [twinMode, setTwinMode] = useState('off'); // adaptive | fixed | both | off
  const [denseMesh, setDenseMesh] = useState(null);
  // Start hidden so we can never render an empty Three.js geometry on mount.
  const [meshVisible, setMeshVisible] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [aiInput, setAiInput] = useState('');
  const [aiAnswer, setAiAnswer] = useState(null);
  const [aiBusy, setAiBusy] = useState(false);
  const [logs, setLogs] = useState([
    { time: new Date().toLocaleTimeString(), text: 'System initialised. Waiting for backend...', type: 'info' },
  ]);

  const wsRef = useRef(null);
  const knownObjectIdsRef = useRef(new Set());
  const knownLoopKeysRef = useRef(new Set());
  // Auto-reveal the dense reconstruction the first time a real mesh lands,
  // so the live surface shows up without the user toggling MESH on. We only
  // force it once; after that the toggle is fully under user control.
  const meshAutoShownRef = useRef(false);

  useEffect(() => {
    if (!started) return;          // hold the WS until user enters the dashboard
    let reconnectTimer = null;
    const connectWS = () => {
      addLog('Connecting to backend: ' + WS_URL, 'info');
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        addLog('WebSocket link established. Spatial tracking active.', 'slam');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.msg === 'video') {
            setVideoFrame(data.video_frame);
            return;
          }
          if (data.msg === 'mesh') {
            setDenseMesh({
              vertices: data.vertices,
              triangles: data.triangles,
              colors: data.colors,
              n_vertices: data.n_vertices,
              n_triangles: data.n_triangles,
              extracted_at: data.extracted_at,
              dimensions: data.dimensions || {},
            });
            if (data.n_vertices > 0 && !meshAutoShownRef.current) {
              meshAutoShownRef.current = true;
              setMeshVisible(true);
            }
            addLog(`Dense mesh updated: ${data.n_vertices} v / ${data.n_triangles} f`, 'slam');
            return;
          }
          if (data.msg === 'map') {
            setMapData({
              points: data.points || [],
              colors: data.colors || [],
              trajectory: data.trajectory || [],
            });
            return;
          }

          // Light state message (pose/objects/telemetry).
          if (data.camera) setCamera(data.camera);

          if (data.objects) {
            setObjects(data.objects);
            const known = knownObjectIdsRef.current;
            data.objects.forEach((obj) => {
              if (!known.has(obj.id)) {
                known.add(obj.id);
                addLog(`Object registered: ${obj.label} #${obj.id}`, 'slam');
              }
            });
            const live = new Set(data.objects.map((o) => o.id));
            for (const id of known) if (!live.has(id)) known.delete(id);
          }
          if (data.scene_graph) setSceneGraph(data.scene_graph);
          if (data.scene_description) setSceneDescription(data.scene_description);

          if (data.loop_events) {
            setLoopEvents(data.loop_events);
            const seen = knownLoopKeysRef.current;
            data.loop_events.forEach((ev) => {
              const k = `${ev.current_kf_id}-${ev.matched_kf_id}`;
              if (!seen.has(k)) {
                seen.add(k);
                addLog(`Loop closure: kf${ev.current_kf_id} ⇄ kf${ev.matched_kf_id} (score ${ev.score.toFixed(2)})`, 'slam');
              }
            });
          }

          if (data.audio) setAudio(data.audio);
          if (data.person) setPerson(data.person);
          if (data.telemetry) setTelemetry(data.telemetry);
          if (data.tracking_status) setTrackingStatus(data.tracking_status);
        } catch (err) {
          console.error('WS parse error:', err);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        addLog('Connection lost. Reconnecting in 2s...', 'error');
        reconnectTimer = setTimeout(connectWS, 2000);
      };
      ws.onerror = (err) => { console.error('WS error:', err); ws.close(); };
    };
    connectWS();
    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, [started]);

  const addLog = (text, type = 'info') => {
    setLogs((prev) => [
      { time: new Date().toLocaleTimeString(), text, type },
      ...prev.slice(0, 49),
    ]);
  };

  const sendReset = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send('reset');
      addLog('Reset requested.', 'info');
    }
  };

  const sendQuery = async () => {
    const text = aiInput.trim();
    if (!text) return;
    setAiBusy(true);
    try {
      const res = await fetch(`${HTTP_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      setAiAnswer(data);
      addLog(`Q: "${text}" → ${data.summary}`, data.confidence > 0.5 ? 'slam' : 'warn');
    } catch (e) {
      setAiAnswer({ summary: 'Query failed.', confidence: 0 });
      addLog('Spatial query failed.', 'error');
    } finally {
      setAiBusy(false);
    }
  };

  const saveSession = async () => {
    addLog('Saving session...', 'info');
    try {
      const res = await fetch(`${HTTP_URL}/api/save`, { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        addLog(`Saved ${data.objects} objects / ${data.keyframes} keyframes.`, 'slam');
      } else {
        addLog('Save failed.', 'error');
      }
    } catch {
      addLog('Save failed.', 'error');
    }
  };

  const exportMesh = async () => {
    addLog('Exporting 3D room mesh...', 'info');
    try {
      const res = await fetch(`${HTTP_URL}/api/export/mesh`);
      const ct = res.headers.get('content-type') || '';
      if (ct.includes('application/json')) {
        const j = await res.json();
        addLog('Export: ' + (j.message || 'no mesh available'), 'warn');
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'room_mesh.ply';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      addLog('Room mesh exported → room_mesh.ply', 'slam');
    } catch {
      addLog('Mesh export failed.', 'error');
    }
  };

  const filteredObjects = objects.filter((o) =>
    o.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
    String(o.id).includes(searchQuery)
  );

  const statusColor =
    trackingStatus === 'TRACKING' ? 'var(--accent-green)' :
    trackingStatus === 'LOST' ? 'var(--accent-red)' :
    trackingStatus === 'NO_CAMERA' ? 'var(--accent-red)' :
    trackingStatus === 'RESET' ? 'var(--accent-orange)' :
    'var(--color-text-secondary)';

  if (!started) {
    return <LandingPage onStart={() => setStarted(true)} />;
  }

  return (
    <div className="dashboard-layout">
      <header className="app-header">
        <div className="logo-container">
          <div className="logo-glow" />
          <h1 className="logo-text">AETHER-SLAM</h1>
          <span className="logo-subtitle">Live 3D room scanner · single webcam</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="panel" style={{
            flexDirection: 'row', alignItems: 'center', gap: '6px',
            padding: '6px 12px', fontSize: '11px', fontFamily: 'var(--font-mono)',
            color: 'var(--accent-cyan)', border: '1px solid rgba(0,243,255,0.3)',
          }}>
            <Camera size={12} /> WEBCAM
          </div>
          <button className="cyber-button" onClick={exportMesh} title="Download the reconstructed room as a .PLY 3D model">
            <Save size={14} /> EXPORT .PLY
          </button>
          <button className="cyber-button" onClick={saveSession} title="Persist session to disk">
            <Save size={14} /> SAVE
          </button>
          <button className="cyber-button" onClick={sendReset} title="Reset SLAM + map">
            RESET
          </button>
          <div className="panel" style={{
            flexDirection: 'row', alignItems: 'center', gap: '8px',
            padding: '6px 12px', fontSize: '11px', fontFamily: 'var(--font-mono)',
          }}>
            <span className={connected ? 'badge-pulse' : 'badge-pulse lost'} />
            {connected ? 'LINK STABLE' : 'LINK OFFLINE'}
          </div>
        </div>
      </header>

      <aside className="sidebar-left panel">
        <div className="hud-card">
          <h3 className="card-title"><Camera size={16} /> Live Visual Overlays</h3>
          <div className="video-feed-container">
            {videoFrame ? (
              <img src={`data:image/jpeg;base64,${videoFrame}`} alt="cam" className="video-frame"
                onLoad={(e) => setVideoSize(`${e.currentTarget.naturalWidth}×${e.currentTarget.naturalHeight}`)} />
            ) : (
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                height: '100%', color: 'var(--color-text-muted)', fontSize: '12px',
              }}>
                {connected ? 'Awaiting video...' : 'Server offline.'}
              </div>
            )}
            <div className="overlay-scanlines" />
            <div className="overlay-hud">
              <span className="rec-dot" />
              <span>CAM // {videoSize}</span>
            </div>
          </div>
        </div>

        <div className="hud-card">
          <h3 className="card-title"><Activity size={16} /> Perception Telemetry</h3>
          <div className="telemetry-grid">
            <div className="telemetry-item">
              <span className="telemetry-label">FPS (fused)</span>
              <span className="telemetry-value green">{telemetry.fps}</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">FPS (tracking)</span>
              <span className="telemetry-value green">{telemetry.slam_fps ?? '–'}</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">SLAM</span>
              <span className="telemetry-value cyan">{telemetry.slam_ms}<span style={{ fontSize: 10 }}> ms</span></span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">YOLO+DEPTH</span>
              <span className="telemetry-value cyan">{telemetry.semantic_ms}<span style={{ fontSize: 10 }}> ms</span></span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">KEYFRAMES</span>
              <span className="telemetry-value orange">{telemetry.keyframes_count}</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">MAP POINTS</span>
              <span className="telemetry-value cyan">{telemetry.tracked_points_count}</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">RAM</span>
              <span className="telemetry-value">{telemetry.ram_pct}<span style={{ fontSize: 10 }}> %</span></span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">VRAM</span>
              <span className="telemetry-value">{telemetry.vram_mb}<span style={{ fontSize: 10 }}> MB</span></span>
            </div>
          </div>
          <h3 className="card-title" style={{ marginTop: 12 }}><Activity size={16} /> Depth Alignment (CG-SSF)</h3>
          <div className="telemetry-grid">
            <div className="telemetry-item">
              <span className="telemetry-label">UPDATE</span>
              <span className={`telemetry-value ${['full', 'init'].includes(telemetry.align_status) ? 'green' : telemetry.align_status === 'partial' ? 'cyan' : 'orange'}`}>
                {telemetry.align_status ?? '–'}
              </span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">ANCHORS</span>
              <span className="telemetry-value cyan">{telemetry.anchors ?? '–'}<span style={{ fontSize: 10 }}> / {telemetry.inliers ?? '–'}</span></span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">SCALE s</span>
              <span className="telemetry-value">{telemetry.align_scale ?? '–'}</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">SHIFT b</span>
              <span className="telemetry-value">{telemetry.align_shift ?? '–'}</span>
            </div>
          </div>
        </div>

        <div className="hud-card" style={{ flexGrow: 1 }}>
          <h3 className="card-title"><ConsoleIcon size={16} /> Spatial Event Terminal</h3>
          <div className="console-box">
            {logs.map((log, i) => (
              <div key={i} className={`console-line ${log.type}`}>
                <span className="console-time">[{log.time}]</span>
                <span className="text">{log.text}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>

      <main className="viewport-container">
        <div className="canvas-hud">
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <MapPin size={14} style={{ color: 'var(--accent-cyan)' }} />
            <span>X: {camera.position[0]?.toFixed(2) || '0.00'}</span>
            <span>Y: {camera.position[1]?.toFixed(2) || '0.00'}</span>
            <span>Z: {camera.position[2]?.toFixed(2) || '0.00'}</span>
          </div>
          <div style={{ borderLeft: '1px solid var(--border-muted)', height: '12px' }} />
          <div style={{ color: statusColor }}>POSE: {trackingStatus}</div>
          <div style={{ borderLeft: '1px solid var(--border-muted)', height: '12px' }} />
          <div style={{ color: 'var(--accent-green)' }} title="Reconstructed surface vertices">
            <Box size={11} style={{ display: 'inline', verticalAlign: 'middle' }} /> {denseMesh?.n_vertices?.toLocaleString() || 0} surf pts
          </div>
          {person?.present && (
            <>
              <div style={{ borderLeft: '1px solid var(--border-muted)', height: '12px' }} />
              <div style={{ color: 'var(--accent-orange)' }}>PERSON {Number(person.distance_m).toFixed(2)} m</div>
            </>
          )}
          {loopEvents.length > 0 && (
            <>
              <div style={{ borderLeft: '1px solid var(--border-muted)', height: '12px' }} />
              <div style={{ color: 'var(--accent-purple)' }}>
                <GitBranch size={11} style={{ display: 'inline', verticalAlign: 'middle' }} /> {loopEvents.length} loops
              </div>
            </>
          )}
        </div>

        {connected && (!denseMesh || (denseMesh.n_vertices || 0) < 4000) && (
          <div style={{
            position: 'absolute', top: 60, left: '50%', transform: 'translateX(-50%)',
            zIndex: 5, maxWidth: 560, textAlign: 'center', pointerEvents: 'none',
            background: 'rgba(10,14,23,0.82)', border: '1px solid rgba(0,243,255,0.35)',
            borderRadius: 8, padding: '10px 16px', color: 'var(--color-text-primary)',
            fontSize: 12, lineHeight: 1.5,
          }}>
            <strong style={{ color: 'var(--accent-cyan)' }}>Scanning the room into 3D…</strong><br />
            Move the webcam <strong>slowly sideways / forward</strong> (don't just rotate in
            place); the live surface mesh builds as you cover the room. Toggle <strong>MESH</strong> to view it, then <strong>EXPORT .PLY</strong> to save the model.
          </div>
        )}

        <div className="canvas-controls">
          <button className={`canvas-btn ${cameraMode === 'free' ? 'active' : ''}`} onClick={() => setCameraMode('free')}>FREE ORBIT</button>
          <button className={`canvas-btn ${cameraMode === 'follow' ? 'active' : ''}`} onClick={() => setCameraMode('follow')}>FOLLOW SLAM</button>
          <button className={`canvas-btn ${cameraMode === 'topdown' ? 'active' : ''}`} onClick={() => setCameraMode('topdown')}>TOP-DOWN</button>
          <div style={{ borderLeft: '1px solid var(--border-muted)', height: '18px', margin: '0 4px' }} />
          <button className={`canvas-btn ${twinMode === 'adaptive' ? 'active' : ''}`} onClick={() => setTwinMode('adaptive')} title="AABB of live point cloud">TWIN: ADAPT</button>
          <button className={`canvas-btn ${twinMode === 'fixed' ? 'active' : ''}`} onClick={() => setTwinMode('fixed')} title="Fixed 4×3×4 m reference room">TWIN: FIXED</button>
          <button className={`canvas-btn ${twinMode === 'both' ? 'active' : ''}`} onClick={() => setTwinMode('both')}>TWIN: BOTH</button>
          <button className={`canvas-btn ${twinMode === 'off' ? 'active' : ''}`} onClick={() => setTwinMode('off')}>TWIN: OFF</button>
          <div style={{ borderLeft: '1px solid var(--border-muted)', height: '18px', margin: '0 4px' }} />
          <button
            className={`canvas-btn ${meshVisible ? 'active' : ''}`}
            onClick={() => setMeshVisible((v) => !v)}
            title="Dense TSDF surface mesh"
          >
            <Box size={11} style={{ marginRight: 4 }} />
            MESH: {meshVisible ? 'ON' : 'OFF'}
            {denseMesh && denseMesh.n_vertices > 0 && (
              <span style={{
                fontSize: 9, marginLeft: 6, opacity: 0.7,
                fontFamily: 'var(--font-mono)',
              }}>
                ({denseMesh.n_vertices}v)
              </span>
            )}
          </button>
        </div>

        <ErrorBoundary label="3D Viewport failed">
          <Viewport3D
            mapData={mapData}
            currentCamera={camera}
            objects={objects}
            person={person}
            audio={audio}
            denseMesh={denseMesh}
            meshVisible={meshVisible}
            cameraMode={cameraMode}
            twinMode={twinMode}
          />
        </ErrorBoundary>
      </main>

      <aside className="sidebar-right panel">
        <div className="hud-card">
          <h3 className="card-title">
            {audio.available ? <Mic size={16} /> : <MicOff size={16} />} Audio Detection
          </h3>
          <div style={{ padding: '4px 2px' }}>
            {!audio.available ? (
              <div style={{ fontSize: 11, color: 'var(--color-text-muted)', lineHeight: 1.5 }}>
                No microphone. Enable mic access in Windows Privacy settings
                (and unmute the device), then it starts automatically.
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>STATUS</span>
                  <span style={{
                    fontSize: 12, fontWeight: 700, letterSpacing: 1,
                    color: audio.vad_active ? 'var(--accent-green)' : 'var(--color-text-muted)',
                  }}>
                    {audio.vad_active ? (audio.type === 'loud' ? 'LOUD' : 'SOUND') : 'SILENT'}
                  </span>
                </div>
                <div style={{
                  height: 14, background: 'rgba(0,0,0,0.35)',
                  border: '1px solid var(--border-muted)', borderRadius: 4, overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%', width: `${Math.round((audio.level || 0) * 100)}%`,
                    background: audio.vad_active
                      ? 'linear-gradient(90deg, var(--accent-green), var(--accent-cyan))'
                      : 'var(--accent-cyan)',
                    transition: 'width 0.08s linear', opacity: audio.vad_active ? 1 : 0.5,
                  }} />
                </div>
                <div style={{
                  display: 'flex', justifyContent: 'space-between', marginTop: 6,
                  fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--color-text-muted)',
                }}>
                  <span>RMS {Number(audio.energy || 0).toFixed(4)}</span>
                  <span>{Math.round((audio.level || 0) * 100)}%</span>
                </div>
                {audio.source === 'person' && (
                  <div style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    marginTop: 6, fontSize: 11, color: 'var(--accent-cyan)',
                  }}>
                    <span>SOURCE → PERSON</span>
                    <span style={{ fontFamily: 'var(--font-mono)' }}>
                      {audio.azimuth_deg > 0 ? 'right ' : audio.azimuth_deg < 0 ? 'left ' : ''}
                      {Math.abs(Math.round(audio.azimuth_deg))}°
                    </span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        <div className="hud-card">
          <h3 className="card-title"><Activity size={16} /> Person Range</h3>
          <div style={{ padding: '4px 2px' }}>
            {person.present ? (
              <>
                <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: 6 }}>
                  <span style={{ fontSize: 34, fontWeight: 700, color: 'var(--accent-green)', fontFamily: 'var(--font-mono)' }}>
                    {Number(person.distance_m || 0).toFixed(2)}
                  </span>
                  <span style={{ fontSize: 13, color: 'var(--color-text-secondary)' }}>m</span>
                </div>
                <div style={{
                  display: 'flex', justifyContent: 'space-between', marginTop: 8,
                  fontSize: 11, color: 'var(--color-text-secondary)',
                }}>
                  <span>BEARING</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                    {person.azimuth_deg > 0 ? 'R ' : person.azimuth_deg < 0 ? 'L ' : ''}
                    {Math.abs(Math.round(person.azimuth_deg))}°
                  </span>
                </div>
                <div style={{
                  display: 'flex', justifyContent: 'space-between', marginTop: 4,
                  fontSize: 10, color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)',
                }}>
                  <span>conf {Number(person.conf || 0).toFixed(2)}</span>
                </div>
              </>
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--color-text-muted)', fontSize: 12, padding: '10px 0' }}>
                No person in view.
              </div>
            )}
          </div>
        </div>

        <div className="hud-card">
          <h3 className="card-title"><Box size={16} /> Room Floor Plan</h3>
          <div style={{ padding: '4px 2px' }}>
            {denseMesh && denseMesh.dimensions && denseMesh.dimensions.floor_area_m2 ? (
              <>
                <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: 6 }}>
                  <span style={{ fontSize: 30, fontWeight: 700, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                    {Number(denseMesh.dimensions.floor_area_m2).toFixed(1)}
                  </span>
                  <span style={{ fontSize: 13, color: 'var(--color-text-secondary)' }}>m² scanned</span>
                </div>
                <div style={{
                  display: 'flex', justifyContent: 'space-around', marginTop: 10,
                  fontFamily: 'var(--font-mono)', textAlign: 'center',
                }}>
                  {[['W', denseMesh.dimensions.width_m], ['D', denseMesh.dimensions.depth_m], ['H', denseMesh.dimensions.height_m]].map(([k, v]) => (
                    <div key={k}>
                      <div style={{ fontSize: 16, color: 'var(--color-text-primary)' }}>{Number(v).toFixed(2)}</div>
                      <div style={{ fontSize: 9, color: 'var(--color-text-muted)' }}>{k} (m)</div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 8, fontSize: 10, color: 'var(--color-text-muted)', textAlign: 'center' }}>
                  Use <b>TOP-DOWN</b> view for the floor plan · monocular estimate
                </div>
              </>
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--color-text-muted)', fontSize: 12, padding: '10px 0' }}>
                Scan the room to estimate its size.
              </div>
            )}
          </div>
        </div>

        {objects.length > 0 && (
        <div className="hud-card">
          <h3 className="card-title"><Database size={16} /> Semantic Registry</h3>
          <div style={{
            display: 'flex', background: 'rgba(0,0,0,0.2)',
            border: '1px solid var(--border-muted)', borderRadius: '4px',
            padding: '4px 8px', alignItems: 'center', gap: '6px',
          }}>
            <Search size={12} color="var(--color-text-muted)" />
            <input
              type="text" placeholder="Filter (label or id)..."
              value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                background: 'transparent', border: 'none', color: '#fff',
                fontSize: '11px', outline: 'none', width: '100%',
              }}
            />
          </div>
          <div className="objects-list">
            {filteredObjects.length > 0 ? filteredObjects.map((obj) => (
              <div key={obj.id} className="object-row">
                <div className="object-info">
                  <span className={`object-badge ${obj.label}`}>{obj.label}</span>
                  <span className="object-name">#{obj.id}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span className="object-coords">
                    [{obj.centroid_3d[0].toFixed(1)},{obj.centroid_3d[1].toFixed(1)},{obj.centroid_3d[2].toFixed(1)}]
                  </span>
                </div>
              </div>
            )) : (
              <div style={{
                textAlign: 'center', color: 'var(--color-text-muted)',
                fontSize: '11px', padding: '12px 0',
              }}>
                No active targets.
              </div>
            )}
          </div>
        </div>
        )}

        {objects.length > 0 && (
        <div className="hud-card">
          <h3 className="card-title"><MessageSquare size={16} /> Ask the Map</h3>
          <div style={{ display: 'flex', gap: 6 }}>
            <input
              type="text"
              placeholder='e.g. "where is the chair"'
              value={aiInput}
              onChange={(e) => setAiInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') sendQuery(); }}
              style={{
                flex: 1, background: 'rgba(0,0,0,0.25)',
                border: '1px solid var(--border-muted)', borderRadius: 4,
                color: '#fff', fontSize: 11, padding: '6px 10px',
                fontFamily: 'var(--font-mono)', outline: 'none',
              }}
            />
            <button
              className="canvas-btn"
              onClick={sendQuery}
              disabled={aiBusy}
              style={{ minWidth: 50 }}
            >
              {aiBusy ? '...' : 'ASK'}
            </button>
          </div>
          {aiAnswer && (
            <div style={{
              marginTop: 8, padding: '8px 10px',
              background: 'rgba(0,243,255,0.04)',
              border: '1px solid rgba(0,243,255,0.2)',
              borderRadius: 4, fontSize: 12, lineHeight: 1.5,
              color: 'var(--color-text-primary)',
            }}>
              <div style={{ color: 'var(--accent-cyan)', fontSize: 10, marginBottom: 4 }}>
                [{aiAnswer.intent}] · conf {Math.round((aiAnswer.confidence || 0) * 100)}%
              </div>
              {aiAnswer.summary}
            </div>
          )}
        </div>
        )}

        {sceneGraph.nodes.length > 0 && (
        <div className="hud-card" style={{ flexGrow: 1 }}>
          <h3 className="card-title"><Boxes size={16} /> Scene Graph</h3>
          {sceneDescription.length === 0 ? (
            <div style={{
              textAlign: 'center', color: 'var(--color-text-muted)',
              fontSize: '11px', padding: '12px 0',
            }}>
              No relations yet.
            </div>
          ) : (
            <div className="scene-graph-list">
              {sceneDescription.slice(0, 8).map((rel, i) => (
                <div key={i} className="scene-rel">{rel}</div>
              ))}
            </div>
          )}
          <div style={{ borderTop: '1px solid var(--border-muted)', paddingTop: 8, fontSize: 10, color: 'var(--color-text-muted)' }}>
            {sceneGraph.nodes.length} nodes · {sceneGraph.edges.length} edges
          </div>
        </div>
        )}
      </aside>
    </div>
  );
};

export default App;
