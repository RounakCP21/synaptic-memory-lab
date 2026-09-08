import { useMemo, useState } from 'react';
import { ArrowDownRight, BrainCircuit, ChevronRight, CircleHelp, Database, Info, RotateCcw, Zap } from 'lucide-react';
import './index.css';

const SIZE = 8;

type PresetName = 'perfect' | 'interference' | 'decay' | 'bdh';

const presets: Record<PresetName, { label: string; sub: string; description: string; load: number; decay: number; eta: number; collinearity: number; sparse: boolean }> = {
  perfect: {
    label: 'Low key overlap',
    sub: 'near-orthogonal keys',
    description: 'Write keys with little shared direction. The matrix still contains cross-talk, but the final write is easier to recover under this toy setup.',
    load: 4, decay: 0.99, eta: 0.72, collinearity: 0.02, sparse: false,
  },
  interference: {
    label: 'High key overlap',
    sub: 'collinear keys',
    description: 'Make write keys share a direction. Each rank-1 update lands in a similar subspace, so later writes alter the response to earlier keys.',
    load: 18, decay: 0.99, eta: 0.8, collinearity: 0.92, sparse: false,
  },
  decay: {
    label: 'Short retention',
    sub: 'lower decay',
    description: 'Lower λ so older writes contribute less to the current state. This is a change to the toy update rule, not a comparison with a full attention implementation.',
    load: 12, decay: 0.75, eta: 0.68, collinearity: 0.32, sparse: false,
  },
  bdh: {
    label: 'Thresholded readout',
    sub: 'thresholded activation',
    description: 'Apply the non-negative threshold to the readout. Only values above θ remain active, making the selected output coordinates visible.',
    load: 10, decay: 0.97, eta: 0.65, collinearity: 0.18, sparse: true,
  },
};

const steps = [
  { name: 'Hebbian write', detail: 'Store a pair', title: 'Each write is rank one', copy: 'A key k and value v update the state with their outer product. The state has fixed dimensions; it does not append another key–value record.', formula: 'Mₜ = λMₜ₋₁ + η(kₜ ⊗ vₜ)' },
  { name: 'Retrieval cue', detail: 'Probe the matrix', title: 'The query reads the state', copy: 'Use a familiar key as q. The matrix produces y = Mq, then cosine similarity compares that vector with the value written at the target step.', formula: 'y = Mq  ·  cosine(y, v*)' },
  { name: 'Key overlap', detail: 'Vary the geometry', title: 'Overlap changes the address', copy: 'Outer products add together. As keys share more direction, a new write is less distinct from earlier writes and the readout can drift.', formula: 'kᵢᵀkⱼ → 1  ⇒  shared subspace ↑' },
  { name: 'Sparse readout', detail: 'Threshold the output', title: 'Thresholding selects outputs', copy: 'The optional readout applies max(0, x − θ). It changes the returned vector, while the stored matrix remains the same fixed size.', formula: 'a = max(0, Mq − θ)' },
  { name: 'Sandbox', detail: 'Change a condition', title: 'Inspect, then vary', copy: 'Adjust load, decay, plasticity, or key overlap. The matrix and metrics are recomputed locally from the same deterministic rule.', formula: 'd = 8  ⇒  synapses = d² = 64' },
];

function vectorFor(index: number, collinearity: number) {
  const base = Array.from({ length: SIZE }, (_, j) => Math.cos((j + 1) * 1.31) * 0.55 + Math.sin((j + 2) * 0.47) * 0.2);
  const direction = Array.from({ length: SIZE }, (_, j) => Math.sin((index + 1) * (j + 1) * 1.7) * 0.76 + Math.cos((index + 3) * (j + 2) * .41) * .22);
  const mixed = base.map((value, j) => value * collinearity + direction[j] * (1 - collinearity));
  const norm = Math.sqrt(mixed.reduce((sum, value) => sum + value * value, 0)) || 1;
  return mixed.map((value) => value / norm);
}

function valueFor(index: number) {
  const raw = Array.from({ length: SIZE }, (_, j) => Math.cos((index + 1) * (j + 1) * .76) * .65 + Math.sin((j + 1) * .35 + index) * .2);
  const norm = Math.sqrt(raw.reduce((sum, value) => sum + value * value, 0)) || 1;
  return raw.map((value) => value / norm);
}

function cosine(a: number[], b: number[]) {
  const dot = a.reduce((sum, value, i) => sum + value * b[i], 0);
  const na = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0)) || 1;
  const nb = Math.sqrt(b.reduce((sum, value) => sum + value * value, 0)) || 1;
  return dot / (na * nb);
}

function pretty(value: number, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

type SimulationConfig = {
  load: number;
  decay: number;
  eta: number;
  collinearity: number;
  sparse: boolean;
};

type SimulationResult = {
  matrix: number[];
  retrieved: number[];
  groundTruth: number[];
  similarity: number;
  mse: number;
  active: number;
};

function runSimulation({ load, decay, eta, collinearity, sparse }: SimulationConfig): SimulationResult {
  let matrix = Array.from({ length: SIZE * SIZE }, () => 0);
  for (let time = 0; time < load; time += 1) {
    const key = vectorFor(time, collinearity);
    const value = valueFor(time);
    matrix = matrix.map((old, index) => {
      const row = Math.floor(index / SIZE);
      const column = index % SIZE;
      return old * decay + eta * key[column] * value[row];
    });
  }
  const targetIndex = Math.max(0, load - 1);
  const query = vectorFor(targetIndex, collinearity);
  const groundTruth = valueFor(targetIndex);
  let retrieved = Array.from({ length: SIZE }, (_, row) => matrix.slice(row * SIZE, row * SIZE + SIZE).reduce((sum, value, column) => sum + value * query[column], 0));
  if (sparse) retrieved = retrieved.map((value) => Math.max(0, value - .14));
  const similarity = cosine(retrieved, groundTruth);
  const mse = retrieved.reduce((sum, value, index) => sum + (value - groundTruth[index]) ** 2, 0) / SIZE;
  return {
    matrix,
    retrieved,
    groundTruth,
    similarity: Math.max(-1, Math.min(1, similarity)),
    mse,
    active: retrieved.filter((value) => value > .02).length,
  };
}

function MiniMatrix({ heat }: { heat: number[] }) {
  const maximum = Math.max(...heat.map(Math.abs), .001);
  return (
    <div className="matrix-mini" aria-label="Animated preview of the synaptic matrix">
      {heat.map((value, index) => {
        const strength = Math.abs(value) / maximum;
        return <div className={`matrix-cell ${strength > .62 ? 'hot' : strength > .28 ? 'warm' : ''}`} key={index} data-testid={`preview-cell-${index}`} />;
      })}
    </div>
  );
}

function Matrix({ heat }: { heat: number[] }) {
  const maximum = Math.max(...heat.map(Math.abs), .001);
  return (
    <div className="matrix" aria-label="8 by 8 recurrent synaptic matrix heatmap">
      {heat.map((value, index) => {
        const strength = Math.abs(value) / maximum;
        const alpha = .08 + strength * .82;
        const positive = value >= 0;
        return (
          <div
            className="matrix-cell"
            key={index}
            title={`M[${Math.floor(index / SIZE) + 1},${index % SIZE + 1}] = ${pretty(value, 3)}`}
            data-testid={`matrix-cell-${index}`}
            style={{ background: positive ? `hsl(173 43% 58% / ${alpha})` : `hsl(9 48% 63% / ${alpha})` }}
          >
            {Math.abs(value) > maximum * .42 ? pretty(value, 1) : ''}
          </div>
        );
      })}
    </div>
  );
}

function Toggle({ on, onClick, label }: { on: boolean; onClick: () => void; label: string }) {
  return (
    <button className={`toggle ${on ? 'on' : ''}`} onClick={onClick} aria-label={label} aria-pressed={on} title={label} data-testid="toggle-sparse">
      <span />
    </button>
  );
}

const ANALYSIS_SETTINGS: SimulationConfig = {
  load: 8,
  decay: .92,
  eta: .72,
  collinearity: 0,
  sparse: false,
};

function AnalysisSurface() {
  const sweep = useMemo(() => [0, .2, .4, .6, .8, 1].map((collinearity) => ({
    collinearity,
    result: runSimulation({ ...ANALYSIS_SETTINGS, collinearity }),
  })), []);
  const chart = { left: 42, top: 16, width: 500, height: 174 };
  const xFor = (value: number) => chart.left + value * chart.width;
  const yFor = (value: number) => chart.top + ((1 - value) / 2) * chart.height;
  const points = sweep.map(({ collinearity, result }) => `${xFor(collinearity)},${yFor(result.similarity)}`).join(' ');

  return (
    <section className="section shell" id="analysis">
      <div className="section-header">
        <div><div className="eyebrow">/ experimental analysis</div><h2 className="section-title">One deterministic sweep.</h2></div>
        <p className="section-intro">A compact browser experiment derived from the same update loop. It is a toy measurement, not a benchmark or a claim about deployed models.</p>
      </div>
      <div className="analysis-shell">
        <div className="analysis-panel">
          <div className="analysis-panel-head">
            <div><div className="eyebrow">TOY-BROWSER EXPERIMENT / E-01</div><h3>Recall vs key collinearity</h3></div>
            <p>Target is always the final write. Only key collinearity changes between runs.</p>
          </div>
          <div className="analysis-chart-wrap">
            <svg className="analysis-chart" viewBox="0 0 570 220" role="img" aria-labelledby="sweep-title sweep-description">
              <title id="sweep-title">Cosine recall across six key collinearity settings</title>
              <desc id="sweep-description">A deterministic line plot computed from the local eight-dimensional matrix engine using fixed settings shown beside the chart.</desc>
              {[1, .5, 0, -.5, -1].map((value) => (
                <g key={value}>
                  <line className={value === 1 || value === -1 ? 'axis' : 'gridline'} x1={chart.left} x2={chart.left + chart.width} y1={yFor(value)} y2={yFor(value)} />
                  <text x="2" y={yFor(value) + 3}>{value.toFixed(1)}</text>
                </g>
              ))}
              <line className="axis" x1={chart.left} x2={chart.left} y1={chart.top} y2={chart.top + chart.height} />
              <line className="axis" x1={chart.left} x2={chart.left + chart.width} y1={chart.top + chart.height} y2={chart.top + chart.height} />
              <polyline className="curve" points={points} />
              {sweep.map(({ collinearity, result }) => (
                <g key={collinearity}>
                  <circle className="point" cx={xFor(collinearity)} cy={yFor(result.similarity)} r="4" tabIndex={0} aria-label={`${Math.round(collinearity * 100)} percent collinearity, cosine similarity ${pretty(result.similarity, 3)}`} data-testid={`analysis-point-${Math.round(collinearity * 100)}`} />
                  <text x={xFor(collinearity)} y={chart.top + chart.height + 18} textAnchor="middle">{Math.round(collinearity * 100)}%</text>
                </g>
              ))}
              <text x={chart.left + chart.width / 2} y="216" textAnchor="middle">key collinearity</text>
              <text x="10" y={chart.top + chart.height / 2} textAnchor="middle" transform={`rotate(-90 10 ${chart.top + chart.height / 2})`}>cosine recall</text>
            </svg>
            <div className="chart-caption"><span>higher is closer to target value</span><span>6 fixed sweep points</span></div>
            <table className="analysis-table">
              <caption>Observed cosine similarity at each fixed key-collinearity setting</caption>
              <thead><tr><th scope="col">key overlap</th><th scope="col">cosine recall</th></tr></thead>
              <tbody>
                {sweep.map(({ collinearity, result }) => (
                  <tr key={collinearity}><td>{Math.round(collinearity * 100)}%</td><td>{pretty(result.similarity, 3)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <aside className="analysis-settings" aria-label="Fixed settings for toy-browser experiment">
          <div className="eyebrow">FIXED SETTINGS</div>
          <h3>Reproducibility note</h3>
          <dl className="settings-list">
            <div className="setting-row"><dt>dimension d</dt><dd>8</dd></div>
            <div className="setting-row"><dt>writes</dt><dd>{ANALYSIS_SETTINGS.load}</dd></div>
            <div className="setting-row"><dt>decay λ</dt><dd>{ANALYSIS_SETTINGS.decay.toFixed(2)}</dd></div>
            <div className="setting-row"><dt>plasticity η</dt><dd>{ANALYSIS_SETTINGS.eta.toFixed(2)}</dd></div>
            <div className="setting-row"><dt>readout</dt><dd>dense</dd></div>
            <div className="setting-row"><dt>query</dt><dd>final write</dd></div>
            <div className="setting-row"><dt>threshold θ</dt><dd>not applied</dd></div>
          </dl>
          <p className="analysis-note">The sweep is deterministic: no random source, network request, or stored result is used. Values update only if this engine changes.</p>
        </aside>
      </div>
    </section>
  );
}

function App() {
  const [activeStep, setActiveStep] = useState(0);
  const [preset, setPreset] = useState<PresetName>('perfect');
  const [load, setLoad] = useState(presets.perfect.load);
  const [decay, setDecay] = useState(presets.perfect.decay);
  const [eta, setEta] = useState(presets.perfect.eta);
  const [collinearity, setCollinearity] = useState(presets.perfect.collinearity);
  const [sparse, setSparse] = useState(presets.perfect.sparse);

  const simulation = useMemo(() => runSimulation({ load, decay, eta, collinearity, sparse }), [load, decay, eta, collinearity, sparse]);

  const applyPreset = (name: PresetName) => {
    const next = presets[name];
    setPreset(name);
    setLoad(next.load);
    setDecay(next.decay);
    setEta(next.eta);
    setCollinearity(next.collinearity);
    setSparse(next.sparse);
  };
  const reset = () => applyPreset('perfect');
  const nextStep = () => setActiveStep((step) => Math.min(4, step + 1));
  const current = steps[activeStep];
  const similarityTone = simulation.similarity > .85 ? 'good' : simulation.similarity > .55 ? 'warn' : 'bad';

  return (
    <main className="lab-app">
      <header className="shell header">
        <a className="brand" href="#top" data-testid="link-top" aria-label="Synaptic Memory Lab home">
          <span className="brand-mark"><BrainCircuit size={17} strokeWidth={1.6} /></span>
          <span><span className="brand-name">SYNAPTIC MEMORY LAB</span><span className="brand-meta">DETERMINISTIC BROWSER STUDY</span></span>
        </a>
        <nav className="header-nav" aria-label="Page sections">
          <a href="#walkthrough" data-testid="link-walkthrough">Walkthrough</a>
          <a href="#sandbox" data-testid="link-sandbox">Sandbox</a>
          <a href="#analysis" data-testid="link-analysis">Analysis</a>
          <a href="#evidence" data-testid="link-evidence">Limits</a>
          <span className="status-dot">LOCAL COMPUTATION</span>
        </nav>
      </header>

      <section className="shell hero" id="top">
        <div className="hero-grid">
          <div className="reveal">
            <div className="eyebrow">/ browser study · associative memory</div>
            <h1>Change the condition.<br /><em>Inspect the recall.</em></h1>
            <p className="hero-copy">A small, deterministic model of outer-product memory. It keeps an 8 × 8 state while key overlap, load, decay, and readout change the retrieved value.</p>
            <div className="hero-note"><Info size={15} /><span>MODEL SCOPE</span> This is a toy browser experiment. It shows one update rule and its failure modes; it does not establish a benchmark result.</div>
            <div className="hero-actions">
              <a className="button button-primary" href="#walkthrough" data-testid="button-start">Read the protocol <ArrowDownRight size={15} /></a>
              <a className="button button-ghost" href="#sandbox" data-testid="button-sandbox">Adjust conditions <Zap size={14} /></a>
            </div>
          </div>
          <div className="apparatus reveal reveal-delay" aria-label="Live matrix apparatus">
            <div className="apparatus-head"><strong>RECURRENT STATE / Mₜ</strong><span>8 × 8 / 64 COEFFICIENTS</span></div>
            <MiniMatrix heat={simulation.matrix} />
            <div className="apparatus-signal">
              <svg viewBox="0 0 500 40" role="img" aria-label="Synaptic signal trace"><path d="M0 24 C18 24 18 24 35 22 S50 4 67 20 S86 42 104 21 S121 12 136 22 S154 29 171 20 S193 1 210 22 S229 33 248 23 S266 15 283 22 S299 28 316 19 S332 9 350 21 S370 36 385 21 S407 6 421 20 S444 29 462 21 S484 18 500 19" /></svg>
            </div>
            <div className="apparatus-foot"><span>λ = {decay.toFixed(2)} &nbsp; η = {eta.toFixed(2)}</span><span className="status-dot">RECOMPUTED</span></div>
          </div>
        </div>
      </section>

      <section className="section shell" id="walkthrough">
        <div className="section-header">
          <div><div className="eyebrow">/ guided protocol · 05 steps</div><h2 className="section-title">From write to failure.</h2></div>
            <p className="section-intro">Read the update rule in order, then vary it. Every cell and metric is computed in this browser; there is no pre-rendered result.</p>
        </div>
        <div className="walkthrough">
          <div className="step-list" role="tablist" aria-label="Walkthrough steps">
            {steps.map((step, index) => (
              <button className={`step-button ${activeStep === index ? 'active' : ''}`} key={step.name} onClick={() => setActiveStep(index)} role="tab" aria-selected={activeStep === index} data-testid={`step-${index + 1}`}>
                <span className="step-number">0{index + 1}</span><span className="step-name">{step.name}</span><span className="step-detail">{step.detail}</span>
              </button>
            ))}
          </div>
          <div className="step-stage" role="tabpanel">
            <div className="stage-copy"><h3>{current.title}</h3><p>{current.copy}</p><div className="formula">{current.formula}</div></div>
            <SimulationCard simulation={simulation} load={load} decay={decay} eta={eta} collinearity={collinearity} sparse={sparse} setLoad={setLoad} setDecay={setDecay} setEta={setEta} setCollinearity={setCollinearity} setSparse={setSparse} />
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 14 }}>
              <button className="button button-ghost button-small" onClick={nextStep} disabled={activeStep === 4} data-testid="button-next-step">Next instrument step <ChevronRight size={14} /></button>
            </div>
          </div>
        </div>
      </section>

      <section className="section shell" id="sandbox">
        <div className="section-header">
          <div><div className="eyebrow">/ controlled conditions</div><h2 className="section-title">Compare conditions.</h2></div>
          <p className="section-intro">Each preset changes the same four controls. Use the values as starting points, not conclusions.</p>
        </div>
        <div className="preset-layout">
          <div className="preset-list" role="list" aria-label="Simulation presets">
            {(Object.keys(presets) as PresetName[]).map((name) => (
              <button className={`preset ${preset === name ? 'active' : ''}`} key={name} onClick={() => applyPreset(name)} data-testid={`preset-${name}`}>
                <span><strong>{presets[name].label}</strong><small>{presets[name].sub}</small></span><ChevronRight size={14} />
              </button>
            ))}
          </div>
          <div className="preset-description">
             <div className="eyebrow">SELECTED CONDITION / {preset.toUpperCase()}</div>
            <h3>{presets[preset].label}</h3><p>{presets[preset].description}</p>
            <div className="preset-readout">
              <div className="readout"><span>cosine recall</span><strong className={similarityTone}>{pretty(simulation.similarity * 100, 1)}%</strong></div>
              <div className="readout"><span>matrix state</span><strong>{load} writes</strong></div>
              <div className="readout"><span>active path</span><strong>{simulation.active} / 8 neurons</strong></div>
            </div>
             <button className="button button-ghost button-small" onClick={reset} style={{ marginTop: 21 }} data-testid="button-reset"><RotateCcw size={13} /> Reset to low overlap</button>
          </div>
        </div>
      </section>

      <AnalysisSurface />

      <section className="section shell" id="context">
        <div className="concept-grid">
          <div className="concept-copy">
            <div className="eyebrow">/ interpretation</div>
            <h2>State size is not recall.</h2>
            <p><strong>Standard attention</strong> keeps keys and values individually addressable and compares a query with the context. Its stored context grows with sequence length.</p>
            <p><strong>Fast weights</strong> replace that list with a running matrix. The state stays O(d²) in this toy rule, while retrieval depends on the geometry of the keys and the order of writes.</p>
            <p><strong>Hebbian language</strong> is an analogy for the update, not a biological result. This lab uses an 8 × 8 matrix with fixed vectors; cortex has dynamics and structure that are not represented here.</p>
            <div className="annotation"><CircleHelp size={14} style={{ verticalAlign: 'middle', marginRight: 7, color: 'hsl(var(--accent))' }} /> <strong>About the sparse switch:</strong> it thresholds the returned vector after matrix multiplication. It does not make the stored matrix sparse.</div>
          </div>
          <div className="architecture" aria-label="Architecture comparison">
             <div className="architecture-card"><header><h3>Standard attention</h3><span>REFERENCE</span></header><p>Keys and values remain individually addressable. <strong>Memory: O(Td)</strong>. Retrieval compares against the context.</p></div>
             <div className="architecture-card featured"><header><h3>Fast weights / outer product</h3><span>THIS TOY</span></header><p>Write k ⊗ v into a recurrent state. <strong>State: O(d²)</strong>. Similar keys superpose.</p></div>
             <div className="architecture-card"><header><h3>Thresholded readout</h3><span>OPTIONAL STEP</span></header><p>Apply max(0, x − θ) after the matrix response. <strong>State remains O(d²)</strong>; output support can shrink.</p></div>
             <div className="architecture-card"><header><h3>Biological analogy</h3><span>LIMITED BRIDGE</span></header><p>Co-activity is a useful intuition, but this model omits the dynamics, constraints, and anatomy of biological learning.</p></div>
          </div>
        </div>
      </section>

      <section className="section shell" id="evidence">
        <div className="evidence">
          <div><div className="eyebrow">/ limits of inference</div><h2>Separate result from claim.</h2><p>This page measures one deterministic toy mechanism. It does not provide a benchmark, an independent reproduction, or evidence about deployment or biology.</p></div>
          <div className="evidence-list">
            <div className="evidence-item"><header><strong>Measured here</strong><span className="tag">local only</span></header><p>The matrix, retrieved vector, cosine similarity, mean squared error, and active count are recomputed from the displayed controls.</p></div>
            <div className="evidence-item"><header><strong>Not reproduced here</strong><span className="tag">outside scope</span></header><p>Results from a paper, model family, or separate implementation require the same task, data, and evaluation protocol before comparison.</p></div>
            <div className="evidence-item"><header><strong>Not implied</strong><span className="tag">do not infer</span></header><p>An interactive demo is not evidence of production reliability, user impact, or biological validity.</p></div>
          </div>
        </div>
      </section>

      <footer className="shell footer"><div className="footer-inner"><p><span style={{ color: 'hsl(var(--primary))' }}>SYNAPTIC MEMORY LAB</span> · BROWSER STUDY</p><p>Deterministic state · no API · no hidden cache</p></div></footer>
    </main>
  );
}

type SimulationProps = {
  simulation: { matrix: number[]; retrieved: number[]; groundTruth: number[]; similarity: number; mse: number; active: number };
  load: number; decay: number; eta: number; collinearity: number; sparse: boolean;
  setLoad: (value: number) => void; setDecay: (value: number) => void; setEta: (value: number) => void; setCollinearity: (value: number) => void; setSparse: (value: boolean) => void;
};

function SimulationCard({ simulation, load, decay, eta, collinearity, sparse, setLoad, setDecay, setEta, setCollinearity, setSparse }: SimulationProps) {
  const tone = simulation.similarity > .85 ? 'good' : simulation.similarity > .55 ? 'warn' : 'bad';
  return (
    <div className="sim-card" data-testid="simulation-card">
       <div className="sim-topbar"><div className="sim-topbar-title"><span /> SYNAPTIC REGISTER</div><div className="sim-topbar-meta">{sparse ? 'THRESHOLDED READOUT / θ = 0.14' : 'DENSE READOUT'} · d = 8</div></div>
      <div className="sim-body">
        <div className="matrix-panel"><div className="matrix-label"><span>MATRIX HEATMAP / Mₜ</span><span>{load} writes</span></div><Matrix heat={simulation.matrix} /><div className="matrix-label" style={{ marginTop: 14, marginBottom: 0 }}><span>TEAL = POSITIVE / CORAL = NEGATIVE</span><span>HOVER CELLS FOR VALUE</span></div></div>
        <div className="sim-side">
          <div className="metric"><div className="metric-label">cosine similarity</div><div className={`metric-value ${tone}`} data-testid="metric-cosine">{pretty(simulation.similarity, 3)}</div></div>
          <div className="metric"><div className="metric-label">mean squared error</div><div className="metric-value" data-testid="metric-mse">{pretty(simulation.mse, 3)}</div></div>
          <div className="metric"><div className="metric-label">retrieved / ground truth</div><div className="metric-value" style={{ fontSize: 16 }} data-testid="metric-vector">{simulation.retrieved.slice(0, 3).map((value) => pretty(value, 2)).join('  ')} / {simulation.groundTruth.slice(0, 3).map((value) => pretty(value, 2)).join('  ')}</div></div>
        </div>
      </div>
      <div className="sim-controls">
        <div className="control"><label htmlFor="load">association load <output>{load}</output></label><input id="load" data-testid="input-load" type="range" min="2" max="20" step="1" value={load} onChange={(event) => setLoad(Number(event.target.value))} title="Number of key-value writes" /></div>
        <div className="control"><label htmlFor="decay">decay λ <output>{decay.toFixed(2)}</output></label><input id="decay" data-testid="input-decay" type="range" min="0" max="1" step=".01" value={decay} onChange={(event) => setDecay(Number(event.target.value))} title="Retention from one write to the next" /></div>
        <div className="control"><label htmlFor="plasticity">plasticity η <output>{eta.toFixed(2)}</output></label><input id="plasticity" data-testid="input-plasticity" type="range" min="0" max="1" step=".01" value={eta} onChange={(event) => setEta(Number(event.target.value))} title="Strength of each outer-product write" /></div>
        <div className="control"><label htmlFor="collinearity">key collinearity <output>{Math.round(collinearity * 100)}%</output></label><input id="collinearity" data-testid="input-collinearity" type="range" min="0" max="1" step=".01" value={collinearity} onChange={(event) => setCollinearity(Number(event.target.value))} title="How much keys share a common direction" /></div>
        <div className="toggle-row"><Toggle on={sparse} onClick={() => setSparse(!sparse)} label="Toggle BDH sparse non-negative activation" /><span>BDH sparse activation</span></div>
        <div className="toggle-row" title="The matrix always remains d squared synapses"><Database size={14} /><span>fixed state: 64 synapses</span></div>
      </div>
    </div>
  );
}

export default App;