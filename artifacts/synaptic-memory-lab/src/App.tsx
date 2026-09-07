import { useMemo, useState } from 'react';
import { ArrowDownRight, BrainCircuit, ChevronRight, CircleHelp, Database, Info, RotateCcw, Zap } from 'lucide-react';
import './index.css';

const SIZE = 8;

type PresetName = 'perfect' | 'interference' | 'decay' | 'bdh';

const presets: Record<PresetName, { label: string; sub: string; description: string; load: number; decay: number; eta: number; collinearity: number; sparse: boolean }> = {
  perfect: {
    label: 'Perfect Recall',
    sub: 'orthogonal keys',
    description: 'Keep the write keys far apart. The outer-product matrix behaves like a tidy associative index: query one key and its value comes back with almost no cross-talk.',
    load: 4, decay: 0.99, eta: 0.72, collinearity: 0.02, sparse: false,
  },
  interference: {
    label: 'Catastrophic Interference',
    sub: 'collinear keys',
    description: 'Now make keys share a direction. Every new rank-1 write lands on the same synapses, so a late memory edits the retrieval landscape of earlier memories.',
    load: 18, decay: 0.99, eta: 0.8, collinearity: 0.92, sparse: false,
  },
  decay: {
    label: 'Hebbian Decay vs KV-Cache',
    sub: 'forgetting over time',
    description: 'A decaying matrix is a bounded dynamical system, not an infinite log. Lower λ and the old traces fade. A KV-cache would retain each token exactly, but pay linear memory.',
    load: 12, decay: 0.75, eta: 0.68, collinearity: 0.32, sparse: false,
  },
  bdh: {
    label: 'BDH Sparse Synaptic Regime',
    sub: 'thresholded activation',
    description: 'Turn on a sparse, non-negative activation readout. BDH reframes the system as neurons and synapses: only activations above θ propagate, making a small active circuit visible.',
    load: 10, decay: 0.97, eta: 0.65, collinearity: 0.18, sparse: true,
  },
};

const steps = [
  { name: 'Hebbian write', detail: 'Store a pair', title: 'A memory is a rank-1 write', copy: 'A key k and value v meet at the synapse. Their outer product writes every pairwise connection in one pass; no dictionary grows behind the scene.', formula: 'Mₜ = λMₜ₋₁ + η(kₜ ⊗ vₜ)' },
  { name: 'Retrieval cue', detail: 'Probe the matrix', title: 'Query the apparatus', copy: 'Present a familiar key as q. The recurrent matrix multiplies it, y = Mq, and the resulting vector is compared against the value that was originally written.', formula: 'y = Mq  ·  cosine(y, v*)' },
  { name: 'Capacity wall', detail: 'Break the claim', title: 'The wall is geometric', copy: 'Outer products superpose. When keys point in similar directions, the new write is not a new address; it is a rewrite of the same synaptic subspace. Watch similarity fall.', formula: 'kᵢᵀkⱼ → 1  ⇒  interference ↑' },
  { name: 'BDH sparse regime', detail: 'Threshold the readout', title: 'Fewer active synapses', copy: 'A sparse non-negative readout applies max(0, x − θ). The matrix remains fixed-size, while the active path through it becomes selective and inspectable.', formula: 'a = max(0, Mq − θ)' },
  { name: 'Sandbox', detail: 'Choose the failure', title: 'Your turn at the bench', copy: 'Use the dials below to construct a claim, then break it. Increase load, align the keys, or shorten the decay constant. The apparatus responds immediately.', formula: 'd = 8  ⇒  synapses = d² = 64' },
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
            style={{ background: positive ? `hsl(164 88% 61% / ${alpha})` : `hsl(5 76% 65% / ${alpha})` }}
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

function App() {
  const [activeStep, setActiveStep] = useState(0);
  const [preset, setPreset] = useState<PresetName>('perfect');
  const [load, setLoad] = useState(presets.perfect.load);
  const [decay, setDecay] = useState(presets.perfect.decay);
  const [eta, setEta] = useState(presets.perfect.eta);
  const [collinearity, setCollinearity] = useState(presets.perfect.collinearity);
  const [sparse, setSparse] = useState(presets.perfect.sparse);

  const simulation = useMemo(() => {
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
    const sim = cosine(retrieved, groundTruth);
    const mse = retrieved.reduce((sum, value, index) => sum + (value - groundTruth[index]) ** 2, 0) / SIZE;
    return {
      matrix,
      retrieved,
      groundTruth,
      similarity: Math.max(-1, Math.min(1, sim)),
      mse,
      active: retrieved.filter((value) => value > .02).length,
    };
  }, [load, decay, eta, collinearity, sparse]);

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
          <span><span className="brand-name">SYNAPTIC MEMORY LAB</span><span className="brand-meta">DATAFORGE 2026 / PATHWAY TRACK</span></span>
        </a>
        <nav className="header-nav" aria-label="Page sections">
          <a href="#walkthrough" data-testid="link-walkthrough">Walkthrough</a>
          <a href="#sandbox" data-testid="link-sandbox">Sandbox</a>
          <a href="#evidence" data-testid="link-evidence">Evidence</a>
          <span className="status-dot">LIVE ENGINE</span>
        </nav>
      </header>

      <section className="shell hero" id="top">
        <div className="hero-grid">
          <div className="reveal">
            <div className="eyebrow">/ instrument 01 · associative memory</div>
            <h1>Turn the dial.<br /><em>Break the memory.</em></h1>
            <p className="hero-copy">A visual essay and working apparatus for one sharp claim: Hebbian outer-product writes buy a fixed O(d²) synaptic matrix — until non-orthogonal keys make the memories collide.</p>
            <div className="hero-note"><Info size={15} /><span>HYPOTHESIS</span> Short-term associative memory can emerge without a growing cache. The failure mode is not hidden. It is geometric.</div>
            <div className="hero-actions">
              <a className="button button-primary" href="#walkthrough" data-testid="button-start">Enter the walkthrough <ArrowDownRight size={15} /></a>
              <a className="button button-ghost" href="#sandbox" data-testid="button-sandbox">Open sandbox <Zap size={14} /></a>
            </div>
          </div>
          <div className="apparatus reveal reveal-delay" aria-label="Live matrix apparatus">
            <div className="apparatus-head"><strong>RECURRENT CORE / Mₜ</strong><span>8 × 8 / 64 SYNAPSES</span></div>
            <MiniMatrix heat={simulation.matrix} />
            <div className="apparatus-signal">
              <svg viewBox="0 0 500 40" role="img" aria-label="Synaptic signal trace"><path d="M0 24 C18 24 18 24 35 22 S50 4 67 20 S86 42 104 21 S121 12 136 22 S154 29 171 20 S193 1 210 22 S229 33 248 23 S266 15 283 22 S299 28 316 19 S332 9 350 21 S370 36 385 21 S407 6 421 20 S444 29 462 21 S484 18 500 19" /></svg>
            </div>
            <div className="apparatus-foot"><span>λ = {decay.toFixed(2)} &nbsp; η = {eta.toFixed(2)}</span><span className="status-dot">WRITING</span></div>
          </div>
        </div>
      </section>

      <section className="section shell" id="walkthrough">
        <div className="section-header">
          <div><div className="eyebrow">/ guided protocol · 05 steps</div><h2 className="section-title">From write to failure.</h2></div>
          <p className="section-intro">Read the claim in order, then use the same engine to disagree with it. There is no pre-rendered animation here — every cell is computed in your browser.</p>
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
          <div><div className="eyebrow">/ controlled experiments</div><h2 className="section-title">Preset the argument.</h2></div>
          <p className="section-intro">Each preset changes the same four physical knobs. Use them as starting conditions, not conclusions.</p>
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
            <div className="eyebrow">ACTIVE CONDITION / {preset.toUpperCase()}</div>
            <h3>{presets[preset].label}</h3><p>{presets[preset].description}</p>
            <div className="preset-readout">
              <div className="readout"><span>cosine recall</span><strong className={similarityTone}>{pretty(simulation.similarity * 100, 1)}%</strong></div>
              <div className="readout"><span>matrix state</span><strong>{load} writes</strong></div>
              <div className="readout"><span>active path</span><strong>{simulation.active} / 8 neurons</strong></div>
            </div>
            <button className="button button-ghost button-small" onClick={reset} style={{ marginTop: 21 }} data-testid="button-reset"><RotateCcw size={13} /> Reset to orthogonal</button>
          </div>
        </div>
      </section>

      <section className="section shell" id="context">
        <div className="concept-grid">
          <div className="concept-copy">
            <div className="eyebrow">/ reading the apparatus</div>
            <h2>One matrix.<br />Four ways to remember.</h2>
            <p><strong>Standard attention</strong> stores a growing set of key–value interactions and retrieves by comparing a query with every key. Its expressive context is explicit, but its memory footprint grows with sequence length.</p>
            <p><strong>Linear attention and fast weights</strong> move the interaction into a running summary. The outer-product update is a fast-weight rule: constant state in sequence length, with a retrieval error term that depends on the geometry of keys.</p>
            <p><strong>Biological Hebbian learning</strong> gives the update its intuition: neurons that activate together strengthen their connection. This lab uses the cleanest toy form of that idea, not a claim that cortex is an 8 × 8 matrix.</p>
            <div className="annotation"><CircleHelp size={14} style={{ verticalAlign: 'middle', marginRight: 7, color: 'hsl(var(--accent))' }} /> <strong>BDH’s neuron–synapse framing:</strong> treat activations as neurons and M as synapses. Sparse non-negative thresholding makes the active circuit a first-class object instead of hiding it in a dense vector.</div>
          </div>
          <div className="architecture" aria-label="Architecture comparison">
            <div className="architecture-card"><header><h3>Standard attention</h3><span>EXPLICIT CACHE</span></header><p>Keys and values remain individually addressable. <strong>Memory: O(Td)</strong>. Retrieval compares against the whole context.</p></div>
            <div className="architecture-card featured"><header><h3>Fast weights / Hebbian</h3><span>THIS LAB</span></header><p>Write k ⊗ v into a recurrent state. <strong>State: O(d²)</strong>. Similar keys superpose and interfere.</p></div>
            <div className="architecture-card"><header><h3>BDH-inspired sparse readout</h3><span>ACTIVE CIRCUIT</span></header><p>Threshold the matrix response with max(0, x − θ). <strong>State stays O(d²)</strong>, active support can shrink.</p></div>
            <div className="architecture-card"><header><h3>Biological analogy</h3><span>CAREFUL BRIDGE</span></header><p>Co-activity strengthens synapses, but real learning has dynamics, constraints, and anatomy absent from this minimal model.</p></div>
          </div>
        </div>
      </section>

      <section className="section shell" id="evidence">
        <div className="evidence">
          <div><div className="eyebrow">/ evidence discipline</div><h2>Make the boundary visible.</h2><p>This instrument teaches a mechanism. It does not turn a toy simulation into a benchmark result. Track which statements are measured here, independently reproduced, or deployed at scale.</p></div>
          <div className="evidence-list">
            <div className="evidence-item"><header><strong>Developer-reported result</strong><span className="tag">label it</span></header><p>Claims about BDH or a particular model family should be attributed to the developers and tied to a named report, release, or technical note.</p></div>
            <div className="evidence-item"><header><strong>External reproduction</strong><span className="tag">separate it</span></header><p>A result reproduced by an independent implementation is stronger evidence, but only under the same task, data, and evaluation protocol.</p></div>
            <div className="evidence-item"><header><strong>Deployment evidence</strong><span className="tag">do not imply</span></header><p>A working interactive demo is not evidence of production reliability, user impact, or biological validity. Those require operational measurements.</p></div>
          </div>
        </div>
      </section>

      <footer className="shell footer"><div className="footer-inner"><p><span style={{ color: 'hsl(var(--primary))' }}>SYNAPTIC MEMORY LAB</span> · DATAFORGE 2026 PATHWAY TRACK</p><p>Fixed state · live browser engine · no API · no hidden cache</p></div></footer>
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
      <div className="sim-topbar"><div className="sim-topbar-title"><span /> LIVE SYNAPTIC REGISTER</div><div className="sim-topbar-meta">{sparse ? 'SPARSE READOUT / θ = 0.14' : 'DENSE READOUT'} · d = 8</div></div>
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