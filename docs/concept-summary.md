# Synaptic Plasticity as Short-Term Memory

## One-page concept summary

### Problem and design pressure

Autoregressive Transformers preserve exact access to prior context by retaining key/value pairs for previously generated tokens. For a head width \(d\) and sequence length \(T\), the decoder-side cache grows as \(O(Td)\) per layer and head family. Each additional token adds persistent state that must be addressed, moved, and eventually evicted. The design pressure is to retain useful associations without retaining every historical record.

### Mechanism

Linear attention exposes one route by regrouping the query–key–value computation into a recurrent fast-weight state. Given a key \(k_t\), target/value \(v_t\), decay \(\lambda\), and plasticity \(\eta\), the state update is

\[
M_t = \lambda M_{t-1} + \eta(k_t \otimes v_t),
\qquad
y_t = M_tq_t.
\]

The outer product writes a rank-one association into a matrix-shaped state; the query retrieves a superposition of written values. With fixed width, \(M\) occupies \(O(d^2)\) state regardless of the number of presented items. This is the same algebraic shape as a Hebbian rule, \(\Delta W_{ij}=\eta x_i y_j\): co-active pre- and post-synaptic patterns strengthen their connection. Attention’s content-addressed read and its memory write become two views of one recurrent substrate.

The fixed-state guarantee changes the failure mode. A KV cache preserves records until its hardware budget is exhausted; a fixed matrix must superpose them. If \(q_i\) is not orthogonal to an unrelated \(k_j\), the read contains a cross-term proportional to \(q_i^\top k_j v_j\). Increasing load or key collinearity produces associative drift. Decay adds a second trade-off: \(\lambda<1\) limits stale traces but also attenuates legitimate long-horizon memories.

### Role of Dragon Hatchling

The Dragon Hatchling (BDH) proposal places memory and reasoning in a biologically inspired network of locally interacting neuron particles. Its GPU-friendly formulation connects sparse non-negative activity, low-rank transformations, and linear-attention-compatible recurrence. The relevant conceptual point is that synapses can be treated as an active working-memory substrate instead of a passive parameter store. A sparse firing rule such as

\[
\phi(x)=\max(0,x-\theta)
\]

can reduce weak, sign-canceling activity before a write or read. At roughly five percent active units, sparse firing may improve local-write signal-to-noise, but it does not remove finite-capacity interference. The lab compares dense and thresholded retrieval under identical load and collision settings. BDH-GPU should not be collapsed into a Mamba-style selective state-space model: both support recurrent inference, but their state semantics differ.

### Architectural comparison

| Architecture | State footprint | Inference per-token FLOPs | Recurrent update | Working-memory limit | GPU efficiency |
|---|---:|---:|---|---|---|
| Standard Transformer / softmax KV cache | \(O(Td)\) cache | \(O(Td)\) attention read | Append K/V records | Context and hardware budget | Excellent optimized kernels; traffic grows with \(T\) |
| Mamba / SSM | \(O(d)\) to \(O(dN)\), model-dependent | Linear in state width | Input-dependent state transition and convolution/scan | Compression and state dimension | Strong scan kernels; hardware depends on implementation |
| Linear attention / fast weights | \(O(d^2)\) matrix state | \(O(d^2)\) read/write | Decay plus outer-product accumulation | Rank, key overlap, and matrix capacity | Parallelizable; matrix state can be bandwidth-sensitive |
| BDH / BDH-CQ | Architecture-dependent sparse synaptic state | Depends on sparse graph and low-rank path | Neuron firing with synaptic updates | Finite synaptic state and interference | GPU-friendly sparse/low-rank formulation reported by developers |

The table is a mechanism-level comparison, not a claim that the systems are interchangeable or benchmark-equivalent.

### Evidence discipline

The BDH preprint reports developer-run scaling and task experiments; those results remain developer-reported unless independently reproduced. A reproduction matches the stated protocol and metrics in a separate implementation. A production deployment establishes operational use, not benchmark validity. This project claims none of those outcomes: its cosine similarity and MSE come from a deterministic browser toy engine. The page includes one descriptive six-point sweep over key collinearity with fixed settings, but it does not estimate uncertainty or establish a general capacity law.

### Limitations and open problems

The fixed matrix is a finite-rank information bottleneck. Orthogonal patterns coexist only within the effective dimension and noise margin; non-orthogonal keys create cross-talk earlier. Thresholding may suppress weak noise while deleting low-amplitude information, and decay trades persistence for freshness. The abstraction omits normalization, learned projections, optimization dynamics, hardware tiling, and the full BDH graph. A complete system must explain how fast synaptic state is consolidated into slow weights without freezing errors, and whether sparse activity improves end-to-end quality rather than only toy retrieval.

### Next steps and primary literature

The next experiment should sweep key coherence and load independently, report confidence intervals over randomized seeds, and compare dense, thresholded, and normalized reads. Primary references are Kosowski et al., **“The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain,”** arXiv:2509.26507 (2025); Sun et al., **“Retentive Network: A Successor to Transformer for Large Language Models,”** arXiv:2307.08621 (2023); Yang et al., **“Gated Linear Attention Transformers with Hardware-Efficient Training,”** arXiv:2312.06635 (2023/2024); and Tyulmankov et al., **“Meta-learning synaptic plasticity and memory addressing for continual familiarity detection,”** *Neuron* (2022). Miconi’s work on Hebbian learning with gradients provides an additional implementation reference.
