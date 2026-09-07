# Synaptic Memory Lab

## A falsifiable visual essay on Hebbian short-term memory

Synaptic Memory Lab is an interactive, browser-native research artifact for the DataForge 2026 Pathway Track. It lets learners write cue–target associations into a recurrent synaptic matrix, probe that matrix, and deliberately push it into interference. The central mechanism is deliberately small enough to inspect:

```text
M_t = λ M_(t−1) + η (k_t ⊗ v_t)
y_t = M_t q_t
```

The app treats the equation as an instrument rather than a metaphor. The matrix is updated live in JavaScript, its cells are rendered as a heatmap, and each retrieval is scored against the known target.

## One-sentence falsifiable claim

Associating context with outer-product Hebbian writes into a recurrent matrix can recall cue–target pairs with fixed `O(d²)` state instead of an ever-growing KV cache, but recall must degrade when keys collide non-orthogonally or when the load exceeds the matrix’s effective rank.

## What to test

1. Start with **Perfect Recall** and inspect the bright, localized writes.
2. Raise **Key interference** or **Memory load** until the retrieved vector blurs.
3. Set **Decay** below 1.0 and watch older traces attenuate.
4. Enable **BDH sparse mode** to apply a non-negative threshold to keys, values, and queries.
5. Compare the live estimate `y = Mq` with the ground-truth target and read both cosine similarity and mean squared error.

The artifact is educational and falsifiable. It is not a reproduction of a language-model benchmark, a claim that biological synapses implement this exact matrix, or a substitute for the BDH paper’s full architecture.

## Audience and prerequisites

This is for ML researchers, engineers, educators, and advanced students who know:

- vectors, dot products, outer products, and matrices;
- the basic query/key/value view of attention;
- the difference between a recurrent state and a token-by-token cache.

No GPU, server, database, or API key is required.

## Learning objectives

After using the artifact, a learner should be able to:

- **Explain** why a softmax attention decoder normally retains one key/value record per prior token.
- **Construct** a Hebbian outer-product write and identify the matrix entries it changes.
- **Calculate** how a query retrieves a value through `Mq`.
- **Compare** cosine similarity and MSE as retrieval diagnostics.
- **Analyze** the role of decay, plasticity, key overlap, load, and sparse activation.
- **Evaluate** the trade-off between fixed recurrent state and interference.
- **Design** a follow-up experiment that separates decay failure from collision failure.

## System architecture

```text
┌─────────────────────────────── Browser ───────────────────────────────┐
│                                                                       │
│  Controls ──> deterministic vector synthesizer ──> Hebbian engine     │
│                                  │                    │               │
│                                  │                    ├─ M heatmap     │
│                                  │                    ├─ y = Mq       │
│                                  │                    └─ metrics       │
│                                  │                                    │
│  Walkthrough + presets ──────────┴──────> React state / derived view  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### What runs live

- Association vectors are synthesized deterministically from the current controls.
- The engine applies the recurrence `M ← λM + η k vᵀ` for each item.
- Optional BDH mode applies `ReLU(x − θ)` to the synthesized activations.
- The selected cue is retrieved with a matrix–vector multiply.
- Cosine similarity and mean squared error are computed against the selected target.
- The heatmap is derived from the current matrix; there is no pre-rendered animation.

### What is not precomputed

There is no remote model, hidden inference service, recorded trace, or server-side result. Presets only set parameter values. The browser recomputes the matrix and metrics after every control change.

## Technical anatomy

### Standard attention

Softmax attention computes:

```text
softmax(QKᵀ / √d)V
```

During autoregressive decoding, prior keys and values are normally retained as a KV cache. For sequence length `T` and head width `d`, the cache grows as `O(Td)` per layer/head family. This is excellent for exact content-addressed retrieval, but the footprint grows with context.

### Linear attention and fast weights

With a factorized or linearized kernel, the sequence can be regrouped into a recurrent matrix-valued state. A simple fast-weight form is:

```text
M_t = λ M_(t−1) + η k_t v_tᵀ
y_t = M_t q_t
```

The state is fixed at `O(d²)` for a single head. The price is that the state superposes associations. If `q_iᵀ k_j` is large for the wrong pair, the wrong value contributes to the read.

### Biological Hebbian rule

The local rule `ΔW_ij = η x_i y_j` strengthens a connection when its pre- and post-synaptic activities co-occur. The artifact uses the same algebraic shape as a teaching model: a key acts as a pre-synaptic pattern, a value acts as a post-synaptic pattern, and their outer product is the temporary write.

### BDH / BDH-CQ framing

The Dragon Hatchling (BDH) paper describes a scale-free network of locally interacting neuron particles and a GPU-friendly formulation. In the conceptual reduction used here, reasoning and memory share a synaptic substrate: activations drive reads and also modulate the fast state. Sparse non-negative firing is represented by thresholded ReLU:

```text
φ(x) = max(0, x − θ)
```

The app uses this to expose one plausible signal-to-noise intervention: suppressing weak, sign-canceling activity can reduce cross-talk in this toy matrix. It does **not** claim that this threshold alone reproduces BDH-GPU or BDH-CQ. BDH-GPU is described as a ReLU/low-rank, linear-attention-compatible GPU formulation; it should not be conflated with a standard Mamba-style selective state-space model.

## Reproduction guide

```bash
git clone <repository-url>
cd <repository-directory>
pnpm install
pnpm --filter @workspace/synaptic-memory-lab run dev
```

Open the preview served by the project. The app is a static Vite bundle and does not require `DATABASE_URL`, a model key, or a third-party integration.

For a production build:

```bash
pnpm --filter @workspace/synaptic-memory-lab run typecheck
pnpm --filter @workspace/synaptic-memory-lab run build
```

### Suggested experiment protocol

Record the baseline with `λ=1`, moderate `η`, two associations, and zero interference. Then hold decay and plasticity constant while increasing only key collinearity. Next hold collinearity low and increase load from 2 to 20. Finally repeat both sweeps with BDH sparse mode enabled. Report the full parameter vector, cosine similarity, MSE, and whether the failure was caused by decay, collision, or load.

## Evidence discipline

This project separates three evidence classes:

1. **Developer-reported results:** results reported by the BDH authors, such as scaling-law experiments or task metrics in the Dragon Hatchling preprint.
2. **External reproductions:** independent implementations or replications; none are claimed by this artifact unless linked and verified.
3. **Production deployments:** operational integrations such as a hosted training or inference deployment; deployment evidence is not the same as an independently reproduced benchmark.

The heatmap and metrics in this repository are original browser experiments on a toy associative-memory engine. They are not BDH benchmark results.

## Primary literature

1. Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, and Michał Bartoszkiewicz. **“The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.”** arXiv:2509.26507, 2025. [arXiv](https://arxiv.org/abs/2509.26507)
2. Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. **“Retentive Network: A Successor to Transformer for Large Language Models.”** arXiv:2307.08621, 2023. [arXiv](https://arxiv.org/abs/2307.08621)
3. Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. **“Gated Linear Attention Transformers with Hardware-Efficient Training.”** arXiv:2312.06635, 2023/2024 revision. [arXiv](https://arxiv.org/abs/2312.06635)
4. D. Tyulmankov et al. **“Meta-learning synaptic plasticity and memory addressing for continual familiarity detection.”** *Neuron*, 2022. The work studies learned synaptic plasticity and memory addressing over long intervals.
5. Thomas Miconi. **“Hebbian learning with gradients: Hebbian convolutional neural networks with modern deep learning frameworks.”** arXiv:2107.01729, 2021. Included as a methodological bridge for implementing local plasticity in modern training systems.

## AI assistance disclosure

AI assistance was used to draft the initial implementation, interaction copy, and technical documentation. The scientific framing, equations, experiment variables, evidence boundaries, and citations should be reviewed by the author before submission. The browser engine is intentionally transparent and should be inspected rather than treated as an authoritative simulator of a biological nervous system.

## Asset provenance

The artifact uses code-generated interface graphics only: the matrix heatmap, vector bars, diagrams, and motion are rendered from live state in the browser. No third-party image assets are included.

## License

MIT License. See `LICENSE` if a license file is added to the public repository.
