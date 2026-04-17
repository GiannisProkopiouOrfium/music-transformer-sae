---
layout: default
title: Sparse Activation Steering for Interpretable Music Control
---

<style>
  audio { width: 100%; max-width: 360px; }
  .audio-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0 24px 0; }
  .audio-grid.three-col { grid-template-columns: 1fr 1fr 1fr; }
  .audio-cell { background: #f6f8fa; border-radius: 8px; padding: 12px; text-align: center; }
  .audio-cell strong { display: block; margin-bottom: 6px; font-size: 0.9em; }
  .audio-cell em { display: block; margin-bottom: 8px; font-size: 0.8em; color: #586069; }
  .method-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold; margin-right: 4px; }
  .badge-dm { background: #ddf4ff; color: #0969da; }
  .badge-sas { background: #dafbe1; color: #1a7f37; }
  .badge-smooth { background: #fff8c5; color: #9a6700; }
  .results-table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }
  .results-table th, .results-table td { padding: 8px 12px; border: 1px solid #d0d7de; text-align: center; }
  .results-table th { background: #f6f8fa; }
  .results-table tr:hover { background: #f6f8fa; }
  h3 { margin-top: 32px; }
  .section-divider { border: none; border-top: 2px solid #d0d7de; margin: 40px 0; }
  .toc { background: #f6f8fa; border-radius: 8px; padding: 16px 24px; margin: 20px 0; }
  .toc ul { margin: 0; padding-left: 20px; }
  .toc li { margin: 4px 0; }
  @media (max-width: 700px) {
    .audio-grid, .audio-grid.three-col { grid-template-columns: 1fr; }
  }
</style>

# Sparse Activation Steering for Interpretable Attribute Control in Symbolic Music Generation

**Authors:** Ioannis Prokopiou¹, Pantelis Vikatos², Maximos Kaliakatsos-Papakostas³, Theodoros Giannakopoulos², Themos Stafylakis⁴

**Affiliations:**
¹ Athens University of Economics and Business &nbsp;
² Orfium &nbsp;
³ Hellenic Mediterranean University &nbsp;
⁴ NCSR "Demokritos" & Archimedes/Athena R.C.

<div style="text-align: center; margin: 20px 0;">
  <a href="https://github.com/salu133445/mmt" class="btn">💻 Code Repository</a>
</div>

---

<div class="toc">
<strong>Contents</strong>
<ul>
  <li><a href="#abstract">Abstract</a></li>
  <li><a href="#methods">Methods Overview</a></li>
  <li><a href="#results">Results Summary</a></li>
  <li><a href="#audio-examples">Audio Examples</a>
    <ul>
      <li><a href="#single-pitch">Single Attribute — Pitch</a></li>
      <li><a href="#single-duration">Single Attribute — Duration</a></li>
      <li><a href="#dual-steering">Dual Steering (Pitch + Duration)</a></li>
      <li><a href="#smooth-steering">Smooth Steering Schedules</a></li>
    </ul>
  </li>
  <li><a href="#code">Code & Reproducibility</a></li>
  <li><a href="#citation">Citation</a></li>
</ul>
</div>

---

## Abstract {#abstract}

Transformer-based architectures have significantly advanced the generation of complex symbolic sequences, yet a critical gap remains in achieving fine-grained, interpretable control over discrete signal attributes. This paper investigates the mechanistic interpretability of the Multitrack Music Transformer (MMT) and proposes a framework for deterministic attribute modulation without retraining, bridging this gap via inference-time activation steering.

We compare two complementary approaches:

- **Difference-in-Means (DiffMean)** — a dense steering method that isolates latent directions for musical attributes in the residual stream and injects steering vectors at all 12 transformer layers.
- **Sparse Activation Steering (SAS)** — a novel approach using Sparse Autoencoders (SAEs) trained on per-layer activations to decompose the residual stream into interpretable features, enabling fine-grained control through a single layer with monosemantic feature manipulation.

We validate the Linear Representation Hypothesis in the symbolic music domain and introduce a **Dual Steering** framework utilizing Gram-Schmidt Orthogonalization to simultaneously control **Pitch** and **Duration** without feature interference.

---

## Methods Overview {#methods}

### Difference-in-Means (DiffMean) — Dense Steering

Steering vectors are computed by contrasting "High" and "Low" attribute clusters from the Symbolic Orchestral Database (SOD):

$$v^{(\ell)} = \mathbb{E}[a^{(\ell)} \mid \text{high}] - \mathbb{E}[a^{(\ell)} \mid \text{low}]$$

At inference, vectors are injected into the residual stream at all 12 layers:

$$h_{\text{steer}}^{(\ell)} \leftarrow h^{(\ell)} + \alpha \, v^{(\ell)}$$

### Sparse Activation Steering (SAS) — Interpretable Steering

A Sparse Autoencoder (SAE) with 8× expansion (512 → 4096 features) is trained per layer. At layer 10, activations are decomposed into monosemantic features, and steering is applied in the sparse feature space:

$$\tilde{a}_\ell = \text{Dec}\!\bigl(\text{TopK}(f(a_\ell) + \lambda\, \mathbf{v}_{\text{SAS}})\bigr) + \Delta$$

where $\Delta = a_\ell - \text{Dec}(\text{Enc}(a_\ell))$ is a correction term that prevents reconstruction error from accumulating.

### Dual Steering

For simultaneous Pitch + Duration control, Gram-Schmidt orthogonalization decouples the two steering directions, ensuring changes to one attribute do not interfere with the other.

---

## Results Summary {#results}

### Conditioned Dual Steering — Success Rates & Quality

<table class="results-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>Strategy</th>
      <th>Both SR (%)</th>
      <th>δ (Degradation)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><span class="method-badge badge-sas">SAS</span></td><td>Gram-Schmidt + EK2</td><td><strong>91.0 ± 3.3</strong></td><td><strong>2.88 ± 0.53</strong></td></tr>
    <tr><td><span class="method-badge badge-sas">SAS</span></td><td>Expanded K 2×</td><td>91.7 ± 3.2</td><td>2.95 ± 0.57</td></tr>
    <tr><td><span class="method-badge badge-sas">SAS</span></td><td>Gram-Schmidt (dur)</td><td>92.7 ± 3.0</td><td>4.11 ± 0.74</td></tr>
    <tr><td><span class="method-badge badge-dm">DiffMean</span></td><td>Gram-Schmidt (pitch)</td><td>85.3 ± 3.3</td><td>4.39 ± 0.68</td></tr>
  </tbody>
</table>

### Fréchet Music Distance (FMD) — Rank Preservation

<table class="results-table">
  <thead>
    <tr><th>Estimator</th><th>SAS Avg FMD</th><th>DiffMean Avg FMD</th><th>SAS Advantage</th></tr>
  </thead>
  <tbody>
    <tr><td>MLE</td><td>281.6</td><td>580.4</td><td>+51.5%</td></tr>
    <tr><td>Ledoit-Wolf</td><td>271.3</td><td>480.5</td><td>+43.5%</td></tr>
    <tr><td>OAS</td><td>271.0</td><td>469.8</td><td>+42.3%</td></tr>
  </tbody>
</table>

Rankings are **fully preserved** across all covariance estimators.

---

## Audio Examples {#audio-examples}

All examples use **conditioned generation**: the model receives the first 16 beats of an existing piece as context, and generates a continuation. Steering is applied only to the continuation. Compare baseline (no steering, α=0 / λ=0) against steered outputs for both methods.

<hr class="section-divider">

### Single Attribute — Pitch Control {#single-pitch}

#### Low → High Pitch

<span class="method-badge badge-dm">DiffMean</span> Song 1109 &nbsp; | &nbsp; Low pitch context → steer upward

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline (α = 0)</strong>
    <audio controls><source src="assets/audio/samples/DM_pitch_low_to_high_song1109_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (α = +0.1)</strong>
    <audio controls><source src="assets/audio/samples/DM_pitch_low_to_high_song1109_steered_a+0.1.mp3" type="audio/mpeg"></audio>
  </div>
</div>

<span class="method-badge badge-sas">SAS</span> Musicalion-3512 &nbsp; | &nbsp; Low pitch context → steer upward

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline (λ = 0)</strong>
    <audio controls><source src="assets/audio/samples/SAS_pitch_low_to_high_Musicalion-3512_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (λ = +0.75)</strong>
    <audio controls><source src="assets/audio/samples/SAS_pitch_low_to_high_Musicalion-3512_steered_l+0.75.mp3" type="audio/mpeg"></audio>
  </div>
</div>

#### High → Low Pitch

<span class="method-badge badge-dm">DiffMean</span> Song 708 &nbsp; | &nbsp; High pitch context → steer downward

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline (α = 0)</strong>
    <audio controls><source src="assets/audio/samples/DM_pitch_high_to_low_song708_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (α = −0.5)</strong>
    <audio controls><source src="assets/audio/samples/DM_pitch_high_to_low_song708_steered_a-0.5.mp3" type="audio/mpeg"></audio>
  </div>
</div>

<span class="method-badge badge-sas">SAS</span> Kunstderfuge-389 &nbsp; | &nbsp; High pitch context → steer downward

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline (λ = 0)</strong>
    <audio controls><source src="assets/audio/samples/SAS_pitch_high_to_low_Kunstderfuge-389_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (λ = +0.5)</strong>
    <audio controls><source src="assets/audio/samples/SAS_pitch_high_to_low_Kunstderfuge-389_steered_l+0.5.mp3" type="audio/mpeg"></audio>
  </div>
</div>

<hr class="section-divider">

### Single Attribute — Duration Control {#single-duration}

#### Short → Long Duration

<span class="method-badge badge-dm">DiffMean</span> Kunstderfuge-766 &nbsp; | &nbsp; Short notes → steer longer

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline (α = 0)</strong>
    <audio controls><source src="assets/audio/samples/DM_duration_low_to_high_Kunstderfuge-766_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (α = +0.5)</strong>
    <audio controls><source src="assets/audio/samples/DM_duration_low_to_high_Kunstderfuge-766_steered_a+0.5.mp3" type="audio/mpeg"></audio>
  </div>
</div>

<span class="method-badge badge-sas">SAS</span> Musicalion-962 &nbsp; | &nbsp; Short notes → steer longer

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline (λ = 0)</strong>
    <audio controls><source src="assets/audio/samples/SAS_duration_short_to_long_Musicalion-962_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (λ = +1.0)</strong>
    <audio controls><source src="assets/audio/samples/SAS_duration_short_to_long_Musicalion-962_steered_l+1.0.mp3" type="audio/mpeg"></audio>
  </div>
</div>

#### Long → Short Duration

<span class="method-badge badge-dm">DiffMean</span> Musicalion-1698 &nbsp; | &nbsp; Long notes → steer shorter

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline (α = 0)</strong>
    <audio controls><source src="assets/audio/samples/DM_duration_high_to_low_Musicalion-1698_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (α = −0.5)</strong>
    <audio controls><source src="assets/audio/samples/DM_duration_high_to_low_Musicalion-1698_steered_a-0.5.mp3" type="audio/mpeg"></audio>
  </div>
</div>

<span class="method-badge badge-sas">SAS</span> Musicalion-1416 &nbsp; | &nbsp; Long notes → steer shorter

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline (λ = 0)</strong>
    <audio controls><source src="assets/audio/samples/SAS_duration_long_to_short_Musicalion-1416_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (λ = −1.5)</strong>
    <audio controls><source src="assets/audio/samples/SAS_duration_long_to_short_Musicalion-1416_steered_l-1.5.mp3" type="audio/mpeg"></audio>
  </div>
</div>

<hr class="section-divider">

### Dual Steering — Simultaneous Pitch + Duration {#dual-steering}

Gram-Schmidt orthogonalization ensures independent control of both attributes.

#### Low Pitch + Short → High Pitch + Long

<span class="method-badge badge-dm">DiffMean</span> Kunstderfuge-1367

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline</strong>
    <audio controls><source src="assets/audio/samples/DM_dual_low_short_to_high_long_Kunstderfuge-1367_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (α_p=+1.5, α_d=+0.5)</strong>
    <audio controls><source src="assets/audio/samples/DM_dual_low_short_to_high_long_Kunstderfuge-1367_steered_ap+1.5_ad+0.5.mp3" type="audio/mpeg"></audio>
  </div>
</div>

<span class="method-badge badge-sas">SAS</span> Musicalion-1109

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline</strong>
    <audio controls><source src="assets/audio/samples/SAS_dual_low_to_high_long_Musicalion-1109_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (λ_p=+0.50, λ_d=+0.75, δ=0.07)</strong>
    <audio controls><source src="assets/audio/samples/SAS_dual_low_to_high_long_Musicalion-1109_steered_lp+0.50_ld+0.75_deg0.07.mp3" type="audio/mpeg"></audio>
  </div>
</div>

#### High Pitch + Long → Low Pitch + Short

<span class="method-badge badge-dm">DiffMean</span> Musicalion-3672

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline</strong>
    <audio controls><source src="assets/audio/samples/DM_dual_high_long_to_low_long_Musicalion-3672_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (α_p=−1.5, α_d=−1.2)</strong>
    <audio controls><source src="assets/audio/samples/DM_dual_high_long_to_low_long_Musicalion-3672_steered_ap-1.5_ad-1.2.mp3" type="audio/mpeg"></audio>
  </div>
</div>

<span class="method-badge badge-sas">SAS</span> Musicalion-708

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Baseline</strong>
    <audio controls><source src="assets/audio/samples/SAS_dual_high_to_low_short_Musicalion-708_baseline.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Steered (λ_p=−0.50, λ_d=−0.25, δ=1.00)</strong>
    <audio controls><source src="assets/audio/samples/SAS_dual_high_to_low_short_Musicalion-708_steered_lp-0.50_ld-0.25_deg1.00.mp3" type="audio/mpeg"></audio>
  </div>
</div>

<hr class="section-divider">

### Smooth Steering Schedules {#smooth-steering}

Comparison of **abrupt** (instant full strength) vs **smooth** (gradual ramp) λ scheduling. Both use DiffMean steering.

#### Pitch — Low → High

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Abrupt (instant α)</strong>
    <em>Song 353</em>
    <audio controls><source src="assets/audio/samples/SMOOTH_pitch_low_to_high_song353_abrupt.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Warmup-Hold (32 beats ramp)</strong>
    <em>Song 353</em>
    <audio controls><source src="assets/audio/samples/SMOOTH_pitch_low_to_high_song353_warmup_hold_32beats.mp3" type="audio/mpeg"></audio>
  </div>
</div>

#### Pitch — High → Low

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Abrupt (α = −1.0)</strong>
    <em>Song 389</em>
    <audio controls><source src="assets/audio/samples/SMOOTH_pitch_high_to_low_song389_abrupt_a-1.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Gradual (256 beats ramp)</strong>
    <em>Song 389</em>
    <audio controls><source src="assets/audio/samples/SMOOTH_pitch_high_to_low_song389_gradual_256beats.mp3" type="audio/mpeg"></audio>
  </div>
</div>

#### Duration — Short → Long

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Abrupt</strong>
    <em>Song 766</em>
    <audio controls><source src="assets/audio/samples/SMOOTH_duration_short_to_long_song766_abrupt.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Gradual (128 beats ramp)</strong>
    <em>Song 766</em>
    <audio controls><source src="assets/audio/samples/SMOOTH_duration_short_to_long_song766_gradual_128beats.mp3" type="audio/mpeg"></audio>
  </div>
</div>

#### Duration — Long → Short

<div class="audio-grid">
  <div class="audio-cell">
    <strong>Abrupt</strong>
    <em>Song 3708</em>
    <audio controls><source src="assets/audio/samples/SMOOTH_duration_long_to_short_song3708_abrupt.mp3" type="audio/mpeg"></audio>
  </div>
  <div class="audio-cell">
    <strong>Gradual (128 beats ramp)</strong>
    <em>Song 3708</em>
    <audio controls><source src="assets/audio/samples/SMOOTH_duration_long_to_short_song3708_gradual_128beats.mp3" type="audio/mpeg"></audio>
  </div>
</div>

---

## Code & Reproducibility {#code}

### Repository Structure

```
├── steering_interventions/       # DiffMean pipeline
│   ├── data_curator.py           # Dataset preparation & segmentation
│   ├── activation_extractor.py   # Extract residual stream activations
│   ├── steering_vector_computer.py   # Compute DiffMean vectors
│   ├── dual_steering/            # Gram-Schmidt dual control
│   └── conditioned_evaluator.py  # Conditioned evaluation
│
├── sparse_steering/              # SAS pipeline
│   ├── train_sae.py              # Train Sparse Autoencoders (per layer)
│   ├── compute_sas_vectors.py    # Compute SAS steering vectors
│   ├── steered_generator_sas.py  # SAS generation with correction term
│   ├── dual_steering/            # SAS dual steering
│   └── conditioned_evaluator_sas.py  # Conditioned SAS evaluation
│
├── metrics_evaluation/           # FMD, quality metrics
├── feature_explainability/       # Interpretability analysis
└── mmt/                          # Base MMT model (Dong et al.)
```

### Quick Start — DiffMean

```bash
# 1. Curate contrastive dataset
python steering_interventions/data_curator.py \
    --concept average_pitch --data_dir data/sod

# 2. Extract activations
python steering_interventions/activation_extractor.py \
    --concept average_pitch --checkpoint exp/sod/ape/best_model.pt

# 3. Compute steering vectors
python steering_interventions/steering_vector_computer.py \
    --concept average_pitch

# 4. Generate with steering
python steering_interventions/steered_generator.py \
    --concept average_pitch --alpha 0.5
```

### Quick Start — SAS

```bash
# 1. Train SAEs on all 12 layers
python sparse_steering/train_sae.py \
    --checkpoint exp/sod/ape/best_model.pt

# 2. Encode concept activations through SAEs
python sparse_steering/encode_concept_activations.py \
    --concept average_pitch

# 3. Compute sparse steering vectors
python sparse_steering/compute_sas_vectors.py \
    --concept average_pitch --tau 0.08

# 4. Generate with SAS steering
python sparse_steering/steered_generator_sas.py \
    --concept average_pitch --lambda_strength 0.5 --layer 10
```

### Quick Start — Dual Steering

```bash
# DiffMean dual (Gram-Schmidt)
python steering_interventions/dual_steering/test_multi_steering.py \
    --strategy gram_schmidt_pitch \
    --alpha_pitch 1.0 --alpha_duration -0.5

# SAS dual (conditioned, with Expanded K)
python sparse_steering/dual_steering/test_dual_conditioned.py \
    --strategy expanded_k --k_multiplier 2 \
    --lambda_pitch 0.5 --lambda_duration 0.75
```

---

## Citation {#citation}

```bibtex
@inproceedings{prokopiou2026sparse,
  title={Sparse Activation Steering for Interpretable Attribute Control 
         in Symbolic Music Generation},
  author={Prokopiou, Ioannis and Vikatos, Pantelis and 
          Kaliakatsos-Papakostas, Maximos and 
          Giannakopoulos, Theodoros and Stafylakis, Themos},
  booktitle={Proceedings of the International Society for Music 
             Information Retrieval Conference (ISMIR)},
  year={2026}
}
```

---

## Acknowledgments

This work was supported by Athens University of Economics and Business, Orfium, Hellenic Mediterranean University, NCSR "Demokritos", and Archimedes/Athena R.C.

Base model: [Multitrack Music Transformer (MMT)](https://github.com/salu133445/mmt) by Hao-Wen Dong et al.

<div style="text-align: center; margin-top: 40px; font-size: 0.9em; color: #666;">
  © 2026 | <a href="https://github.com/GiannisProkopiouOrfium">Ioannis Prokopiou</a>
</div>
