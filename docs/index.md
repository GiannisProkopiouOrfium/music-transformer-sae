---
layout: default
title: Latent Space Disentanglement via Activation Steering
---

# Latent Space Disentanglement via Activation Steering for Interpretable Attribute Control in Symbolic Music Generation

**Authors:** Ioannis Prokopiou¹, Pantelis Vikatos², Maximos Kaliakatsos-Papakostas³, Theodoros Giannakopoulos², Themos Stafylakis⁴

**Affiliations:**  
¹ Athens University of Economics and Business  
² Orfium  
³ Hellenic Mediterranean University  
⁴ NCSR "Demokritos" & Archimedes/Athena R.C.

---

<div style="text-align: center; margin: 20px 0;">
  <a href="YOUR_PAPER_LINK.pdf" class="btn">📄 Read Paper</a>
  <a href="https://github.com/GiannisProkopiouOrfium/music-transformer-sae" class="btn">💻 View Code</a>
</div>

---

## Abstract

Transformer-based architectures have significantly advanced the generation of complex symbolic sequences, yet a significant gap remains in achieving fine-grained, interpretable control over discrete signal attributes. This paper investigates the mechanistic interpretability of the Multitrack Music Transformer (MMT) and proposes a framework for deterministic attribute modulation without retraining to bridge this gap via inference-time activation steering.

Utilizing the Difference-in-Means (DiffMean) methodology, we isolate latent directions for signal attributes, specifically **Pitch** and **Duration**, within the residual stream. We validate the Linear Representation Hypothesis in this domain and introduce a **Dual Steering** framework utilizing Gram-Schmidt Orthogonalization to address feature entanglement.

---

## Approach & Methodology

Our framework enables inference-time steerability without expensive fine-tuning.

### Key Components

* **Model:** We utilize the pre-trained **Multitrack Music Transformer (MMT)**, which processes musical events (Pitch, Duration, Instrument, etc.) as discrete tokens.

* **Latent Vector Extraction:** We use the **Difference-in-Means (DiffMean)** methodology to calculate steering vectors by contrasting "High" and "Low" attribute clusters (e.g., High Pitch vs. Low Pitch) from the Symbolic Orchestral Database (SOD).

* **Inference-Time Steering:** We inject these vectors into the residual stream during generation:

$$
h_{steer}^{(l)} \leftarrow h^{(l)} + \alpha v^{(l)}
$$

* **Dual Steering (Disentanglement):** To control Pitch and Duration simultaneously without interference, we apply **Gram-Schmidt Orthogonalization**. This mathematically decouples correlated features, ensuring independent control.

### Pipeline Overview

1. **Dataset Curation:** Classify songs by musical attributes (pitch, duration) using calibrated thresholds
2. **Activation Extraction:** Extract residual stream activations from all 12 transformer layers
3. **Steering Vector Computation:** Calculate difference vectors: `v = mean(high) - mean(low)`
4. **Vector Composition:** Apply Gram-Schmidt orthogonalization for dual steering
5. **Generation:** Inject steering vectors at inference time to control musical attributes

---

## Results Summary

### Single Steering
- **Pitch Control:** 15-25 semitone changes with minimal quality degradation
- **Duration Control:** 8-12 tick changes maintaining musical coherence

### Dual Steering
- **Success Rate:** 88.5% dual success (both attributes steered correctly)
- **Best Strategy:** Gram-Schmidt with pitch priority
- **Quality:** Average degradation of 2.14 (near-baseline)

### Conditioned Dual Steering
Testing ability to override strong conditioning context:

| Scenario | Success Rate | Avg Degradation |
|----------|--------------|-----------------|
| Low+Short → High+Long | **96.1%** | 3.03 |
| Low+Long → High+Short | 90.6% | **1.33** |
| High+Long → Low+Short | 85.6% | 2.59 |
| High+Short → Low+Long | 82.2% | 5.80 |

**Key Finding:** Upward steering (low→high) significantly easier than downward (high→low), with 82-96% success even when fighting 16-beat conditioning context.

---

## Audio Examples

All examples use the same conditioning context (first 16 beats) to demonstrate steering effectiveness. Listen to the dramatic changes between baseline (no intervention) and steered generations.

### 1. Pitch Control

Demonstrating the ability to override pitch characteristics in the conditioning context.

#### Scenario 1: High Context → Low Continuation

<table>
  <tr>
    <th>Baseline (No Steering)</th>
    <th>Steered (α=-0.5)</th>
  </tr>
  <tr>
    <td><audio controls><source src="assets/audio/pitch_audios/high_to_low/pitch_down_baseline_h708-0.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
    <td><audio controls><source src="assets/audio/pitch_audios/high_to_low/pitxh_down_h708-m05.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
  </tr>
</table>

#### Scenario 2: Low Context → High Continuation

<table>
  <tr>
    <th>Baseline (No Steering)</th>
    <th>Steered (α=+1.0)</th>
  </tr>
  <tr>
    <td><audio controls><source src="assets/audio/pitch_audios/low_to_high/pitch_up_baseline_l1109-0.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
    <td><audio controls><source src="assets/audio/pitch_audios/low_to_high/pitch_up_l1109-01.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
  </tr>
</table>

---

### 2. Duration Control

Demonstrating control over rhythmic density and note lengths.

#### Scenario 1: Long Context → Short Continuation

<table>
  <tr>
    <th>Baseline (No Steering)</th>
    <th>Steered (α=-0.5)</th>
  </tr>
  <tr>
    <td><audio controls><source src="assets/audio/duration_audios/high_to_low/Musicalion-1698_baseline.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
    <td><audio controls><source src="assets/audio/duration_audios/high_to_low/Musicalion-1698_m0.5.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
  </tr>
</table>

#### Scenario 2: Short Context → Long Continuation

<table>
  <tr>
    <th>Baseline (No Steering)</th>
    <th>Steered (α=+0.5)</th>
  </tr>
  <tr>
    <td><audio controls><source src="assets/audio/duration_audios/low_to_high/Kunstderfuge-766_baseline.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
    <td><audio controls><source src="assets/audio/duration_audios/low_to_high/Kunstderfuge-766_0.5.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
  </tr>
</table>

---

### 3. Dual Steering (Disentangled Control)

Simultaneous control of **Pitch AND Duration** using Gram-Schmidt Orthogonalization to prevent feature interference.

#### Scenario 1: High Pitch, Short Duration → Low Pitch, Long Duration

<table>
  <tr>
    <th>Baseline (α_pitch=0, α_duration=0)</th>
    <th>Steered (α_pitch=-0.5, α_duration=+1.0)</th>
  </tr>
  <tr>
    <td><audio controls><source src="assets/audio/Dual Steering/High_pitch_short_dur_to_low_short_2940/Musicalion-2940_BASELINE.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
    <td><audio controls><source src="assets/audio/Dual Steering/High_pitch_short_dur_to_low_short_2940/Musicalion-2940_ap-0.5_ad+1.0.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-size: 0.9em; padding-top: 10px;">
      <strong>Result:</strong> 82.2% success rate | Avg Degradation: 5.80
    </td>
  </tr>
</table>

#### Scenario 2: Low Pitch, Short Duration → High Pitch, Long Duration

<table>
  <tr>
    <th>Baseline (α_pitch=0, α_duration=0)</th>
    <th>Steered (α_pitch=+1.5, α_duration=+0.5)</th>
  </tr>
  <tr>
    <td><audio controls><source src="assets/audio/Dual Steering/Low_pitch_short_dur_to_high_long_1367/Kunstderfuge-1367_lowpitchshortdurationtohighlong_BASELINE.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
    <td><audio controls><source src="assets/audio/Dual Steering/Low_pitch_short_dur_to_high_long_1367/16_Kunstderfuge-1367_lowpitchshortdurationtohighlong_ap+1.5_ad+0.5.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-size: 0.9em; padding-top: 10px;">
      <strong>Result:</strong> 96.1% success rate | Avg Degradation: 3.03
    </td>
  </tr>
</table>

#### Scenario 3: Low Pitch, Long Duration → High Pitch, Short Duration

<table>
  <tr>
    <th>Baseline (α_pitch=0, α_duration=0)</th>
    <th>Steered (α_pitch=+1.0, α_duration=-0.5)</th>
  </tr>
  <tr>
    <td><audio controls><source src="assets/audio/Dual Steering/low_pitch_long_dur_to_low_short/Musicalion-2743_lowpitchlongdurationtohighshort_BASELINE.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
    <td><audio controls><source src="assets/audio/Dual Steering/low_pitch_long_dur_to_low_short/19_Musicalion-2743_lowpitchlongdurationtohighshort_ap+1.0_ad-0.5.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-size: 0.9em; padding-top: 10px;">
      <strong>Result:</strong> 90.6% success rate | Avg Degradation: 1.33 (best quality)
    </td>
  </tr>
</table>

#### Scenario 4: High Pitch, Long Duration → Low Pitch, Short Duration

<table>
  <tr>
    <th>Baseline (α_pitch=0, α_duration=0)</th>
    <th>Steered (α_pitch=-1.5, α_duration=-1.2)</th>
  </tr>
  <tr>
    <td><audio controls><source src="assets/audio/Dual Steering/High_pitch_long_dur_to_low_long_3672/Musicalion-3672_highpitchlongdurationtolowshort_BASELINE.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
    <td><audio controls><source src="assets/audio/Dual Steering/High_pitch_long_dur_to_low_long_3672/10_Musicalion-3672_highpitchlongdurationtolowshort_ap-1.5_ad-1.2.mp3" type="audio/wav">Your browser does not support audio.</audio></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-size: 0.9em; padding-top: 10px;">
      <strong>Result:</strong> 85.6% success rate | Avg Degradation: 2.59
    </td>
  </tr>
</table>

---

## Code & Reproducibility

The complete codebase is available on GitHub:

**Repository:** [https://github.com/GiannisProkopiouOrfium/music-transformer-sae](https://github.com/GiannisProkopiouOrfium/music-transformer-sae)

### Key Scripts

- `steering_interventions/data_curator.py` - Dataset preparation and segmentation
- `steering_interventions/activation_extractor.py` - Extract activations from transformer layers
- `steering_interventions/steering_vector_computer.py` - Compute steering vectors via DiffMean
- `steering_interventions/dual_steering/vector_composition.py` - Gram-Schmidt orthogonalization
- `steering_interventions/dual_steering/test_multi_conditioned.py` - Conditioned dual steering

### Quick Start

```bash
# Clone repository
git clone https://github.com/GiannisProkopiouOrfium/music-transformer-sae.git
cd music-transformer-sae

# Install dependencies
pip install -r requirements.txt

# Run dual steering experiment
python steering_interventions/dual_steering/test_multi_conditioned.py \
    --pitch_vectors outputs/steering_vectors/average_pitch_steering_vectors.pt \
    --duration_vectors outputs/steering_vectors/average_duration_steering_vectors.pt \
    --output_dir outputs/phase4_conditioned \
    --strategies gram_schmidt_pitch
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{prokopiou2026latent,
  title={Latent Space Disentanglement via Activation Steering for Interpretable Attribute Control in Symbolic Music Generation},
  author={Prokopiou, Ioannis and Vikatos, Pantelis and Kaliakatsos-Papakostas, Maximos and Giannakopoulos, Theodoros and Stafylakis, Themos},
  booktitle={EUSIPCO 2026},
  year={2026}
}
```

---

## Acknowledgments

This work was supported by Athens University of Economics and Business, Orfium, Hellenic Mediterranean University, NCSR "Demokritos", and Archimedes/Athena R.C.

Base model: [Multitrack Music Transformer (MMT)](https://github.com/salu133445/mmt) by Hao-Wen Dong and colleagues.

---

<div style="text-align: center; margin-top: 40px; font-size: 0.9em; color: #666;">
  © 2026 | <a href="https://github.com/GiannisProkopiouOrfium">Ioannis Prokopiou</a>
</div>
