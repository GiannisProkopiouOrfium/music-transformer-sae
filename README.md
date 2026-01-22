# Multitrack Music Transformer - Activation Steering Extension

This repository contains the official implementation of "Multitrack Music Transformer" (ICASSP 2023) **extended with Activation Steering for Interpretable Attribute Control**.

## 🎵 Research Papers

### Original MMT Paper
__Multitrack Music Transformer__<br>
Hao-Wen Dong, Ke Chen, Shlomo Dubnov, Julian McAuley and Taylor Berg-Kirkpatrick<br>
_IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, 2023<br>
[[homepage](https://salu133445.github.io/mmt/)]
[[paper](https://arxiv.org/pdf/2207.06983.pdf)]
[[code](https://github.com/salu133445/mmt)]

### Activation Steering Extension
__Latent Space Disentanglement via Activation Steering for Interpretable Attribute Control in Symbolic Music Generation__<br>
Ioannis Prokopiou, Pantelis Vikatos, Maximos Kaliakatsos-Papakostas, Theodoros Giannakopoulos, Themos Stafylakis<br>
_EUSIPCO 2026_<br>
[[demo page](https://giannisprokopiouorfium.github.io/music-transformer-sae/)]
[[code](https://github.com/GiannisProkopiouOrfium/music-transformer-sae)]

---

## 🎯 What's New: Activation Steering

This fork extends the original MMT with **mechanistic interpretability** and **inference-time steering** capabilities:

### Key Features

- ✅ **Single-Concept Steering**: Control pitch or duration independently
- ✅ **Dual-Concept Steering**: Simultaneously control both pitch and duration using Gram-Schmidt orthogonalization
- ✅ **Context Override**: Successfully steer against strong conditioning context (82-96% success rate)
- ✅ **No Retraining Required**: All control happens at inference time via activation injection
- ✅ **Validated on SOD Dataset**: Tested on 720 generation experiments across 4 fight scenarios

### Results Summary

| Control Type | Success Rate | Quality (Degradation) |
|--------------|--------------|----------------------|
| Single (Pitch) | ~95% | 2-3 semitones, minimal degradation |
| Single (Duration) | ~93% | 8-12 ticks, minimal degradation |
| Dual (Both) | **88.5%** | Average 2.14 degradation |
| Conditioned Dual | **82-96%** | Scenario-dependent (1.33-5.80) |

**🎧 [Listen to Audio Examples](https://giannisprokopiouorfium.github.io/music-transformer-sae/)**

---

## Content

- [Prerequisites](#prerequisites)
- [Activation Steering Quick Start](#activation-steering-quick-start)
- [Original MMT Pipeline](#original-mmt-pipeline)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Generation](#generation-inference)
- [Activation Steering Pipeline](#activation-steering-pipeline)
  - [Dataset Curation](#dataset-curation)
  - [Activation Extraction](#activation-extraction)
  - [Steering Vector Computation](#steering-vector-computation)
  - [Dual Steering](#dual-steering)
- [Citation](#citation)

---

## Prerequisites

We recommend using Conda. You can create the environment with the following command.

```sh
conda env create -f environment.yml
```

Or install the steering-specific requirements:

```sh
pip install -r requirements_new.txt
```

---

## Activation Steering Quick Start

### 1. Run Complete Dual Steering Experiment

```bash
# Phase 4: Conditioned dual steering (fight scenarios)
python steering_interventions/dual_steering/test_multi_conditioned.py \
    --pitch_vectors outputs/steering_vectors/average_pitch_steering_vectors.pt \
    --duration_vectors outputs/steering_vectors/average_duration_steering_vectors.pt \
    --output_dir outputs/phase4_conditioned \
    --n_songs 5 \
    --conditioning_beats 16 \
    --alphas_pitch "0.5,0.75,1.0,1.25,1.5,-0.5,-0.75,-1.0,-1.25,-1.5" \
    --alphas_duration "0.5,0.75,1.0,1.25,1.5,-0.5,-0.75,-1.0,-1.25,-1.5" \
    --strategies gram_schmidt_pitch \
    --gpu 0
```

### 2. Convert Results to Audio

```bash
# Convert top 20 examples to audio for listening
python steering_interventions/dual_steering/convert_npy_to_audio.py \
    --input_dir outputs/phase4_conditioned \
    --listening_list outputs/phase4_conditioned/listening_list.json \
    --top_n 20
```

### 3. View Results

- **JSON Analysis**: `outputs/phase4_conditioned/conditioned_results.json`
- **Listening List**: `outputs/phase4_conditioned/listening_list.json`
- **Audio Files**: `outputs/phase4_conditioned/{scenario}/{strategy}/ap{X}_ad{Y}/`

---

## Original MMT Pipeline

## Preprocessing

### Preprocessed Datasets

The preprocessed datasets can be found [here](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/h3dong_ucsd_edu/Er7nrsVc7NhNtYVSdWpHMQwBS5U1dXo0q0eQEi2LW-DVGw).

Extract the files to `data/{DATASET_KEY}/processed/json` and `data/{DATASET_KEY}/processed/notes`, where `DATASET_KEY` is `sod`, `lmd`, `lmd_full` or `snd`.

### Preprocessing Scripts

__You can skip this section if you download the preprocessed datasets.__

#### Step 1 -- Download the datasets

Please download the [Symbolic orchestral database (SOD)](https://qsdfo.github.io/LOP/database.html). You may download it via command line as follows.

```sh
wget https://qsdfo.github.io/LOP/database/SOD.zip
```

We also support the following two datasets:

- [Lakh MIDI Dataset (LMD)](https://qsdfo.github.io/LOP/database.html):

  ```sh
  wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
  ```

- [SymphonyNet Dataset](https://symphonynet.github.io/):

  ```sh
  gdown https://drive.google.com/u/0/uc?id=1j9Pvtzaq8k_QIPs8e2ikvCR-BusPluTb&export=download
  ```

#### Step 2 -- Prepare the name list

Get a list of filenames for each dataset.

```sh
find data/sod/SOD -type f -name *.mid -o -name *.xml | cut -c 14- > data/sod/original-names.txt
```

> Note: Change the number in the cut command for different datasets.

#### Step 3 -- Convert the data

Convert the MIDI and MusicXML files into MusPy files for processing.

```sh
python convert_sod.py
```

> Note: You may enable multiprocessing with the `-j` option, for example, `python convert_sod.py -j 10` for 10 parallel jobs.

#### Step 4 -- Extract the note list

Extract a list of notes from the MusPy JSON files.

```sh
python extract.py -d sod
```

#### Step 5 -- Split training/validation/test sets

Split the processed data into training, validation and test sets.

```sh
python split.py -d sod
```

## Training

### Pretrained Models

The pretrained models can be found [here](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/h3dong_ucsd_edu/EqYq6KHrcltHvgJTmw7Nl6MBtv4szg4RUZUPXc4i_RgEkw).

### Training Scripts

Train a Multitrack Music Transformer model.

- Absolute positional embedding (APE):

  `python mmt/train.py -d sod -o exp/sod/ape -g 0`

- Relative positional embedding (RPE):

  `python mmt/train.py -d sod -o exp/sod/rpe --no-abs_pos_emb --rel_pos_emb -g 0`

- No positional embedding (NPE):

  `python mmt/train.py -d sod -o exp/sod/npe --no-abs_pos_emb --no-rel_pos_emb -g 0`

## Generation (Inference)

Generate new samples using a trained model.

```sh
python mmt/generate.py -d sod -o exp/sod/ape -g 0
```

## Evaluation

Evaluate the trained model using objective evaluation metrics.

```sh
python mmt/evaluate.py -d sod -o exp/sod/ape -ns 100 -g 0
```

---

## Activation Steering Pipeline

### Overview

The activation steering framework enables inference-time control of musical attributes without retraining:

```
1. Data Curation → 2. Activation Extraction → 3. Steering Vector Computation → 4. Dual Steering
```

### Dataset Curation

Extract songs with extreme attribute values for contrastive learning:

```bash
# Find songs with high/low pitch and duration
python steering_interventions/data_curator.py \
    --dataset_path data/sod/train.pkl \
    --output_path outputs/curated_songs.pkl \
    --concepts pitch average_duration \
    --n_samples 100
```

**Key Script**: `steering_interventions/data_curator.py`
- Computes statistics for each song (pitch, duration, note density, etc.)
- Identifies extreme examples using thresholds from `config.py`
- Saves curated dataset for activation extraction

### Activation Extraction

Extract activations from high/low attribute clusters:

```bash
# Extract activations for pitch concept
python steering_interventions/activation_extractor.py \
    --curated_songs outputs/curated_songs.pkl \
    --concept pitch \
    --output_path outputs/activations/pitch_activations.pkl \
    --model_path exp/sod/ape-abs-linear-complex/best_model.pt \
    --gpu 0
```

**Key Script**: `steering_interventions/activation_extractor.py`
- Hooks into transformer residual stream (layers 1-6)
- Saves activations separately for high/low clusters
- Output: `{concept}_activations.pkl` with high/low activation tensors

### Steering Vector Computation

Compute steering vectors using DiffMean methodology:

```bash
# Compute steering vectors (high - low activations)
python steering_interventions/steering_vector_computer.py \
    --activations outputs/activations/pitch_activations.pkl \
    --output_path outputs/steering_vectors/pitch_steering_vectors.pt \
    --method diffmean
```

**Key Script**: `steering_interventions/steering_vector_computer.py`
- Applies DiffMean: `v = mean(activations_high) - mean(activations_low)`
- Normalizes vectors for stable intervention
- Saves per-layer steering vectors: `{concept}_steering_vectors.pt`

### Dual Steering

Control multiple attributes simultaneously using orthogonalization:

```bash
# Run dual steering with Gram-Schmidt orthogonalization
python steering_interventions/dual_steering/dual_steering_experiment.py \
    --pitch_vectors outputs/steering_vectors/pitch_steering_vectors.pt \
    --duration_vectors outputs/steering_vectors/average_duration_steering_vectors.pt \
    --output_dir outputs/dual_steering \
    --strategies gram_schmidt_pitch gram_schmidt_duration simple_addition \
    --alphas_pitch "-1.5,-1.0,0.0,1.0,1.5" \
    --alphas_duration "-1.5,-1.0,0.0,1.0,1.5" \
    --gpu 0
```

**Key Scripts**:
- `steering_interventions/dual_steering/dual_steering_experiment.py`: Main experiment runner
- `steering_interventions/dual_steering/vector_composition.py`: Orthogonalization strategies
- `steering_interventions/dual_steering/test_multi_conditioned.py`: Phase 4 conditioned experiments

**Strategies**:
- `gram_schmidt_pitch`: Orthogonalize duration w.r.t. pitch (best: 88.5% success)
- `gram_schmidt_duration`: Orthogonalize pitch w.r.t. duration
- `simple_addition`: Direct addition (baseline, high interference)

---

## Deterministic Analysis

Evaluate steering quality using automated metrics:

```bash
# Run deterministic analysis on Phase 4 results
python run_mmt_deterministic_analysis.py \
    --input_folder outputs/phase4_conditioned \
    --output_file outputs/deterministic_analysis_results.json \
    --batch_size 32
```

**Key Components** (in `deterministic_analysis/`):
- `batch_deterministic_analyzer.py`: Batch analyzer with parallel processing
- `feature_specific_analyzers.py`: Pitch, duration, note density analyzers
- `music_quality_assessor.py`: Music quality degradation assessment
- `midi_feature_extractors.py`: Extract features from MIDI files

**Metrics**:
- **Pitch Change**: Average semitone shift (expected vs. actual)
- **Duration Change**: Average tick shift (expected vs. actual)
- **Quality Degradation**: Pitch range, note count, velocity variance comparison

---

## LLM Evaluation

Evaluate steering quality using language models:

```bash
# Run LLM evaluation on listening list
python run_llm_batch_evaluation.py \
    --listening_list outputs/phase4_conditioned/listening_list.json \
    --output_file outputs/llm_evaluation_results.json \
    --model gpt-4o \
    --top_n 20
```

**Key Scripts**:
- `run_llm_batch_evaluation.py`: Batch LLM evaluation runner
- `llm_text_evaluation.py`: LLM-based text generation evaluation

See [LLM_EVALUATION_GUIDE.md](LLM_EVALUATION_GUIDE.md) for detailed instructions.

---

## Key Directory Structure

```
steering_interventions/
├── data_curator.py                    # Step 1: Extract extreme songs
├── activation_extractor.py            # Step 2: Extract activations
├── steering_vector_computer.py        # Step 3: Compute steering vectors
├── dual_steering/
│   ├── dual_steering_experiment.py    # Phase 3: Dual steering experiment
│   ├── test_multi_conditioned.py      # Phase 4: Conditioned experiments
│   ├── vector_composition.py          # Gram-Schmidt orthogonalization
│   └── convert_npy_to_audio.py        # Convert tokens to audio
├── contrasting_interventions/
│   ├── find_contrasting_songs.py      # Find contrastive pairs
│   └── run_contrasting_interventions.py  # Run contrastive steering

deterministic_analysis/
├── batch_deterministic_analyzer.py    # Automated quality assessment
├── feature_specific_analyzers.py      # Pitch/duration/density analyzers
└── music_quality_assessor.py          # Quality degradation metrics

outputs/
├── steering_vectors/                  # Computed steering vectors
│   ├── pitch_steering_vectors.pt
│   └── average_duration_steering_vectors.pt
├── phase3_results/                    # Dual steering results
└── phase4_conditioned/                # Conditioned steering results
```

---

## Acknowledgment

The code is based largely on the [x-transformers](https://github.com/lucidrains/x-transformers) library developed by [lucidrains](https://github.com/lucidrains).

The activation steering extension was developed at:
- **Athens University of Economics and Business (AUEB)** - Information Systems Laboratory
- **Orfium** - AI Research
- **Hellenic Mediterranean University** - Music Technology and Acoustics Lab
- **NCSR Demokritos** - Institute of Informatics and Telecommunications
- **Archimedes/Athena Research Center**

---

## Citation

### Activation Steering Extension

If you use the activation steering framework, please cite:

```bibtex
@inproceedings{prokopiou2026activation,
  author    = {Prokopiou, Ioannis and Vikatos, Pantelis and Kaliakatsos-Papakostas, Maximos and Giannakopoulos, Theodoros and Stafylakis, Themos},
  title     = {Latent Space Disentanglement via Activation Steering for Interpretable Attribute Control in Symbolic Music Generation},
  booktitle = {European Signal Processing Conference (EUSIPCO)},
  year      = {2026}
}
```

### Original Multitrack Music Transformer

Please cite the following paper if you use the code provided in this repository.

 > Hao-Wen Dong, Ke Chen, Shlomo Dubnov, Julian McAuley, and Taylor Berg-Kirkpatrick, "Multitrack Music Transformer," _IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, 2023.

```bibtex
@inproceedings{dong2023mmt,
    author = {Hao-Wen Dong and Ke Chen and Shlomo Dubnov and Julian McAuley and Taylor Berg-Kirkpatrick},
    title = {Multitrack Music Transformer},
    booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year = 2023,
}
```
