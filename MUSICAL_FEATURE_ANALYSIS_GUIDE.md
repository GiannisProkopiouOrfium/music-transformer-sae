"""
Musical Feature Analysis with LLM Integration for SAE Interpretability
====================================================================

This guide explains how the SAE pipeline tracks input-output correlations and provides
a framework for using LLMs to understand musical features from MIDI files.

## 1. How Input-Output Correlation Works

### Input Association Tracking
The SAE pipeline maintains perfect correlation between inputs and SAE features through several mechanisms:

#### A. During Activation Extraction:
```python
# From extract_from_dataloader() in sae_pipeline.py

for batch_idx, batch in enumerate(dataloader):
    inputs = batch["input_ids"].to(device)
    _ = model(inputs)  # Forward pass triggers hook
    
    # Store input associations for each sample
    batch_size = inputs.shape[0]
    for i in range(batch_size):
        self.input_tokens.append(inputs[i].cpu().numpy())
        self.sequence_ids.append(seq_id)
        seq_id += 1

# Metadata stores complete mapping
metadata = {
    "input_associations": [
        {"sequence_id": sid, "input_tokens": tokens.tolist()}
        for sid, tokens in zip(self.sequence_ids, self.input_tokens)
    ]
}
```

#### B. Activation Hook Mechanism:
```python
def hook_fn(module, input, output):
    # Capture layer activations while preserving batch order
    activations = output.mean(dim=1)  # [batch, hidden]
    self.activations.append(activations.detach().cpu())

# Hook registers on target transformer layer
handle = layers[target_layer].register_forward_hook(hook_fn)
```

#### C. Data Storage Format:
The pipeline stores correlations in HDF5 format:
```
activations.h5:
├── activations/          # Shape: [num_samples, hidden_size]
├── metadata/
│   ├── input_associations/
│   │   ├── sequence_0/   # input_tokens: [seq_len] token IDs
│   │   ├── sequence_1/   # input_tokens: [seq_len] token IDs
│   │   └── ...
│   ├── layer_idx: 3
│   └── hidden_size: 512
```

### Feature-Input Mapping:
After SAE training, each feature activation can be traced back to specific input sequences:
```python
# For feature F activated in sample S:
sample_index = S
feature_value = sae_features[sample_index, F]
original_tokens = metadata["input_associations"][sample_index]["input_tokens"]
# Now you know which MIDI tokens caused feature F to activate
```

## 2. Using LLMs for Musical Feature Analysis

### Setting Up LLM Integration

#### Step 1: Prepare Musical Context Data
```python
import json
from pathlib import Path

def extract_musical_context(json_path):
    """Extract interpretable musical information from MMM-encoded JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    context = {
        "title": data["metadata"]["title"],
        "tempo": data["tempos"],
        "key_signature": data["key_signatures"],
        "time_signature": data["time_signatures"],
        "tracks": []
    }
    
    # Extract track information
    for track in data["tracks"]:
        track_info = {
            "instrument": track["program"],
            "name": track.get("name", "Unknown"),
            "notes": track["notes"][:50],  # First 50 notes for analysis
            "note_range": {
                "min_pitch": min(note["pitch"] for note in track["notes"]),
                "max_pitch": max(note["pitch"] for note in track["notes"]),
            },
            "rhythmic_patterns": extract_rhythm_patterns(track["notes"]),
            "harmonic_intervals": extract_intervals(track["notes"])
        }
        context["tracks"].append(track_info)
    
    return context

def extract_rhythm_patterns(notes):
    """Extract rhythmic patterns from note sequence."""
    durations = [note["duration"] for note in notes]
    return {
        "avg_duration": sum(durations) / len(durations),
        "duration_variety": len(set(durations)),
        "common_durations": max(set(durations), key=durations.count)
    }

def extract_intervals(notes):
    """Extract harmonic intervals between consecutive notes."""
    if len(notes) < 2:
        return {"intervals": []}
    
    intervals = []
    for i in range(len(notes) - 1):
        interval = notes[i+1]["pitch"] - notes[i]["pitch"]
        intervals.append(interval)
    
    return {
        "intervals": intervals[:20],  # First 20 intervals
        "avg_interval": sum(intervals) / len(intervals),
        "interval_variety": len(set(intervals))
    }
```

#### Step 2: Create Feature-Music Correlation Analyzer
```python
def analyze_feature_music_correlation(sae_features, activations_metadata, music_contexts):
    """
    Correlate SAE features with musical concepts using extracted contexts.
    
    Args:
        sae_features: [num_samples, num_features] SAE feature activations
        activations_metadata: Metadata with input associations
        music_contexts: List of musical contexts for each sample
    """
    feature_correlations = {}
    
    for feature_idx in range(sae_features.shape[1]):
        feature_activations = sae_features[:, feature_idx]
        
        # Find samples where this feature is highly active
        high_activation_threshold = np.percentile(feature_activations, 90)
        active_samples = np.where(feature_activations > high_activation_threshold)[0]
        
        # Analyze musical characteristics of highly activating samples
        musical_patterns = analyze_musical_patterns_for_feature(
            active_samples, music_contexts, activations_metadata
        )
        
        feature_correlations[feature_idx] = {
            "activation_strength": float(np.mean(feature_activations[active_samples])),
            "activation_frequency": len(active_samples) / len(feature_activations),
            "musical_patterns": musical_patterns
        }
    
    return feature_correlations

def analyze_musical_patterns_for_feature(active_samples, music_contexts, metadata):
    """Analyze common musical patterns in samples that activate a specific feature."""
    patterns = {
        "instruments": [],
        "pitch_ranges": [],
        "rhythmic_characteristics": [],
        "harmonic_intervals": [],
        "tempos": [],
        "keys": []
    }
    
    for sample_idx in active_samples:
        # Get the original input tokens for this sample
        tokens = metadata["input_associations"][sample_idx]["input_tokens"]
        
        # Get corresponding musical context (you'll need to match this)
        # This requires mapping between your sequence IDs and original JSON files
        context = music_contexts[sample_idx]
        
        # Extract patterns
        for track in context["tracks"]:
            patterns["instruments"].append(track["instrument"])
            patterns["pitch_ranges"].append(track["note_range"])
            patterns["rhythmic_characteristics"].append(track["rhythmic_patterns"])
            patterns["harmonic_intervals"].extend(track["harmonic_intervals"]["intervals"])
        
        patterns["tempos"].extend([t["qpm"] for t in context["tempo"]])
        patterns["keys"].extend([k["root"] for k in context["key_signature"]])
    
    # Summarize patterns
    summary = {
        "common_instruments": most_common(patterns["instruments"]),
        "typical_pitch_range": {
            "min": np.mean([r["min_pitch"] for r in patterns["pitch_ranges"]]),
            "max": np.mean([r["max_pitch"] for r in patterns["pitch_ranges"]])
        },
        "common_intervals": most_common(patterns["harmonic_intervals"]),
        "typical_tempo": np.mean(patterns["tempos"]) if patterns["tempos"] else None,
        "common_keys": most_common(patterns["keys"])
    }
    
    return summary

def most_common(items):
    """Find most common items in a list."""
    if not items:
        return None
    return max(set(items), key=items.count)
```

#### Step 3: Generate LLM Prompts for Feature Interpretation
```python
def generate_feature_interpretation_prompt(feature_idx, correlation_data):
    """Generate a detailed prompt for LLM analysis of a specific SAE feature."""
    patterns = correlation_data[feature_idx]["musical_patterns"]
    
    prompt = f'''
# Musical Feature Analysis Request

## Feature #{feature_idx} Characteristics
- **Activation Strength**: {correlation_data[feature_idx]["activation_strength"]:.3f}
- **Activation Frequency**: {correlation_data[feature_idx]["activation_frequency"]:.1%}

## Musical Context Where This Feature Activates Strongly

### Instrumental Context:
- **Most Common Instruments**: {patterns["common_instruments"]}
- **Typical Pitch Range**: {patterns["typical_pitch_range"]["min"]:.0f} - {patterns["typical_pitch_range"]["max"]:.0f} MIDI

### Harmonic Context:
- **Common Intervals**: {patterns["common_intervals"]} semitones
- **Typical Keys**: {patterns["common_keys"]}

### Rhythmic Context:
- **Typical Tempo**: {patterns["typical_tempo"]:.1f} BPM

## Analysis Request

Based on this musical context, please provide:

1. **Musical Concept Identification**: What specific musical concept or pattern might this feature represent? Consider:
   - Harmonic functions (tonic, dominant, subdominant)
   - Melodic patterns (scales, arpeggios, sequences)
   - Rhythmic patterns (syncopation, specific note values)
   - Textural elements (counterpoint, homophony)
   - Instrumental techniques (specific playing styles)

2. **Theoretical Explanation**: Explain why this combination of musical characteristics might form a coherent concept that a neural network would learn to detect.

3. **Examples**: Provide specific examples of musical passages or compositions where you might expect to find this pattern.

4. **Feature Name Suggestion**: Suggest a descriptive name for this feature based on the musical concept it appears to represent.

Please be specific and reference music theory concepts where appropriate.
'''
    
    return prompt

# Example usage for generating prompts for top features
def analyze_top_features_with_llm(feature_correlations, top_k=10):
    """Generate LLM prompts for the most interesting features."""
    
    # Sort features by activation strength or frequency
    sorted_features = sorted(
        feature_correlations.items(),
        key=lambda x: x[1]["activation_strength"] * x[1]["activation_frequency"],
        reverse=True
    )
    
    prompts = {}
    for feature_idx, correlation_data in sorted_features[:top_k]:
        prompt = generate_feature_interpretation_prompt(
            feature_idx, 
            {feature_idx: correlation_data}
        )
        prompts[feature_idx] = prompt
        
        print(f"\n{'='*60}")
        print(f"PROMPT FOR FEATURE {feature_idx}")
        print('='*60)
        print(prompt)
        print(f"\n{'='*60}\n")
    
    return prompts
```

### Step 4: Complete Integration Example

```python
def complete_musical_sae_analysis(results_dir, music_json_dir):
    """
    Complete pipeline for SAE feature analysis with musical context.
    
    Args:
        results_dir: Path to SAE experiment results
        music_json_dir: Path to processed JSON MIDI files
    """
    
    # Load SAE results
    with h5py.File(results_dir / "activations.h5", 'r') as f:
        activations = f["activations"][:]
        metadata = {
            "input_associations": [
                {"sequence_id": i, "input_tokens": f[f"metadata/input_associations/sequence_{i}"][:]}
                for i in range(len(activations))
            ]
        }
    
    with h5py.File(results_dir / "analysis_results.h5", 'r') as f:
        sae_features = f["sae_features"][:]
    
    # Extract musical contexts from JSON files
    json_files = list(Path(music_json_dir).glob("**/*.json"))
    music_contexts = []
    
    for i, json_file in enumerate(json_files[:len(activations)]):
        context = extract_musical_context(json_file)
        music_contexts.append(context)
    
    # Correlate SAE features with musical patterns
    feature_correlations = analyze_feature_music_correlation(
        sae_features, metadata, music_contexts
    )
    
    # Generate LLM prompts for top features
    llm_prompts = analyze_top_features_with_llm(feature_correlations, top_k=15)
    
    # Save results
    output_file = results_dir / "musical_feature_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            "feature_correlations": feature_correlations,
            "llm_prompts": llm_prompts
        }, f, indent=2)
    
    print(f"Analysis complete! Results saved to {output_file}")
    print(f"Found {len(feature_correlations)} features with musical correlations")
    print(f"Generated {len(llm_prompts)} LLM prompts for top features")
    
    return feature_correlations, llm_prompts

# Run the complete analysis
if __name__ == "__main__":
    results_dir = Path("results/layer3_experiment")
    music_json_dir = Path("data/sod/processed/json")
    
    correlations, prompts = complete_musical_sae_analysis(results_dir, music_json_dir)
```

## 3. Using the Analysis Results

### With OpenAI API:
```python
import openai

def analyze_feature_with_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a music theory expert analyzing neural network features."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

# Analyze each feature
for feature_idx, prompt in prompts.items():
    interpretation = analyze_feature_with_openai(prompt)
    print(f"Feature {feature_idx} Analysis:")
    print(interpretation)
    print("-" * 80)
```

### With Local LLM (like Ollama):
```python
import requests

def analyze_feature_with_ollama(prompt, model="llama2"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]
```

This framework provides a complete pipeline for understanding what musical concepts your SAE features have learned by correlating them with the original MIDI content and using LLMs to interpret the patterns.
"""
