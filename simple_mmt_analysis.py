#!/usr/bin/env python3
"""
Simple MMT Deterministic Analysis

A simplified version that directly analyzes your .pt intervention files
without the complex package dependencies.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import traceback

# Add MMT modules to path
current_dir = Path(__file__).parent
mmt_dir = current_dir / "mmt"
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(mmt_dir))

try:
    import torch
    from mmt import representation
    import muspy
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class SimpleMMTAnalyzer:
    """Simple analyzer for MMT intervention results."""
    
    def __init__(self, output_dir: str = "simple_analysis_results"):
        """Initialize analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load encoding
        encoding_path = "mmt/encoding.json"
        if not Path(encoding_path).exists():
            raise FileNotFoundError(f"Could not find encoding at {encoding_path}")
        
        self.encoding = representation.load_encoding(encoding_path)
        print(f"✅ Loaded encoding from: {encoding_path}")
        
        # Results storage
        self.results = {}
    
    def load_pt_file(self, pt_file: Path):
        """Load and convert a .pt file to MusPy music."""
        try:
            # Load tensor data
            tensor_data = torch.load(str(pt_file), map_location='cpu')
            
            # Extract sequence
            if 'generated' in tensor_data:
                sequence = tensor_data['generated']
            else:
                print(f"Warning: No 'generated' key in {pt_file.name}")
                return None, None
            
            # Convert to numpy
            if isinstance(sequence, torch.Tensor):
                seq_np = sequence.numpy()
            else:
                seq_np = sequence
            
            # Remove batch dimension if present
            if len(seq_np.shape) == 3 and seq_np.shape[0] == 1:
                seq_np = seq_np[0]
            
            # Convert to MusPy Music
            music = representation.decode(seq_np, self.encoding)
            
            # Extract metadata
            metadata = {
                'condition_name': tensor_data.get('condition_name', 'unknown'),
                'intervention_type': tensor_data.get('intervention_type', 'unknown'),
                'strength': tensor_data.get('strength', 0.0),
                'feature_id': tensor_data.get('feature_id', 'unknown'),
                'layer': tensor_data.get('layer', 0),
                'original_file': str(pt_file)
            }
            
            return music, metadata
            
        except Exception as e:
            print(f"Error loading {pt_file.name}: {e}")
            return None, None
    
    def extract_basic_features(self, music):
        """Extract basic musical features from MusPy music object."""
        features = {}
        
        try:
            # Basic info
            features['duration'] = music.get_end_time()
            features['total_notes'] = len([note for track in music.tracks for note in track.notes])
            features['num_tracks'] = len(music.tracks)
            
            # Get all notes
            all_notes = []
            for track in music.tracks:
                all_notes.extend(track.notes)
            
            if all_notes:
                # Pitch analysis
                pitches = [note.pitch for note in all_notes]
                features['min_pitch'] = min(pitches)
                features['max_pitch'] = max(pitches)
                features['pitch_range'] = max(pitches) - min(pitches)
                features['unique_pitches'] = len(set(pitches))
                features['mean_pitch'] = sum(pitches) / len(pitches)
                
                # Velocity analysis
                velocities = [note.velocity for note in all_notes]
                features['min_velocity'] = min(velocities)
                features['max_velocity'] = max(velocities)
                features['velocity_range'] = max(velocities) - min(velocities)
                features['mean_velocity'] = sum(velocities) / len(velocities)
                
                # Duration analysis
                durations = [note.duration for note in all_notes]
                features['mean_note_duration'] = sum(durations) / len(durations)
                features['min_note_duration'] = min(durations)
                features['max_note_duration'] = max(durations)
                
                # Timing analysis
                start_times = [note.start for note in all_notes]
                features['first_note_time'] = min(start_times)
                features['last_note_time'] = max(start_times)
                
                # Calculate note density (notes per second)
                if features['duration'] > 0:
                    features['note_density'] = features['total_notes'] / features['duration']
                else:
                    features['note_density'] = 0
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            features['extraction_error'] = str(e)
        
        return features
    
    def analyze_single_intervention(self, pt_file: Path):
        """Analyze a single intervention file."""
        print(f"Analyzing: {pt_file.name}")
        
        # Load music and metadata
        music, metadata = self.load_pt_file(pt_file)
        
        if music is None:
            return None
        
        # Extract features
        features = self.extract_basic_features(music)
        
        # Combine with metadata
        analysis = {
            'metadata': metadata,
            'features': features,
            'file_info': {
                'file_path': str(pt_file),
                'file_size': pt_file.stat().st_size,
                'analyzed_at': datetime.now().isoformat()
            }
        }
        
        return analysis
    
    def find_intervention_files(self, interventions_dir: str):
        """Find all .pt intervention files."""
        interventions_path = Path(interventions_dir)
        
        if not interventions_path.exists():
            print(f"❌ Directory not found: {interventions_dir}")
            return {}
        
        # Find all .pt files
        pt_files = list(interventions_path.rglob("*.pt"))
        
        if not pt_files:
            print("❌ No .pt files found!")
            return {}
        
        print(f"✅ Found {len(pt_files)} .pt files")
        
        # Group by feature
        grouped_files = {}
        
        for pt_file in pt_files:
            # Try to extract feature info from filename
            filename = pt_file.name
            
            # Look for feature pattern in filename
            feature_id = "unknown"
            for part in filename.split('_'):
                if part.startswith('feature') and len(part) > 7:
                    feature_id = part.replace('feature', '')
                    break
            
            if feature_id not in grouped_files:
                grouped_files[feature_id] = []
            
            grouped_files[feature_id].append(pt_file)
        
        return grouped_files
    
    def run_analysis(self, interventions_dir: str):
        """Run analysis on all intervention files."""
        print("🎵 Starting Simple MMT Analysis")
        print("=" * 50)
        print(f"Input directory: {interventions_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Find files
        grouped_files = self.find_intervention_files(interventions_dir)
        
        if not grouped_files:
            print("❌ No files to analyze")
            return None
        
        # Analyze each file
        all_analyses = {}
        successful = 0
        failed = 0
        
        for feature_id, files in grouped_files.items():
            print(f"\n📊 Analyzing Feature {feature_id} ({len(files)} files)")
            
            feature_analyses = {}
            
            for pt_file in files:
                try:
                    analysis = self.analyze_single_intervention(pt_file)
                    if analysis:
                        key = pt_file.stem  # filename without extension
                        feature_analyses[key] = analysis
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    print(f"❌ Error analyzing {pt_file.name}: {e}")
                    failed += 1
            
            all_analyses[feature_id] = feature_analyses
        
        # Generate summary
        summary = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(interventions_dir),
                'output_directory': str(self.output_dir),
                'total_files_found': successful + failed,
                'successful_analyses': successful,
                'failed_analyses': failed
            },
            'features_analyzed': list(grouped_files.keys()),
            'detailed_results': all_analyses
        }
        
        # Save results
        results_file = self.output_dir / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create summary report
        self.create_summary_report(summary)
        
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Results: {successful} successful, {failed} failed")
        print(f"💾 Saved to: {results_file}")
        
        return summary
    
    def create_summary_report(self, summary):
        """Create a human-readable summary report."""
        report_file = self.output_dir / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# MMT Intervention Analysis Summary\n\n")
            
            # Overview
            info = summary['analysis_info']
            f.write("## Overview\n\n")
            f.write(f"- **Analysis Date**: {info['timestamp']}\n")
            f.write(f"- **Input Directory**: {info['input_directory']}\n")
            f.write(f"- **Total Files**: {info['total_files_found']}\n")
            f.write(f"- **Successful**: {info['successful_analyses']}\n")
            f.write(f"- **Failed**: {info['failed_analyses']}\n")
            f.write(f"- **Success Rate**: {info['successful_analyses']/info['total_files_found']*100:.1f}%\n\n")
            
            # Features analyzed
            f.write("## Features Analyzed\n\n")
            for feature_id in summary['features_analyzed']:
                feature_data = summary['detailed_results'][feature_id]
                f.write(f"### Feature {feature_id}\n")
                f.write(f"- **Files analyzed**: {len(feature_data)}\n")
                
                if feature_data:
                    # Get basic stats
                    sample_analysis = next(iter(feature_data.values()))
                    if 'features' in sample_analysis:
                        features = sample_analysis['features']
                        f.write(f"- **Sample duration**: {features.get('duration', 'N/A'):.2f}s\n")
                        f.write(f"- **Sample notes**: {features.get('total_notes', 'N/A')}\n")
                        f.write(f"- **Sample pitch range**: {features.get('pitch_range', 'N/A')}\n")
                
                f.write("\n**Conditions:**\n")
                for condition_name, analysis in feature_data.items():
                    condition = analysis['metadata']['condition_name']
                    intervention = analysis['metadata']['intervention_type']
                    strength = analysis['metadata']['strength']
                    f.write(f"- {condition_name}: {intervention} (strength: {strength})\n")
                f.write("\n")
        
        print(f"📄 Summary report: {report_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple MMT Intervention Analysis")
    parser.add_argument("interventions_dir", help="Directory containing intervention .pt files")
    parser.add_argument("--output-dir", default="simple_analysis_results", 
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    try:
        # Create analyzer and run analysis
        analyzer = SimpleMMTAnalyzer(output_dir=args.output_dir)
        results = analyzer.run_analysis(args.interventions_dir)
        
        if results:
            print("\n🎉 Analysis completed successfully!")
            print(f"Check {args.output_dir}/ for detailed results")
        else:
            print("\n❌ Analysis failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())