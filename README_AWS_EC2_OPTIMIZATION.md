# SAE Analysis AWS EC2 Optimization - Complete Implementation

## Overview
This document provides comprehensive instructions for running SAE (Sparse Autoencoder) analysis on AWS EC2 instances with GPU optimization and automated pipeline management.

## Quick Start

### 1. AWS EC2 Instance Setup

**Recommended Instances:**
- **Small**: g4dn.xlarge (4 vCPUs, 16GB RAM, T4 GPU) - $0.526/hour
- **Medium**: g4dn.2xlarge (8 vCPUs, 32GB RAM, T4 GPU) - $0.752/hour  
- **Large**: p3.2xlarge (8 vCPUs, 61GB RAM, V100 GPU) - $3.06/hour

**Setup Commands:**
```bash
# Copy the setup script to your EC2 instance
scp -i your-key.pem setup_aws_ec2.sh ubuntu@your-instance-ip:~/

# SSH into instance and run setup
ssh -i your-key.pem ubuntu@your-instance-ip
chmod +x setup_aws_ec2.sh
./setup_aws_ec2.sh
```

### 2. Running the Analysis Pipeline

```bash
# Activate environment
source ~/sae_analysis/venv/bin/activate

# Run complete pipeline
python -m pipeline.main \
  --config configs/aws_ec2_config.yaml \
  --experiment-name music_sae_experiment_$(date +%Y%m%d_%H%M%S) \
  --stage all \
  --output-dir ./experiments

# Or run individual stages:
python -m pipeline.main --config configs/aws_ec2_config.yaml --stage extract --experiment-name my_exp
python -m pipeline.main --config configs/aws_ec2_config.yaml --stage train --experiment-name my_exp --resume
python -m pipeline.main --config configs/aws_ec2_config.yaml --stage analyze --experiment-name my_exp --resume
```

### 3. Monitoring System Performance

```bash
# System monitoring
./monitor.sh

# GPU memory profiling
python gpu_profiler.py --test-allocation --max-memory-gb 8

# View logs
tail -f logs/pipeline.log
```

## Configuration Files

### AWS EC2 Configuration (`configs/aws_ec2_config.yaml`)
Optimized for cloud instances with GPU support and large-scale processing.

### Local Configuration (`configs/local_config.yaml`)
For development and testing on local machines with limited resources.

### Production Configuration (`configs/production_config.yaml`)
For production deployments with maximum performance optimization.

## File Structure

```
mmt/
├── utils/                      # Optimization utilities
│   ├── gpu.py                 # GPU management and optimization
│   ├── memory.py              # Memory optimization 
│   └── storage.py             # HDF5 storage optimization
├── pipeline/                   # Automated pipeline
│   ├── main.py                # Main pipeline orchestrator
│   ├── extract.py             # Activation extraction
│   ├── train.py               # SAE training
│   └── analyze.py             # Feature analysis
├── configs/                    # Configuration files
│   ├── aws_ec2_config.yaml    # AWS EC2 settings
│   ├── local_config.yaml      # Local development
│   └── production_config.yaml # Production settings
├── setup_aws_ec2.sh           # EC2 setup script
└── gpu_profiler.py            # Memory profiling tool
```

## Key Features

### 1. GPU Optimization
- **Automatic device detection**: Detects and configures optimal GPU/CPU usage
- **Memory management**: Optimizes batch sizes based on available GPU memory
- **Mixed precision training**: Uses automatic mixed precision when available
- **Memory monitoring**: Real-time GPU memory usage tracking

### 2. Automated Pipeline
- **Stage-based execution**: Run extraction, training, and analysis independently
- **Resume capability**: Continue from where you left off
- **Configuration-driven**: Easy to modify parameters without code changes
- **Comprehensive logging**: Detailed logs for debugging and monitoring

### 3. Storage Optimization
- **HDF5 compression**: Efficient storage with gzip compression
- **Chunked storage**: Optimized chunk sizes for different data patterns
- **Memory-mapped access**: Efficient data loading for large datasets

### 4. Instance-Specific Settings
- **Dynamic batch sizing**: Automatically adjusts based on available resources
- **Memory profiling**: Identifies optimal settings for your specific instance
- **Scalable configurations**: Easily switch between development and production settings

## Usage Examples

### Example 1: Full Pipeline on g4dn.xlarge
```bash
python -m pipeline.main \
  --config configs/aws_ec2_config.yaml \
  --experiment-name sae_layer2_analysis \
  --stage all \
  --output-dir ./experiments
```

### Example 2: Large-Scale Production Run
```bash
python -m pipeline.main \
  --config configs/production_config.yaml \
  --experiment-name production_sae_full \
  --stage all \
  --output-dir /data/experiments
```

### Example 3: Development Testing
```bash
python -m pipeline.main \
  --config configs/local_config.yaml \
  --experiment-name dev_test \
  --stage all \
  --output-dir ./dev_experiments
```

## Performance Expectations

### g4dn.xlarge (T4 GPU)
- **Extraction**: ~50,000 samples in 30 minutes
- **Training**: 100 epochs in 2-3 hours  
- **Analysis**: 10 minutes for comprehensive report

### p3.2xlarge (V100 GPU)
- **Extraction**: ~200,000 samples in 45 minutes
- **Training**: 200 epochs in 3-4 hours
- **Analysis**: 15 minutes for detailed analysis

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Check memory usage
   ./monitor.sh
   
   # Profile memory allocation
   python gpu_profiler.py --test-allocation
   
   # Reduce batch size in config file
   ```

2. **CUDA Not Available**
   ```bash
   # Verify CUDA installation
   nvidia-smi
   nvcc --version
   
   # Test PyTorch CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Data Loading Issues**
   ```bash
   # Check data paths in config
   ls -la data/sod/processed/
   
   # Verify model checkpoint
   ls -la exp/sod/mmm/model.pt
   ```

## Cost Optimization

### Instance Selection
- **Development**: g4dn.xlarge for basic testing
- **Production**: g4dn.2xlarge for balanced performance/cost
- **Research**: p3.2xlarge for maximum performance

### Cost Monitoring
```bash
# Estimate costs based on runtime
echo "Instance: g4dn.xlarge"
echo "Rate: $0.526/hour" 
echo "Expected runtime: 4 hours"
echo "Estimated cost: $2.10"
```

### Spot Instances
Consider using EC2 Spot Instances for 60-90% cost savings on non-urgent workloads.

## Next Steps

1. **Run the setup script** on your AWS EC2 instance
2. **Copy your trained model** and data to the instance
3. **Start with local config** for initial testing
4. **Scale to AWS config** for full analysis
5. **Monitor performance** and adjust configurations as needed

The optimized pipeline provides a production-ready solution for large-scale SAE analysis with comprehensive GPU support, automated workflows, and efficient resource utilization.
