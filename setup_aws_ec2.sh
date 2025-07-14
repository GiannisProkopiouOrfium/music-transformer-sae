#!/bin/bash

# AWS EC2 Setup Script for SAE Analysis
# This script sets up the environment on a fresh Ubuntu EC2 instance

set -e  # Exit on any error

echo "=== Setting up AWS EC2 for SAE Analysis ==="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and development tools
echo "Installing Python and development tools..."
sudo apt-get install -y python3 python3-pip python3-venv git htop nvtop

# Install CUDA (for GPU instances)
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA..."
    
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    
    # Install CUDA
    sudo apt-get install -y cuda-11-8
    
    # Add to PATH
    echo 'export PATH="/usr/local/cuda-11.8/bin:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
    
    echo "CUDA installation completed"
else
    echo "No NVIDIA GPU detected, skipping CUDA installation"
fi

# Create project directory
echo "Setting up project directory..."
mkdir -p ~/sae_analysis
cd ~/sae_analysis

# Clone or copy the mmt repository
echo "Setting up MMT repository..."
# Note: Replace with actual repository URL or copy files
# git clone <repository_url> .

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib seaborn
pip install h5py pyyaml
pip install jupyter notebook
pip install psutil

# Music-specific dependencies
pip install pretty_midi mido
pip install x-transformers

# Create directories
echo "Creating directory structure..."
mkdir -p experiments
mkdir -p data
mkdir -p logs

# Create startup script
echo "Creating startup script..."
cat > start_analysis.sh << 'EOF'
#!/bin/bash
cd ~/sae_analysis
source venv/bin/activate

echo "SAE Analysis Environment Ready!"
echo "GPU Status:"
nvidia-smi || echo "No GPU available"
echo ""
echo "To run analysis:"
echo "python -m pipeline.main --config configs/aws_ec2_config.yaml --experiment-name my_experiment"
EOF

chmod +x start_analysis.sh

# Create systemd service for auto-startup (optional)
echo "Creating system service..."
sudo tee /etc/systemd/system/sae-analysis.service > /dev/null << EOF
[Unit]
Description=SAE Analysis Service
After=network.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/sae_analysis
ExecStart=/home/ubuntu/sae_analysis/start_analysis.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# Enable service
sudo systemctl enable sae-analysis.service

# Set up log rotation
echo "Setting up log rotation..."
sudo tee /etc/logrotate.d/sae-analysis > /dev/null << EOF
/home/ubuntu/sae_analysis/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    copytruncate
}
EOF

# Create monitoring script
echo "Creating monitoring script..."
cat > monitor.sh << 'EOF'
#!/bin/bash
echo "=== System Status ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df -h / | awk 'NR==2 {print $5}')"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
fi

echo ""
echo "=== Active Processes ==="
ps aux | grep python | grep -v grep | head -5
EOF

chmod +x monitor.sh

# Set up SSH key for GitHub (if needed)
echo "Setting up SSH for GitHub..."
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -C "ubuntu@ec2" -f ~/.ssh/id_rsa -N ""
    echo "SSH key generated. Add the following public key to your GitHub account:"
    cat ~/.ssh/id_rsa.pub
fi

# Source bashrc to get CUDA in PATH
source ~/.bashrc

echo ""
echo "=== Setup Complete! ==="
echo "To get started:"
echo "1. Run: source ~/sae_analysis/venv/bin/activate"
echo "2. Start analysis: ./start_analysis.sh"
echo "3. Monitor system: ./monitor.sh"
echo ""
echo "Configuration files are in configs/"
echo "Experiments will be saved in experiments/"
echo "Logs will be in logs/"
echo ""

if lspci | grep -i nvidia > /dev/null; then
    echo "GPU detected. Verify CUDA installation:"
    echo "nvidia-smi"
    echo "python -c 'import torch; print(torch.cuda.is_available())'"
fi

echo "Setup completed successfully!"
