#!/bin/bash
# =============================================================================
# EC2 Setup Script for Music Flamingo Local Evaluation
# =============================================================================
#
# Recommended EC2 instance:
#   - g5.xlarge  (1x A10G 24GB, ~$1.00/hr on-demand, ~$0.33/hr spot)
#   - p3.2xlarge (1x V100 16GB, works but tight — may need fp16 offload)
#
# AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
#   or: any Ubuntu 22.04 + NVIDIA drivers + CUDA 12.x
#
# Storage: 50 GB gp3 (model weights ~16GB + deps + audio)
#
# Usage:
#   1. Launch EC2 instance with GPU AMI
#   2. scp this script + your audio files to the instance
#   3. Run: bash setup_ec2.sh
#   4. Run: bash run_flamingo.sh  (created by this script)
#
# After completion, scp the results back:
#   scp -r ec2-user@<ip>:~/mmt/flamingo_evaluation/results/ ./flamingo_evaluation/results/
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "  Music Flamingo EC2 Setup"
echo "=========================================="

# --- System packages ---
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg git python3-pip python3-venv

# --- Create virtual environment ---
cd ~
python3 -m venv flamingo_env
source flamingo_env/bin/activate

# --- Install custom transformers fork (required for Music Flamingo) ---
echo "Installing custom Transformers fork..."
pip install --upgrade pip
pip install --upgrade "git+https://github.com/lashahub/transformers@modular-mf"
pip install accelerate torch openai

# --- Clone / copy project ---
# If the project isn't already here, the user should scp it.
# Minimal files needed:
#   flamingo_evaluation/  (the whole package)
#   AUDIO EVAL TRIM/ALL/  (75 MP3 files)
if [ ! -d "mmt/flamingo_evaluation" ]; then
    echo ""
    echo "⚠️  Project not found at ~/mmt/"
    echo "   Please scp your project files first:"
    echo ""
    echo "   # From your local machine:"
    echo "   scp -r flamingo_evaluation/ ec2-user@<EC2_IP>:~/mmt/flamingo_evaluation/"
    echo "   scp -r 'AUDIO EVAL TRIM/ALL/' ec2-user@<EC2_IP>:~/mmt/'AUDIO EVAL TRIM/ALL/'"
    echo ""
    exit 1
fi

cd ~/mmt

# --- Create convenience run script ---
cat > run_flamingo.sh << 'RUNEOF'
#!/bin/bash
set -euo pipefail
source ~/flamingo_env/bin/activate
cd ~/mmt

echo "=========================================="
echo "  Step 1: Music Flamingo Descriptions"
echo "  (local GPU — no quota limits)"
echo "=========================================="
python -m flamingo_evaluation.run_all --step describe --local

echo ""
echo "=========================================="
echo "  Step 2: GPT-4o Comparison"
echo "  (requires OPENAI_API_KEY)"
echo "=========================================="
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "⚠️  Set OPENAI_API_KEY first:"
    echo "   export OPENAI_API_KEY='sk-...'"
    echo "   Then re-run: bash run_flamingo.sh"
    exit 1
fi
python -m flamingo_evaluation.run_all --step compare --model gpt-4o

echo ""
echo "✅ Done! Results in flamingo_evaluation/results/"
echo "   scp them back to your local machine:"
echo "   scp -r ec2-user@\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):~/mmt/flamingo_evaluation/results/ ./flamingo_evaluation/results/"
RUNEOF

chmod +x run_flamingo.sh

echo ""
echo "=========================================="
echo "  ✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. export OPENAI_API_KEY='sk-...'"
echo "  2. bash run_flamingo.sh"
echo ""
echo "Or run steps separately:"
echo "  source ~/flamingo_env/bin/activate"
echo "  cd ~/mmt"
echo "  python -m flamingo_evaluation.run_all --step describe --local"
echo "  python -m flamingo_evaluation.run_all --step compare --model gpt-4o"
