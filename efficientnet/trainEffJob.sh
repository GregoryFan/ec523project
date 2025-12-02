#!/bin/bash -l

# Project account
#$ -P ec523bn

# Job name
#$ -N efficientnet_train

# Request 1 GPU with at least compute capability 6.0
#$ -l gpus=1
#$ -l gpu_c=6.0

# Request 4 CPU cores
#$ -pe omp 4

# Runtime and memory
#$ -l h_rt=48:00:00

# Output files
#$ -o train_$JOB_ID.out
#$ -e train_$JOB_ID.err

# Merge error and output streams
#$ -j y

# Load modules and activate environment
module load python3/3.10.5
module load cuda/11.8

source ~/envs/torch_env310/bin/activate
export CUDA_VISIBLE_DEVICES=0
# Change to working directory
cd /projectnb/ec523bn/students/jchen07/ec523project/efficientnet

# Print job info
echo "========================================="
echo "Starting EfficientNet-B0 training job"
echo "Job ID: $JOB_ID"
echo "Running on host: $(hostname)"
echo "Date: $(date)"
echo "Assigned GPU(s): $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""

# Verify CUDA/PyTorch setup
echo "Checking PyTorch and CUDA compatibility..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'PyTorch CUDA version: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Check GPU availability with nvidia-smi
echo "GPU status from nvidia-smi:"
nvidia-smi
echo ""

# Run training (PyTorch will automatically use CUDA_VISIBLE_DEVICES)
echo "Starting training..."
python3 trainEff.py

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "========================================="