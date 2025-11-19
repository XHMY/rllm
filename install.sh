
git clone --recurse-submodules https://github.com/XHMY/rllm.git
cd rllm

# Create a conda environment
conda create -n rllm python=3.10 -y
conda activate rllm

# Install verl
bash scripts/install_verl.sh

# Install rLLM
pip install -e .

# Apply multi-agent patch to verl 0.5.0
git -C ./verl apply ./multi_agent_rllm_verl.patch