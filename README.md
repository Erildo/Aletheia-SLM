
# HyperFocus-SLM: Novel Small Language Model Architecture

A 5M parameter language model with dynamic compute routing and hybrid attention-SSM architecture.

## Architecture Features

- **Quantum Attention Router**: Dynamic token routing (25% deep path)
- **Grouped Query Attention**: 75% KV cache reduction
- **Simplified SSM**: Linear complexity long-range modeling
- **Hybrid Processing**: Parallel attention + state space models

## Quick Start

### Development (Codespaces - No GPU needed)
```bash
# Setup
source venv/bin/activate
pip install -r requirements.txt

# Test architecture
python -c "from src.model import HyperFocusLM; print('✓ Import successful')"

# Run notebook
jupyter notebook notebooks/prototype.ipynb
