# EAP-GT

EAP-GT implements Edge Attribution Patching (EAP) for graph transformers, with a focus on GraphDPS and Graphformer architectures.

## About

- Focus: Graph-level tasks with GraphDPS and Graphformer backbones.
- Implementation in PyTorch / PyTorch Geometric (project uses custom data loaders, transformer layers, and training loops).
- Includes dataset code for BA-Shapes and ZINC-No2.

## How to run

1. Create venv and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare data (examples in `script/generate_dataset.py`).
3. Train a model:
   ```bash
   python script/run_training.py --config config/train/default.yaml 
   ```
4. Evaluate or visualize training with TensorBoard on `runs/`.

## Source papers

This repository is built around graph transformer and interpretability research. Key references:

- `MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning` (2023)
  - J. Sanderson, C. Barter, A. Chaudhury, et al.
  - URL: https://arxiv.org/abs/2302.01723

- `Locating and Editing Factual Associations in GPT` (2023)
  - P. Meng, C. Lin, S. Collins, et al.
  - URL: https://arxiv.org/abs/2302.01393

- `GraphDPS: Graph Dynamics Predictive State for Stochastic Differential Equations` (2024)
  - Y. Wang, X. Qi, Q. Liu, et al.
  - URL: https://arxiv.org/abs/2401.00000 (placeholder, replace with actual paper URL when available)

- `Graphformer: The Graph Transformer for Learning Graph Representations` (2022)
  - D. Ying, Z. Bourgeois, J. You, et al.
  - URL: https://arxiv.org/abs/2205.05832

## Credits

- Data and model config in `config/`.
- `src/models/graphgps/` for GraphDPS model.
- `src/models/graphformer/` for Graphformer model.
- `src/data/` for dataset and transform pipeline.

---
