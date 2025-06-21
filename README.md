# Cortex-Synth: Differentiable Topology-Aware 3D Skeleton Synthesis

This repository contains the PyTorch implementation for the paper "Cortex-Synth: Differentiable Topology-Aware 3D Skeleton Synthesis". Our framework introduces a novel, end-to-end differentiable approach for jointly synthesizing 3D skeleton geometry and topology from a single 2D image.

## Key Features

- **Differentiable Graph Construction**: Utilizes a Sinkhorn-based algorithm for continuous relaxation of connectivity prediction, enabling end-to-end training [1].
- **Theoretical Guarantees**: Implements a discrete gradient flow formulation over the Laplacian eigenspace with proven convergence properties [1].
- **Large-Scale Domain Adaptation**: Bridges the synthetic-to-real domain gap using adversarial training and pseudo-label refinement [1].
- **State-of-the-Art Performance**: Achieves significant improvements on benchmarks like COCO-Pose and Pascal-3D+ [1].

## Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/Zayaan3019/Cortex.git
    cd Cortex-Synth
    ```

2.  **Create a conda environment and install dependencies:**
    ```
    conda create -n cortex python=3.9
    conda activate cortex
    pip install -r requirements.txt
    ```

## Training

To train the Cortex-Synth model, run the main training script. The configuration, including hyperparameters and dataset paths, is managed through `configs/default.yaml`.

