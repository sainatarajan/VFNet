
# VFNet - Variational FoldingNet for Dental Point Clouds

**VFNet** (Variational FoldingNet) is a fully probabilistic variational autoencoder designed for point cloud processing, particularly suited for dental reconstruction tasks. The model was introduced in the paper [Variational Autoencoding of Dental Point Clouds](https://arxiv.org/abs/2307.10895), where they tackle key challenges in digital dentistry by utilizing probabilistic methods to improve point cloud generation and shape completion.

## Overview

In digital dentistry, advancements have been made in 3D reconstruction, yet challenges remain, particularly in handling dental point clouds. The VFNet model offers a novel solution by using a **Variational Autoencoder** (VAE) framework combined with **FoldingNet**-based encoding to improve point cloud processing.

The VFNet model performs exceptionally well in tasks like:
- **Mesh generation**
- **Shape completion**
- **Representation learning**

### Simplicity and Customization

This repository is designed to be extremely simple to use. You can train the model on any custom dataset with **just a folder of meshes** of a particular object. There’s no complicated setup required. Just organize your mesh files and point cloud data, and you're ready to go!

## Features
- **Fully probabilistic VAE model** for point cloud data.
- **FoldingNet-based encoder** for efficient feature encoding.
- **Support for different datasets** like `modelnet40`, and `shapenetcore`, but can easily be extended to any custom dataset.

## Setup

### Requirements

To run the VFNet model, ensure the following dependencies are installed. You can install them all at once with the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```txt
h5py==3.12.1
matplotlib==3.7.5
nflows==0.14
numpy==2.2.3
pandas==2.2.3
plyfile==1.1
pyntcloud==0.3.1
scikit_learn==1.6.1
scipy==1.15.2
torch==2.2.0+cu121
tqdm==4.66.5
trimesh==4.5.3
```

### Dataset

The model supports various datasets:

- `modelnet40`
- `shapenetcore`

However, you can easily extend it to any **custom dataset** by just providing a folder containing mesh files of a particular object. There’s no additional complexity — just prepare your data, and the repo will handle the rest.

### Training

You can start training with the following command:

```bash
python main.py
```

### Model Configuration

- **Encoder**: By default, VFNet uses the `foldnet` encoder. 
- **Decoder**: By default, the model uses the `stochman` decoder.
- **Training Mode**: You can train the model normally or choose to train only the standard deviation with `--std_training`.

### Optimizer and Scheduler

VFNet uses the **Adamax** optimizer by default, with a **ReduceLROnPlateau** learning rate scheduler. You can adjust the learning rate and other optimizer parameters as needed.

### Resuming Training

You can resume training from a previously saved checkpoint using the `--resume_from` argument. The model and optimizer states will be restored from the specified path.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
