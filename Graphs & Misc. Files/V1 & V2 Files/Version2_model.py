import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from quanvolutional import QuanvolutionalLayer

"""
HQCNN: Hybrid Quantum-Classical Convolutional Neural Network Module

Combines optional raw image compression via fixed PCA with a quantum
feature extraction branch (QuanvolutionalLayer) and a classical
fully-connected classification head for MNIST-like image tasks.
"""

class HQCNN(nn.Module):
    """
    Hybrid Quantum-Classical CNN with raw PCA compression and quanvolutional encoding.

    Args:
        patch_size (int): Side length of image patches for quantum encoding.
        stride (int): Stride between patches.
        padding (int): Zero-padding around images for patch extraction.
        q (int): Number of qubits for gray-level encoding in ENEQR.
        num_classes (int): Number of output classes.
        backend: Quantum backend (e.g., Qiskit) for the QuanvolutionalLayer.
        random_seed (int): Seed for reproducibility of quantum operations.
        n_raw_pca_components (int): Number of PCA components for raw image compression.
        use_precomputed (bool): If True, bypass raw PCA and expect precomputed quantum states.

    Attributes:
        raw_pca (PCA or None): Fixed PCA transformer applied to raw images.
        quanv (QuanvolutionalLayer): Quantum patch-based feature extractor.
        num_features (int): Number of quantum features (quantum wires).
        pool (nn.MaxPool2d): Max pooling layer for spatial downsampling.
        fc1, fc2, fc3 (nn.Linear): Fully-connected layers for classification head.
        bn1, bn2 (nn.BatchNorm1d): Batch normalization layers.
        relu (nn.ReLU): Activation function.
        dropout (nn.Dropout): Dropout regularization.
    """
    def __init__(self, patch_size=2, stride=2, padding=0, q=2, num_classes=10, backend=None, random_seed=None,
                n_raw_pca_components=784, use_precomputed=True):
        super(HQCNN, self).__init__()
        self.n_raw_pca_components = n_raw_pca_components
        self.use_precomputed = use_precomputed
        
        # ----- Raw Image Compression -----
        # If n_raw_pca_components is not 784, load precomputed PCA parameters.
        if n_raw_pca_components == 784 or use_precomputed:
            # Skip PCA: we use the raw image directly.
            self.raw_pca = None
        
        # ----- Quantum Encoding Branch -----
        # We now feed the compressed image (of shape compressed_size x compressed_size) 
        # to the quantum branch. Note: we keep the patch_size parameter for the quantum branch unchanged.
        self.quanv = QuanvolutionalLayer(
            patch_size=patch_size,  # remains as given (e.g., 2)
            stride=stride,
            padding=padding,
            q=q,
            backend=backend,
            random_seed=random_seed,
            use_precomputed=use_precomputed
        )
        
        # Compute the grid of patches dynamically for both modes:
        # For raw input: compressed_size = sqrt(PCA components) or 28 when no PCA
        # For precomputed: input has grid directly, which equals ((28 - patch_size)//stride + 1)
        if use_precomputed:
            grid_size = ((28 - patch_size) // stride) + 1
        else:
            grid_size = ((int(n_raw_pca_components ** 0.5) - patch_size) // stride) + 1
        # After 2×2 pooling
        pooled_dim = grid_size // 2
        
        # Calculate the number of quantum features (wires) as given by the quanv layer.
        self.num_features = self.quanv.calculate_length(patch_size, q)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FC layer input dimension: wires × (pooled_dim)^2
        fc_input_dim = self.num_features * (pooled_dim ** 2)
        # Deeper MLP head with BatchNorm
        self.fc1    = nn.Linear(fc_input_dim, 128)
        self.fc3    = nn.Linear(128, num_classes)
        self.relu   = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass through HQCNN.

        Args:
            x (torch.Tensor): Input batch.
                - If raw mode: shape (B, 28, 28) grayscale images.
                - If precomputed mode: shape (B, grid, grid, wires).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        batch_size = x.size(0)
        
        if self.use_precomputed:
            # Use x directly as precomputed ENEQR state.
            x_compressed = x  # Expected shape: (batch_size, num_patch_rows, num_patch_cols, dm_dim, dm_dim)
        elif self.n_raw_pca_components == 784:
            x_compressed = x  # Already 28x28
        
        # Pass the compressed image (e.g., 16x16) to the quantum branch.
        q_features = self.quanv(x_compressed)  # Expected shape: (batch_size, num_features, 14, 14)
        q_features = torch.real(q_features)

        pooled = self.pool(q_features)         # (batch_size, num_features, 7, 7)
        # Deeper head forward
        x = pooled.reshape(batch_size, -1).float()

        x = self.fc1(x)                            # (B, 128)
        x = self.relu(x)

        out = self.fc3(x)                          # (B, num_classes)
        return out
