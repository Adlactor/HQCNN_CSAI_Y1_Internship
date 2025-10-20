import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from qiskit_aer import Aer
from ENEQR_pennylane import ENEQR
from quantum_circuit import SVC
from LRUCache import LRUCache

"""
Quanvolutional Module

Implements a classical-quantum hybrid 'quanvolutional' layer that:
    1. Extracts patches from input images.
    2. Encodes each patch into a quantum state via ENEQR (or uses precomputed states).
    3. Applies a variational quantum circuit (SVC) to extract expectation-based features.
    4. Normalizes and reshapes features for downstream classical processing.
Includes:
    - MinMaxNorm: Per-sample min-max normalization to [0,1].
    - QuanvolutionalLayer: Patch-based quantum feature extractor.
"""
    
class MinMaxNorm(nn.Module):
    """
    Per-sample Min-Max Normalization Layer

    Scales each sample in a batch independently to the [0, 1] range.
    Differentiable across all steps, works with any number of leading batch dims.
    """
    def __init__(self, eps: float = 1e-8):
        """
        Initialize normalization layer.

        Args:
            eps (float): Small constant to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor per sample to [0,1].

        Args:
            x (torch.Tensor): Input of shape (batch, ...).

        Returns:
            torch.Tensor: Normalized tensor of same shape.
        """
        # Compute min and max across all non-batch dimensions
        dims = tuple(range(1, x.ndim))
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
        return (x - x_min) / (x_max - x_min + self.eps)

class QuanvolutionalLayer(nn.Module):
    """
    Hybrid Quantum Convolutional Layer

    Extracts quantum features from image patches. Steps:
        a. Optionally extract patches and encode via ENEQR into statevectors.
        b. Cache or load precomputed statevectors for speed.
        c. Batch statevectors and run through SVC variational circuit.
        d. Reshape, normalize (MinMaxNorm), and permute features.
    """
    def __init__(self, patch_size, stride=2, padding=0, q=2, backend=None, random_seed=None, 
                cache_capacity=50000, use_precomputed=True):
        """
        Initialize the QuanvolutionalLayer.

        Args:
            patch_size (int): Side length of square patches.
            stride (int): Step size between patches.
            padding (int): Zero-padding around image borders.
            q (int): Qubit count for gray-level encoding.
            backend: Qiskit backend (defaults to statevector simulator).
            random_seed (int): Seed for reproducibility.
            cache_capacity (int): Max entries in ENEQR cache.
            use_precomputed (bool): Skip ENEQR if True, assume statevectors given.
        """
        super(QuanvolutionalLayer, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.q = q
        self.backend = backend if backend is not None else Aer.get_backend('aer_simulator_statevector')
        self.random_seed = random_seed
        self.use_precomputed = use_precomputed

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
        
        # Only initialize cache and ENEQR-related code if not using precomputed values.
        if not self.use_precomputed:
            self.eneqr_cache = LRUCache(capacity=cache_capacity)
        
        # Calculate the number of wires (qubits) required based on patch_size and q.
        self.wires = self.calculate_length(patch_size, q)
        # Instantiate the new differentiable SVC.
        self.svc = SVC(wires=self.wires, seed=random_seed)

        self.minmax_norm = MinMaxNorm()

    def calculate_length(self, patch_size, q):
        """
        Compute total qubit wires from patch size and gray qubits.

        wires = 2 * ceil(log2(patch_size)) + q + 1

        Args:
            patch_size (int): Patch side length.
            q (int): Number of gray-level qubits.

        Returns:
            int: Total qubit count.
        """
        return 2 * int(math.ceil(math.log2(patch_size))) + q + 1

    def extract_patches(self, image):
        """
        Slice a 2D square image into overlapping patches.

        Args:
            image (np.ndarray): 2D array (H x W), H == W.

        Returns:
            patches (List[np.ndarray]): List of patch arrays.
            nrow, ncol (int, int): Patch grid dimensions.

        Raises:
            ValueError: If image is not 2D square.
        """
        # Validate shape
        if not (image.ndim == 2 and image.shape[0] == image.shape[1]):
            raise ValueError("Image must be a 2D square array.")

        # Apply padding if specified
        if self.padding > 0:
            image = np.pad(image, pad_width=self.padding, mode='constant', constant_values=0)

        # Calculate number of patches along each dimension
        num_patches = ((image.shape[0] - self.patch_size) // self.stride) + 1

        patches = []
        for i in range(num_patches):
            for j in range(num_patches):
                patch = image[i * self.stride : i * self.stride + self.patch_size,
                            j * self.stride : j * self.stride + self.patch_size]
                patches.append(patch)

        return patches, num_patches, num_patches

    def apply_quantum(self, patches, nrow, ncol):
        """
        Encode patches into quantum states and extract features via SVC.

        Steps:
            1. Convert each patch to a statevector (either precomputed or via ENEQR).
            2. Stack statevectors into a batch tensor.
            3. Execute SVC to obtain Pauli-Z expectations per wire.
            4. Reshape output to (nrow, ncol, wires).

        Args:
            patches (List): Raw patch arrays or statevector tensors.
            nrow (int), ncol (int): Grid dimensions of patches.

        Returns:
            features (torch.Tensor): Quantum features [nrow, ncol, wires].
        """
        q_inputs = []

        for patch in patches:
            if self.use_precomputed:
                # patch is already a state-vector (torch.Tensor or numpy array)
                if isinstance(patch, torch.Tensor):
                    sv = patch
                else:
                    sv = torch.from_numpy(patch)
            else:
                patch_scaled = np.rint(patch * 255).astype(np.int32)
                key          = tuple(patch_scaled.ravel())
                sv = self.eneqr_cache.get(key)
                if sv is None:
                    eneqr = ENEQR(patch_scaled.tolist(), self.q)
                    sv    = eneqr(torch.zeros(1)).squeeze(0)      # (wires,)
                    self.eneqr_cache.put(key, sv)
            q_inputs.append(sv)

        q_batch = torch.from_numpy(np.stack([sv.detach().cpu().numpy() for sv in q_inputs])).to(q_inputs[0].device)
        exp_batch = self.svc(q_batch)                             # (N, wires)

        features = exp_batch.view(nrow, ncol, self.svc.wires)     # (rows, cols, wires)
        return features

    def forward(self, x):
        """
        Forward pass: extract and quantum-process patches for a batch of images.

        Args:
            x (torch.Tensor): Input shape:
                - If use_precomputed: (B, H, W, wires)
                - Else: (B, 1, H, W) or (B, H, W)

        Returns:
            torch.Tensor: Quantum feature maps of shape (B, wires, new_H, new_W).
        """
        batches = []
        if self.use_precomputed:
            # x[b,i,j] already a state-vector tensor of shape (wires,)
            Batch, H, W, _ = x.shape
            for b in range(Batch):
                patches = [ x[b, i, j] for i in range(H) for j in range(W) ]
                batches.append((patches, H, W))
        else:
            # raw images: extract pixel patches and ENEQR-encode
            x_np = x.detach().cpu().numpy() if torch.is_tensor(x) else x
            for img in x_np:
                if img.ndim == 3 and img.shape[0] == 1:
                    img = img.squeeze(0)
                patches, H, W = self.extract_patches(img)
                batches.append((patches, H, W))

        batch_features = []

        for patches, H, W in batches:
            features = self.apply_quantum(patches, H, W)    
        
            # print("test")
            num_features = features.shape[2]
            # might need real()
            flat = features.view(-1, num_features)
            features_scaled = self.minmax_norm(flat)
            features_scaled = features_scaled.view(H, W, num_features)
            batch_features.append(features_scaled)

        B = len(batch_features)
        R, C, F = batch_features[0].shape
        batch_features_tensor = torch.empty((B, R, C, F), dtype=batch_features[0].dtype, device=x.device)
        for i, feat in enumerate(batch_features):
            batch_features_tensor[i].copy_(feat)
        # Permute dimensions if needed
        return batch_features_tensor.permute(0, 3, 1, 2)

    def visualize_features(self, features):
        """
        Plot each quantum feature map as a grayscale image.

        Args:
            features (np.ndarray): Shape (H, W, num_features).
        """
        _, _, num_features = features.shape

        plt.figure(figsize=(12, 6))

        for k in range(num_features):
            plt.subplot(1, num_features, k + 1)
            plt.title(f"Feature {k + 1}")
            plt.imshow(features[:, :, k], cmap='gray', vmin=0, vmax=1)
            plt.colorbar()

        plt.tight_layout()
        plt.show()