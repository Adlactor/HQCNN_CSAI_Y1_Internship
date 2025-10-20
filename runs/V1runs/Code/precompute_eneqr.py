import math
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from ENEQR_pennylane import ENEQR

"""
Module: ENEQR Precomputation Script

Precomputes quantum-encoded patch states for MNIST images using the ENEQR encoder
and stores them in an HDF5 file for efficient downstream loading.
Each 2x2 patch is encoded into a quantum expectation vector and saved.
"""

class ENEQRPrecomputer:
    """
    Precompute and store ENEQR quantum states for image patches.

    Attributes:
        q (int): Number of gray-level encoding qubits.
        patch_size (int): Side length of square patches (pixels).
        stride (int): Step size between patch extraction windows.
        dataset_dir (str): Directory to download/load MNIST data.
        output_h5 (str): Path to output HDF5 file.
        nrow (int): Number of patches per row (and column).
        ncol (int): Same as nrow (square image assumption).
        n_wires (int): Total qubit wires used by ENEQR encoder.
    """
    def __init__(self, q=2, patch_size=2, stride=2, dataset_dir='./data', output_h5='eneqr_states.h5'):   ## Changed to 2 from 4 for V1-3 Checks (Mahmoud Sallam)
        """
        Initialize precomputation parameters.

        Args:
            q (int): Gray-level encoding qubits (bit-depth for pixel intensity).
            patch_size (int): Side length of the extracted image patches.
            stride (int): Horizontal/vertical step between patches.
            dataset_dir (str): Local directory for MNIST data storage.
            output_h5 (str): Filename for storing precomputed HDF5 states.
        """
        self.q = q
        self.patch_size = patch_size
        self.stride = stride
        self.dataset_dir = dataset_dir
        self.output_h5 = output_h5
        self.nrow = (28 - self.patch_size) // self.stride + 1
        self.ncol = self.nrow
        self.n_wires = 2 * int(math.ceil(math.log2(self.patch_size))) + self.q + 1

    def run(self):
        """
        Execute the precomputation:
            1. Download/load MNIST train and test sets.
            2. For each image, extract patches, encode via ENEQR, compute quantum outputs.
            3. Store the resulting state arrays and labels in an HDF5 dataset.
        """
        # Load datasets with pixel values in [0,1]
        train_ds = MNIST(root=self.dataset_dir, train=True, download=True, transform=ToTensor())
        test_ds  = MNIST(root=self.dataset_dir, train=False, download=True, transform=ToTensor())

        with h5py.File(self.output_h5, 'w') as h5f:
            for split, ds in [('train', train_ds), ('test', test_ds)]:
                grp = h5f.create_group(split)
                N   = len(ds)

                ds_states = grp.create_dataset(
                    'states',
                    shape=(N, self.nrow, self.ncol, self.n_wires),
                    dtype='f4'
                )
                grp.create_dataset(
                    'labels',
                    data=ds.targets.numpy(),
                    dtype='i8'
                )

                for idx in tqdm(range(N), desc=f'Precomputing {split} set'):
                    img, _ = ds[idx]
                    img_np = img.squeeze(0).numpy()
                    states_np = np.zeros((self.nrow, self.ncol, self.n_wires), dtype=np.float32)

                    for i in range(self.nrow):
                        for j in range(self.ncol):
                            patch = img_np[
                                i*self.stride : i*self.stride+self.patch_size,
                                j*self.stride : j*self.stride+self.patch_size
                            ]
                            patch_int = np.rint(patch * 255).astype(int)
                            eneqr = ENEQR(patch_int.tolist(), self.q)
                            with torch.no_grad():
                                out = eneqr(torch.zeros((1,))).numpy()
                            states_np[i, j] = out

                    ds_states[idx] = states_np

        print(f'Precomputed ENEQR states saved to {self.output_h5}')

if __name__ == '__main__':
    precomputer = ENEQRPrecomputer()
    precomputer.run()
