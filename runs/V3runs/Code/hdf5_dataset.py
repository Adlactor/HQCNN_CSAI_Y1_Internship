import h5py
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    """
    HDF5Dataset wraps an HDF5 file storing precomputed quantum states and labels.

    Each worker lazily opens the file to avoid inter-process file handle issues.

    Attributes:
        h5_path (str): Path to the HDF5 file.
        split (str): Dataset split key, e.g., 'train' or 'test'.
        _length (int): Number of samples in the chosen split.
        _file (h5py.File | None): Active file handle for reading data.
    """
    def __init__(self, h5_path: str, split: str = 'train'):
        """
        Initialize the HDF5Dataset.

        Args:
            h5_path (str): File system path to the .h5 file.
            split (str): Group name inside HDF5 ('train' or 'test').
        """
        # Store only path/split and dataset length
        self.h5_path = h5_path
        self.split   = split

        # Temporarily open to get length, then close immediately
        with h5py.File(self.h5_path, 'r') as f:             # no persistent handle :contentReference[oaicite:3]{index=3}
            self._length = f[f'{self.split}/states'].shape[0]  # read-only metadata :contentReference[oaicite:4]{index=4}

        self._file = None  # will hold the h5py.File inside each worker

    def __len__(self):
        """
        Return the number of samples in this dataset split.

        Returns:
            int: Length of dataset (number of images).
        """
        return self._length  # no file access here :contentReference[oaicite:5]{index=5}

    def __getitem__(self, idx):
        """
        Retrieve a single sample (quantum state tensor and label).

        Lazily opens the HDF5 file on first access per worker process to avoid
        serialization issues when using DataLoader with multiple workers.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                x: State tensor of shape (rows, cols, wires), dtype float.
                y: Label tensor, dtype long.
        """
        # Lazily open file in this worker process on first call :contentReference[oaicite:6]{index=6}
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')

        # Read precomputed state-vector and label
        state = self._file[f'{self.split}/states'][idx]   # NumPy array slice :contentReference[oaicite:7]{index=7}
        label = int(self._file[f'{self.split}/labels'][idx])

        # Convert to torch.Tensor
        x = torch.from_numpy(state).float()
        y = torch.tensor(label, dtype=torch.long)
        return x, y
