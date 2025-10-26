# HQCNN with Quanvolutional Encoding
## Internship Preamble

> All credits towards the initial codebase and dissertation are fully attributed to Mr. Hoo Kai Ping.

This is a Summer Internship Project related to the analysis, understanding, and review of Hybrid Quantum Convolutional Neural Networks, which was a Final Year Project operated by Hoo Kai Ping. This Internship was sanctioned by Dr. Tan Chye Cheah and operated by myself. My work focuses on replicating the results of the HQCNN and verifying their integrity while structuring the review to allow undergraduate students to get a foothold on Quantum Convolutional Neural Networks and Quantum Encoding.

I intend to further the progress in this project at a later date to fulfil goals set in the reflection section of the review, which include:
- Analysis of other datasets, such as CIFAR-10.
- Research & development of improvements on the pre-existing HQCNN model.
- Performing more exhaustive testing, which includes increasing the epoch value.


## Project Overview

Hybrid Quantum-Classical Convolutional Neural Network (HQCNN) for image classification (e.g., MNIST). Two execution modes:

* **Precompute & Run**: Encode every image patch into quantum expectation states using the ENEQR encoder, store them in `eneqr_states.h5`, then train/evaluate HQCNN on these precomputed states.
* **Direct Run**: Skip precomputation if you already have `eneqr_states.h5`, and run HQCNN training/evaluation directly.

## Directory Structure

```
├── ENEQR_pennylane.py      # Quantum image encoder module
├── quantum_circuit.py      # Structured Variational Circuit (SVC) module
├── quanvolutional.py       # QuanvolutionalLayer + normalization
├── LRUCache.py             # LRU cache for encoder outputs
├── precompute_eneqr.py     # Script to precompute ENEQR states into .h5
├── hdf5_dataset.py         # HDF5Dataset wrapper for loading `.h5`
├── loaddata.py             # Classic MNIST DataLoader utility
├── model.py                # HQCNN model definition
├── main.py                 # Training & evaluation entrypoint (uses HDF5Dataset)
└── README.md               # This file
```

## Prerequisites
* **Python** 3.11 or higher
* **pip** package manager

**Install required packages and specific versions:**

Ensure that all packages are installed on your Python client and that the Python client possessing all the required packages is the one executing the program. This is a known issue if you have more than one Python interpreter (e.g., Anaconda, Python 3.12, PyCharm, JupyterLab, etc.).

```bash
pip install \
  torch==2.5.1+cu124 \
  torchvision==0.20.1+cu124 \
  pennylane==0.40.0 \
  qiskit==1.3.0 \                 // Note that the "convert_to_target" function is deprecated from 1.3 onwards and is removed from 2.0 onwards as per 
                                  // https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.4/qiskit.providers.convert_to_target
  qiskit-aer==0.16.0 \
  h5py==3.12.1 \
  numpy==1.26.4 \
  matplotlib==3.10.0 \
  tqdm==4.67.1
```

*(All other dependencies will be installed automatically with these packages.)*

## Usage

### 1. Precompute Mode

Generate quantum-encoded patch states and save to `eneqr_states.h5`:

```bash
python precompute_eneqr.py
```

**What happens**:

1. Downloads MNIST (train/test) to `./data`.
2. Splits each 28×28 image into 2×2 patches (stride=2 by default).
3. Encodes each patch via the ENEQR PennyLane encoder into a vector of length **wires** (default q=4).
4. Stores `states` (shape `[N, nrow, ncol, wires]`) and `labels` in `eneqr_states.h5`.
  * NOTE: for V1-3, you need to precompute the dataset by modifying q=2 instead of 4, for V4 keep q=4.

After completion, you will see:

```
Precomputed ENEQR states saved to eneqr_states.h5
```

### 2. Direct Run Mode

Skip precompute if you already have `eneqr_states.h5` in the project root.

Train and evaluate HQCNN on the HDF5 dataset.

```bash
python main.py
```

**What happens**:

1. Loads `eneqr_states.h5` via `HDF5Dataset`.
2. Constructs HQCNN (quanvolutional→pool→MLP head).
3. Trains for **10 epochs** (default) with `batch_size=64`.
4. Logs metrics to `runs/` (TensorBoard) and saves `checkpoint.pth` after each epoch.

### 3. Viewing Logs

TensorBoard is launched automatically on port **6040** by `main.py`. To open the dashboard manually:

```bash
tensorboard --logdir runs --port 6040
```

Visit: [http://localhost:6040](http://localhost:6040)

## Configuration

All key hyperparameters live in **main.py**:

| Parameter       | Default | Description                                 |
| --------------- | ------- | ------------------------------------------- |
| `patch_size`    | 2       | Side length of patches for quantum encoding |
| `stride`        | 2       | Step between patch extractions              |
| `q`             | 4       | Number of gray-level qubits in ENEQR        |
| `num_epochs`    | 10      | Training epochs                             |
| `batch_size`    | 64      | Samples per training batch                  |
| `learning_rate` | 0.001   | AdamW optimizer learning rate               |

Modify these values directly in **main.py** before running.

## Alternative: Classic MNIST

If you prefer to use the classical MNIST pipeline (without quantum encoding):

1. Uncomment the `get_mnist_dataloaders` import and usage in **main.py**.
2. Comment out HDF5Dataset sections.
3. Ensure `loaddata.py` is present (it lives in this repo).
4. Run `python main.py` to train with raw pixel inputs.

## Outputs

* **runs/**: TensorBoard event files
* **checkpoint.pth**: Latest model & optimizer state
* **version1\_eneqr.png**, **svc\_circuit\_version4.png** (if you invoke circuit drawing in demos)

## Troubleshooting

* **FileNotFoundError**: Run `precompute_eneqr.py` to generate `eneqr_states.h5`.
* **GPU fallback**: If no CUDA, code runs on CPU automatically.
* **Dependencies**: Ensure all packages are installed in the same environment 

## Addendum Notes - M S:
* **Dependencies Note #02:** Ensure the versions of all dependencies follow that of the **Prerequisites**, this can be checked with `pip list`.
* **Tensorboard Event File Access**: Initiate Command Prompt and run `tensorboard --logdir_spec [Folder Address directly from c:/] --port 6040` then boot up [http://localhost:6040](http://localhost:6040), If that doesn't work, create a new folder and relocate the .0 file there, then copy the folder address and use that when initiating tensorboard via CMD Prompt.







