import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

"""
Differentiable Structured Variational Circuit (SVC) Module

Defines a PyTorch nn.Module that wraps a PennyLane variational quantum circuit
for feature encoding or density-matrix-based inputs. Supports batched inputs,
parameter initialization, and circuit visualization.
"""

class SVC(nn.Module):
    """
    Structured Variational Quantum Circuit

    Attributes:
        wires (int): Number of qubits in the circuit.
        pairs (List[Tuple[int,int]]): Qubit index pairs for entangling layers.
        num_pairs (int): Number of parameterized blocks (= len(pairs)).
        weight_shapes (dict): Shapes of trainable parameters for TorchLayer.
        dev (qml.Device): PennyLane device (default.qubit) for simulation.
        circuit (qml.QNode): PennyLane QNode wrapping the _batched_circuit method.
        quantum_layer (qml.qnn.TorchLayer): PyTorch layer for quantum execution.
    """
    def __init__(self, wires=5, seed=None):
        """
        Initialize the SVC with a given number of wires and optional seed.

        Args:
            wires (int): Total qubits in the circuit.
            seed (int, optional): Random seed for reproducibility.
        """
        super(SVC, self).__init__()
        self.wires = wires
        if seed is not None:
            np.random.seed(seed)
        self.pairs = self.generate_pairs()
        self.num_pairs = len(self.pairs)
        self.weight_shapes = {"params": (self.num_pairs, 10)}
        
        # Use Pennylane's default.qubit statevector simulator.
        self.dev = qml.device("default.qubit", wires=self.wires)

        # Create a QNode by binding the batched circuit to the device
        self.circuit = qml.QNode(self._batched_circuit, self.dev, interface="torch", diff_method="best")

        # Wrap QNode as a PyTorch layer with specified trainable shapes
        self.quantum_layer = qml.qnn.TorchLayer(self.circuit, weight_shapes=self.weight_shapes)

    def initialize_params(self):
        """
        Generate trainable parameters for each variational block.

        Returns:
            nn.Parameter: Tensor of shape (num_pairs, 10) with values in [0,2π).
        """
        initial_params = np.random.uniform(0, 2 * np.pi, (self.num_pairs, 10)).astype(np.float32)
        return nn.Parameter(torch.tensor(initial_params))
    
    def variational_block(self, p, q1, q2):
        """
        Apply a parameterized variational block on qubit pair (q1, q2).

        Args:
            p (Tensor): 10-element parameter vector for this block.
            q1, q2 (int): Indices of the two qubits in the block.
        """
        # First single-qubit RX & RZ on each qubit (params 0-3)
        for i, q in enumerate([q1, q2]):
            qml.RX(p[2 * i], wires=q)
            qml.RZ(p[2 * i + 1], wires=q)
        # Controlled-RY gate from q1 to q2 using parameter p[4]
        qml.CRY(p[4], wires=[q1, q2])
        # Second block: for each qubit in [q1, q2], apply RX and RZ rotations using parameters 5-8.
        for i, q in enumerate([q1, q2]):
            qml.RX(p[2 * i + 5], wires=q)
            qml.RZ(p[2 * i + 6], wires=q)
        # Controlled-RY gate from q2 to q1 using parameter p[9]
        qml.CRY(p[9], wires=[q2, q1])
    
    def _batched_circuit(self, params, inputs):
        """
        Core circuit logic supporting batched inputs.

        Args:
            params (Tensor): Shape (num_pairs, 10) trainable parameters.
            inputs (Tensor): Either a feature vector of length 'wires' or
                                a statevector of length 2**wires.
        Returns:
            List[Expectation]: Pauli-Z expectations for each qubit.
        """
        # State preparation: either direct StatePrep or RY rotations
        if inputs.shape[-1] == 2 ** self.wires:          # batched state‑vector
            qml.StatePrep(inputs, wires=range(self.wires))
        elif inputs.shape[-1] == self.wires:             # batched feature vector
            for idx in range(self.wires):
                qml.RY(inputs[:, idx], wires=idx)
        else:
            raise ValueError("Unexpected input length")

        # Apply variational blocks on each qubit pair
        for k, (q1, q2) in enumerate(self.pairs):
            p = params[k]
            self.variational_block(p, q1, q2)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

    def generate_pairs(self):
        """
        Produce qubit adjacency pairs for entanglement layers.

        Returns:
            List[Tuple[int,int]]: Ordering: even pairs, wrap-around, odd pairs.
        """
        # Pairs of (0,1), (2,3), ...
        layer1 = [(i, i+1) for i in range(0, self.wires - 1, 2)]
        # Wrap‑around (last,0) to boost entanglement
        wrap = (self.wires - 1, 0)                                                            # Will comment out once I get to V1 (as its a main difference between V1 and V2 - Mahmoud Sallam)
        # Second layer: odd‑indexed adjacent pairs
        layer2 = [(i, i+1) for i in range(1, self.wires - 1, 2)]
        # Return layer1, then wrap, then layer2
        return layer1 + layer2 # + [wrap]

    def forward(self, inputs):
        """
        Forward pass through the quantum TorchLayer.

        Args:
            inputs (Tensor): Input tensor matching QNode expectations.
        Returns:
            Tensor: Stacked expectation values (shape: batch x wires).
        """
        # Execute quantum layer; outputs a list of expectations
        output = self.quantum_layer(inputs)
        return output

    def get_circuit(self):
        """
        Retrieve the underlying PennyLane QNode.

        Returns:
            QNode: The circuit function bound to the device.
        """
        return self.circuit
    
    def draw_circuit(self, params=None, inputs=None, style="text"):
        """
        Render the quantum circuit diagram.

        Args:
            params (Tensor): Parameters to visualize (required).
            inputs (Tensor): Circuit inputs for completeness.
            style (str): 'text' for ASCII, 'mpl' for Matplotlib.
        Returns:
            Diagram: ASCII string or Matplotlib figure tuple.
        """
        if params is None:
            raise ValueError("Pass an explicit 'params' tensor when drawing.")

        if style == "text":
            return qml.draw(self.circuit)(params, inputs)
        elif style == "mpl":
            return qml.draw_mpl(self.circuit)(params, inputs)
        else:
            raise ValueError("Unknown style. Use 'text' or 'mpl'.")
        