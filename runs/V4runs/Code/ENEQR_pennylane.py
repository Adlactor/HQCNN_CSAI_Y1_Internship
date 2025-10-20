import math
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn

"""
ENEQR PennyLane Encoder Module

This module exposes the ENEQR quantum image encoder as a PyTorch nn.Module via
PennyLane's TorchLayer. It converts a grayscale image into a quantum state via
controlled operations, then measures expectation values across all wires.
"""

class ENEQR(nn.Module):
    """
    ENEQR Quantum Encoder

    Wraps a PennyLane QNode in a TorchLayer to encode grayscale image pixels
    into a quantum circuit. Preserves original qubit-mapping logic:
        - Position wires for coordinate encoding
        - Gray-intensity wires for amplitude encoding
        - Auxiliary wire for multi-controlled operations

    Attributes:
        image_data (List[List[int]]): Square 2D grayscale values (0-255)
        q (int): Number of qubits dedicated to gray-level encoding
        n (int): Number of position qubit pairs per dimension
        wires (int): Total number of circuit wires
        dev (qml.Device): PennyLane device for circuit execution
        _torch_layer (qml.qnn.TorchLayer): Torch-compatible quantum layer
    """
    def __init__(self, image_data, q: int = 8):
        """
        Initialize encoder configuration and build the TorchLayer.

        Parameters:
            image_data (List[List[int]]): Square grayscale image values
            q (int): Bit-depth qubits (controls quantization levels)
        """
        super().__init__()
        self.image_data = image_data
        self.q = q
        self.n = self._calc_n()
        # Total wires = gray qubits + auxiliary + position qubits (2*n)
        self.wires = 2 * self.n + self.q + 1
        self.dev = qml.device("default.qubit", wires=self.wires)

        # Construct the TorchLayer wrapping the QNode
        self._torch_layer = self._build_layer()

    def forward(self, x=None):
        """
        Forward pass through the TorchLayer.

        Args:
            x (torch.Tensor): Placeholder argument for Sequential compatibility

        Returns:
            torch.Tensor: Expectation values shape (batch_size, wires)
        """
        # Input x is unused; QNode uses stored image_data
        return self._torch_layer(x)

    def _calc_n(self):
        """
        Calculate minimum qubit pairs to index image dimensions.

        Ensures image_data is non-empty and square. Computes n = ceil(log2(size)).

        Returns:
            int: Number of bits per coordinate axis

        Raises:
            ValueError: If image_data is empty or not square
        """
        rows = len(self.image_data)
        if rows == 0 or any(len(r) != rows for r in self.image_data):
            raise ValueError("Image must be square and non‑empty")
        return int(math.ceil(math.log2(rows)))

    def _build_layer(self):
        """
        Build a PennyLane QNode and wrap it as a TorchLayer.

        Returns:
            qml.qnn.TorchLayer: PyTorch layer executing the quantum circuit
        """
        gray_wires = list(range(self.q))
        aux_wire = self.q
        pos_wires = list(range(self.q + 1, self.q + 1 + 2 * self.n))

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(inputs=None):            # TorchLayer requires this arg
            # Apply Hadamards on all position wires to create superposition
            self._hadamards(pos_wires)

            for y, row in enumerate(self.image_data):
                for x, gray in enumerate(row):
                    self._encode_pixel(gray, y, x,
                                        pos_wires, gray_wires, aux_wire)

            return [qml.expval(qml.PauliZ(w)) for w in range(self.wires)]

        torch_layer = qml.qnn.TorchLayer(circuit, weight_shapes={})
        return torch_layer

    def _hadamards(self, pos_wires):
        """
        Apply Hadamard gate on each position wire to create equal superposition.

        Args:
            pos_wires (List[int]): Wire indices for position qubits
        """
        for w in pos_wires:
            qml.Hadamard(wires=w)

    def _encode_pixel(self, gray, y, x, pos_wires, gray_wires, aux_wire):
        """
        Encode a single pixel's gray value at position (y, x) into the circuit.

        Steps:
        1. Normalize gray intensity into q-bit integer level
        2. If level == 0, skip
        3. Compute binary strings for pos and gray bits
        4. Activate auxiliary via multi-controlled X
        5. Apply CNOTs from aux to gray wires where gray bits=1
        6. Deactivate auxiliary

        Args:
            gray (int): Pixel intensity [0..255]
            y (int): Row index
            x (int): Column index
            pos_wires (List[int]): Position qubit wires
            gray_wires (List[int]): Gray-level qubit wires
            aux_wire (int): Auxiliary wire index

        Raises:
            ValueError: If gray outside [0,255]
        """
        # Validate intensity range
        if not 0 <= gray <= 255:
            raise ValueError("Gray out of range 0‑255")

        norm = int(round(gray // (256 / (2 ** self.q))))
        if norm == 0:
            return

        bin_pos = f"{y:0{self.n}b}{x:0{self.n}b}"
        bin_gray = f"{norm:0{self.q}b}"
        one_bits = [i for i, b in enumerate(bin_gray) if b == "1"]

        self._activate_aux(bin_pos, aux_wire, pos_wires)
        for i in one_bits:
            qml.CNOT(wires=[aux_wire, gray_wires[i]])
        self._deactivate_aux(bin_pos, aux_wire, pos_wires)

    def _activate_aux(self, bin_pos, aux_wire, pos_wires):
        """
        Prepare auxiliary wire conditioned on position bits.

        Args:
            bin_pos (str): Binary coordinate string
            aux_wire (int): Auxiliary wire index
            pos_wires (List[int]): Position qubit wires
        """
        # Flip position wires where bit=0 to use in multi-controlled gate
        for i, b in enumerate(bin_pos):
            if b == "0":
                qml.PauliX(wires=pos_wires[i])
        qml.MultiControlledX(wires=pos_wires + [aux_wire])

    def _deactivate_aux(self, bin_pos, aux_wire, pos_wires):
        """
        Revert auxiliary and position-wire flips post-encoding.

        Args:
            bin_pos (str): Binary coordinate string
            aux_wire (int): Auxiliary wire index
            pos_wires (List[int]): Position qubit wires
        """
        # Undo multi-controlled activation
        qml.MultiControlledX(wires=pos_wires + [aux_wire])
        for i, b in reversed(list(enumerate(bin_pos))):
            if b == "0":
                qml.PauliX(wires=pos_wires[i])