"""
Hardware Intermediate Representation (IR) for difflogic models.

Provides LogicGate and HardwareModel classes that represent a logic gate
network in a hardware-friendly form suitable for Verilog generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# Mapping from difflogic gate index to canonical gate name.
ALL_OPERATIONS = [
    "ZERO",          # 0
    "AND",           # 1
    "NOT_IMPLIES",   # 2  a & ~b
    "A",             # 3  buffer a
    "NOT_IMPLIED_BY", # 4 b & ~a
    "B",             # 5  buffer b
    "XOR",           # 6
    "OR",            # 7
    "NOT_OR",        # 8  NOR
    "NOT_XOR",       # 9  XNOR
    "NOT_B",         # 10 ~b
    "IMPLIED_BY",    # 11 ~b | a
    "NOT_A",         # 12 ~a
    "IMPLIES",       # 13 ~a | b
    "NOT_AND",       # 14 NAND
    "ONE",           # 15
]


@dataclass
class LogicGate:
    """Represents a single logic gate in the hardware IR."""

    gate_id: int
    """Unique gate identifier (globally unique across all layers)."""

    gate_type: str
    """Gate type string, one of ALL_OPERATIONS."""

    input_ids: List[int]
    """IDs of the two input signals (gate IDs or negative for primary inputs).

    Convention: primary input *i* is encoded as -(i+1) so that input 0 → -1.
    """

    layer: int = 0
    """Layer index the gate belongs to (0-based)."""

    depth: int = 0
    """Critical-path depth from primary inputs."""

    fanout: int = 0
    """Number of gates that use this gate's output."""

    def is_primary_input_ref(self, idx: int) -> bool:
        return idx < 0

    def primary_input_index(self, idx: int) -> int:
        """Decode a primary-input reference to its 0-based index."""
        assert self.is_primary_input_ref(idx)
        return -(idx + 1)


@dataclass
class HardwareModel:
    """Hardware IR for a complete difflogic network."""

    num_inputs: int
    """Number of primary Boolean inputs."""

    num_outputs: int
    """Number of primary outputs (classes)."""

    gates: List[LogicGate] = field(default_factory=list)
    """All gates in topological order."""

    output_gate_ids: List[int] = field(default_factory=list)
    """IDs of gates that drive the output bus (one per output bit)."""

    num_out_per_class: int = 1
    """Number of last-layer gate outputs per class (for GroupSum)."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_gate(self, gate: LogicGate) -> None:
        self.gates.append(gate)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compute_depths(self) -> None:
        """Assign *depth* to every gate (longest path from any primary input)."""
        depth_map: Dict[int, int] = {}
        for gate in self.gates:  # already in topological order
            d = 0
            for inp in gate.input_ids:
                if inp >= 0:  # gate reference
                    d = max(d, depth_map.get(inp, 0) + 1)
                # primary inputs have depth 0
            gate.depth = d
            depth_map[gate.gate_id] = d

    def compute_fanouts(self) -> None:
        """Count how many times each gate is referenced."""
        fanout_map: Dict[int, int] = {}
        for gate in self.gates:
            for inp in gate.input_ids:
                if inp >= 0:
                    fanout_map[inp] = fanout_map.get(inp, 0) + 1
        for gate in self.gates:
            gate.fanout = fanout_map.get(gate.gate_id, 0)

    def topological_sort(self) -> List[LogicGate]:
        """Return gates in topological evaluation order.

        The gates list is expected to be built layer-by-layer so it is
        already topologically ordered; this method verifies and returns
        that ordering.
        """
        visited: Set[int] = set()
        order: List[LogicGate] = []
        gate_lookup: Dict[int, LogicGate] = {g.gate_id: g for g in self.gates}

        def visit(gid: int) -> None:
            if gid in visited:
                return
            g = gate_lookup[gid]
            for inp in g.input_ids:
                if inp >= 0:
                    visit(inp)
            visited.add(gid)
            order.append(g)

        for g in self.gates:
            visit(g.gate_id)
        return order

    # ------------------------------------------------------------------
    # Layer grouping
    # ------------------------------------------------------------------

    def gates_by_layer(self) -> Dict[int, List[LogicGate]]:
        """Return a dict mapping layer index → list of gates in that layer."""
        layers: Dict[int, List[LogicGate]] = {}
        for g in self.gates:
            layers.setdefault(g.layer, []).append(g)
        return layers

    def max_depth(self) -> int:
        """Return the maximum gate depth (critical path length)."""
        if not self.gates:
            return 0
        return max(g.depth for g in self.gates)

    def gate_count(self) -> int:
        return len(self.gates)

    def resource_summary(self) -> Dict[str, int]:
        """Return a simple resource estimation dictionary."""
        type_counts: Dict[str, int] = {}
        for g in self.gates:
            type_counts[g.gate_type] = type_counts.get(g.gate_type, 0) + 1
        return {
            "total_gates": self.gate_count(),
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "critical_path_depth": self.max_depth(),
            **{f"gate_{k.lower()}": v for k, v in type_counts.items()},
        }

    # ------------------------------------------------------------------
    # Combinational evaluation (software reference)
    # ------------------------------------------------------------------

    def evaluate(self, inputs: List[bool]) -> List[bool]:
        """Evaluate the network for a single input vector (Python reference)."""
        assert len(inputs) == self.num_inputs, (
            f"Expected {self.num_inputs} inputs, got {len(inputs)}"
        )

        def resolve(ref: int) -> bool:
            if ref < 0:
                return bool(inputs[-(ref + 1)])
            return signal_map[ref]

        signal_map: Dict[int, bool] = {}
        for gate in self.gates:
            a_ref, b_ref = gate.input_ids
            a = resolve(a_ref)
            b = resolve(b_ref)
            signal_map[gate.gate_id] = _eval_gate(gate.gate_type, a, b)

        return [signal_map[gid] for gid in self.output_gate_ids]


def _eval_gate(gate_type: str, a: bool, b: bool) -> bool:
    """Evaluate a single logic gate."""
    t = gate_type.upper()
    if t == "ZERO":
        return False
    elif t == "AND":
        return a and b
    elif t == "NOT_IMPLIES":
        return a and not b
    elif t == "A":
        return a
    elif t == "NOT_IMPLIED_BY":
        return b and not a
    elif t == "B":
        return b
    elif t == "XOR":
        return a != b
    elif t == "OR":
        return a or b
    elif t == "NOT_OR":
        return not (a or b)
    elif t == "NOT_XOR":
        return a == b
    elif t == "NOT_B":
        return not b
    elif t == "IMPLIED_BY":
        return (not b) or a
    elif t == "NOT_A":
        return not a
    elif t == "IMPLIES":
        return (not a) or b
    elif t == "NOT_AND":
        return not (a and b)
    elif t == "ONE":
        return True
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")
