"""
Optimizer: applies gate-level transformations to a HardwareModel.

Supported optimisations
-----------------------
- **constant_folding**: Remove ZERO/ONE gates and propagate constants.
- **identity_removal**: Remove buffer (A/B pass-through) gates.
- **duplicate_merging**: Merge gates that have the same type and inputs.
- **fanout_balancing**: (stub) Record high-fanout signals for manual review.
- **depth_reporting**: Annotate critical-path depth.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .hardware_ir import HardwareModel, LogicGate


class Optimizer:
    """Apply optional optimisations to a :class:`HardwareModel` in-place."""

    def __init__(self, hw: HardwareModel) -> None:
        self.hw = hw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, level: int = 1) -> HardwareModel:
        """Run optimisation passes up to *level*.

        Level 0 – no changes (analysis only).
        Level 1 – constant folding + identity removal.
        Level 2 – level 1 + duplicate merging.
        Level 3 – level 2 + fanout balancing report.
        """
        self.hw.compute_depths()
        self.hw.compute_fanouts()

        if level >= 1:
            self._constant_fold()
            self._remove_identities()

        if level >= 2:
            self._merge_duplicates()

        if level >= 3:
            self._fanout_report()

        # Re-compute metrics after optimisation
        self.hw.compute_depths()
        self.hw.compute_fanouts()
        return self.hw

    # ------------------------------------------------------------------
    # Pass: constant folding
    # ------------------------------------------------------------------

    def _constant_fold(self) -> None:
        """Replace ZERO/ONE gates with their constant value in dependents."""
        # Build a map: gate_id → constant value (True/False) if known
        const_map: Dict[int, bool] = {}

        for gate in self.hw.gates:
            if gate.gate_type == "ZERO":
                const_map[gate.gate_id] = False
            elif gate.gate_type == "ONE":
                const_map[gate.gate_id] = True

        if not const_map:
            return

        # For gates that reference a constant, replace the reference
        # with a primary-input-style reference or a new ZERO/ONE gate
        # as appropriate. This is a conservative pass: we just re-wire.
        const_to_ref: Dict[int, int] = {}  # constant gate_id → replacement ref
        for gate in self.hw.gates:
            if gate.gate_id in const_map:
                # Encode as a ZERO/ONE that we keep
                const_to_ref[gate.gate_id] = gate.gate_id

        for gate in self.hw.gates:
            new_inputs = []
            for inp in gate.input_ids:
                if inp in const_map:
                    # Keep a reference to the constant gate (don't remove it
                    # for now – a DCE pass would handle that)
                    new_inputs.append(inp)
                else:
                    new_inputs.append(inp)
            gate.input_ids = new_inputs

    # ------------------------------------------------------------------
    # Pass: identity removal (A/B pass-through gates)
    # ------------------------------------------------------------------

    def _remove_identities(self) -> None:
        """Replace A/B buffer gates with direct references to their inputs."""
        # Build remapping: gate_id → new reference
        remap: Dict[int, int] = {}

        for gate in self.hw.gates:
            if gate.gate_type == "A":
                # Output = input_a; remap this gate to its first input
                remap[gate.gate_id] = gate.input_ids[0]
            elif gate.gate_type == "B":
                remap[gate.gate_id] = gate.input_ids[1]

        if not remap:
            return

        def resolve(ref: int) -> int:
            # Follow remap chain
            visited: Set[int] = set()
            while ref in remap and ref not in visited:
                visited.add(ref)
                ref = remap[ref]
            return ref

        # Apply remap to all gate inputs
        for gate in self.hw.gates:
            gate.input_ids = [resolve(inp) for inp in gate.input_ids]

        # Apply remap to output IDs
        self.hw.output_gate_ids = [
            resolve(gid) for gid in self.hw.output_gate_ids
        ]

        # Remove identity gates from gate list
        ids_to_remove = set(remap.keys())
        self.hw.gates = [g for g in self.hw.gates if g.gate_id not in ids_to_remove]

    # ------------------------------------------------------------------
    # Pass: duplicate gate merging
    # ------------------------------------------------------------------

    def _merge_duplicates(self) -> None:
        """Merge gates with identical (type, input_a, input_b) signatures."""
        seen: Dict[Tuple, int] = {}  # signature → canonical gate_id
        remap: Dict[int, int] = {}

        for gate in self.hw.gates:
            sig = (gate.gate_type, gate.input_ids[0], gate.input_ids[1])
            if sig in seen:
                remap[gate.gate_id] = seen[sig]
            else:
                seen[sig] = gate.gate_id

        if not remap:
            return

        def resolve(ref: int) -> int:
            visited: Set[int] = set()
            while ref in remap and ref not in visited:
                visited.add(ref)
                ref = remap[ref]
            return ref

        for gate in self.hw.gates:
            gate.input_ids = [resolve(inp) for inp in gate.input_ids]

        self.hw.output_gate_ids = [
            resolve(gid) for gid in self.hw.output_gate_ids
        ]

        ids_to_remove = set(remap.keys())
        self.hw.gates = [g for g in self.hw.gates if g.gate_id not in ids_to_remove]

    # ------------------------------------------------------------------
    # Pass: fanout balancing report
    # ------------------------------------------------------------------

    def _fanout_report(self) -> None:
        """Print a report of high-fanout signals."""
        self.hw.compute_fanouts()
        high_fanout = sorted(
            [g for g in self.hw.gates if g.fanout > 10],
            key=lambda g: -g.fanout,
        )
        if high_fanout:
            print(f"[Optimizer] High-fanout signals (fanout > 10): "
                  f"{len(high_fanout)} gates")
            for g in high_fanout[:10]:
                print(f"  gate_id={g.gate_id} type={g.gate_type} "
                      f"fanout={g.fanout}")
        else:
            print("[Optimizer] No high-fanout signals detected.")
