"""
Model exporter: converts a trained difflogic Sequential model into a
HardwareModel (IR) and optionally serialises it as a JSON file.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..difflogic import LogicLayer, GroupSum
from .hardware_ir import ALL_OPERATIONS, HardwareModel, LogicGate


def export_model(
    model: nn.Sequential,
    *,
    verbose: bool = False,
) -> HardwareModel:
    """Extract a trained difflogic model into a :class:`HardwareModel`.

    Parameters
    ----------
    model:
        A trained ``torch.nn.Sequential`` whose layers are a sequence of
        :class:`~difflogic.LogicLayer` objects followed by a final
        :class:`~difflogic.GroupSum`.
    verbose:
        Print diagnostic information during extraction.

    Returns
    -------
    HardwareModel
        The hardware IR ready for Verilog generation or optimisation.
    """
    # Validate structure
    assert isinstance(model[-1], GroupSum), (
        "Last layer must be GroupSum, got {}".format(type(model[-1]))
    )

    num_classes: int = model[-1].k
    num_inputs: Optional[int] = None
    logic_layers = []

    for layer in model:
        if isinstance(layer, LogicLayer):
            if num_inputs is None:
                num_inputs = layer.in_dim
            logic_layers.append(layer)
        elif isinstance(layer, (GroupSum, nn.Flatten)):
            pass
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    assert num_inputs is not None, "No LogicLayer found in model"

    if verbose:
        print(f"Exporting model: {num_inputs} inputs, {num_classes} classes, "
              f"{len(logic_layers)} logic layers")

    hw = HardwareModel(num_inputs=num_inputs, num_outputs=num_classes)

    # Global gate id counter; primary inputs are encoded as -(i+1).
    next_gate_id = 0
    # Map from (layer_id, local_index) → global gate_id
    prev_layer_ids = list(range(num_inputs))  # will hold gate_ids per position

    for layer_idx, layer in enumerate(logic_layers):
        indices_a = layer.indices[0].cpu().tolist()
        indices_b = layer.indices[1].cpu().tolist()
        gate_ops = layer.weights.argmax(-1).cpu().tolist()

        out_dim = layer.out_dim
        current_layer_ids = []

        for local_id in range(out_dim):
            op_idx = int(gate_ops[local_id])
            gate_type = ALL_OPERATIONS[op_idx]

            ia = int(indices_a[local_id])
            ib = int(indices_b[local_id])

            # Map to global IDs.  For layer 0 the "previous layer" is the
            # primary input bus, encoded as -(i+1).
            if layer_idx == 0:
                ref_a = -(ia + 1)
                ref_b = -(ib + 1)
            else:
                ref_a = prev_layer_ids[ia]
                ref_b = prev_layer_ids[ib]

            gate = LogicGate(
                gate_id=next_gate_id,
                gate_type=gate_type,
                input_ids=[ref_a, ref_b],
                layer=layer_idx,
            )
            hw.add_gate(gate)
            current_layer_ids.append(next_gate_id)
            next_gate_id += 1

        prev_layer_ids = current_layer_ids

        if verbose:
            print(f"  Layer {layer_idx}: {out_dim} gates extracted")

    # Last layer outputs: one per neuron.  GroupSum sums these per class.
    last_layer = logic_layers[-1]
    out_dim_last = last_layer.out_dim
    num_out_per_class = out_dim_last // num_classes
    hw.num_out_per_class = num_out_per_class

    # All last-layer gate ids are the outputs
    hw.output_gate_ids = list(
        range(next_gate_id - out_dim_last, next_gate_id)
    )

    hw.compute_depths()
    hw.compute_fanouts()

    if verbose:
        summary = hw.resource_summary()
        print("Resource summary:", summary)

    return hw


def export_model_to_json(
    model: nn.Sequential,
    path: Optional[str] = None,
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Export model to a JSON-serialisable dictionary (and optionally save it).

    The JSON schema is::

        {
            "inputs": N,
            "outputs": C,
            "num_out_per_class": K,
            "gates": [
                {"id": 0, "type": "AND", "inputs": [-1, -2], "layer": 0}
            ]
        }

    Primary-input references follow the ``-(i+1)`` convention used internally.
    """
    hw = export_model(model, verbose=verbose)

    data: Dict[str, Any] = {
        "inputs": hw.num_inputs,
        "outputs": hw.num_outputs,
        "num_out_per_class": hw.num_out_per_class,
        "output_gate_ids": hw.output_gate_ids,
        "gates": [
            {
                "id": g.gate_id,
                "type": g.gate_type,
                "inputs": g.input_ids,
                "layer": g.layer,
                "depth": g.depth,
            }
            for g in hw.gates
        ],
    }

    if path is not None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        if verbose:
            print(f"IR JSON written to {path}")

    return data


def hardware_model_from_json(data: Dict[str, Any]) -> HardwareModel:
    """Reconstruct a :class:`HardwareModel` from the JSON dictionary produced
    by :func:`export_model_to_json`."""
    hw = HardwareModel(
        num_inputs=data["inputs"],
        num_outputs=data["outputs"],
    )
    hw.num_out_per_class = data.get("num_out_per_class", 1)
    hw.output_gate_ids = data["output_gate_ids"]

    for g in data["gates"]:
        hw.add_gate(
            LogicGate(
                gate_id=g["id"],
                gate_type=g["type"],
                input_ids=g["inputs"],
                layer=g.get("layer", 0),
                depth=g.get("depth", 0),
            )
        )

    hw.compute_fanouts()
    return hw
