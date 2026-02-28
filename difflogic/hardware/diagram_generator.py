"""
Diagram generator: produces Graphviz (DOT) and Matplotlib-based diagrams
for the hardware IR.

Generates:
  - Gate-level graph (Graphviz DOT / SVG / PNG / PDF)
  - Layered architecture diagram (ASCII or Graphviz)
  - Resource estimation summary (text)
"""

from __future__ import annotations

import os
from typing import List, Optional

from .hardware_ir import HardwareModel


class DiagramGenerator:
    """Generate diagrams and resource summaries for a HardwareModel."""

    def __init__(self, hw: HardwareModel) -> None:
        self.hw = hw

    # ------------------------------------------------------------------
    # Graphviz DOT generation
    # ------------------------------------------------------------------

    def generate_dot(self, max_gates: int = 200) -> str:
        """Return a Graphviz DOT string for the gate-level graph.

        To keep diagrams readable, only the first *max_gates* gates are
        included when the network is large.
        """
        hw = self.hw
        lines: List[str] = []
        lines.append("digraph model {")
        lines.append("    rankdir=LR;")
        lines.append('    node [shape=box fontname="Courier"];')

        # Primary inputs
        for i in range(hw.num_inputs):
            lines.append(f'    in_{i} [label="IN[{i}]" shape=circle style=filled fillcolor=lightblue];')

        gates_to_show = hw.gates[:max_gates]
        truncated = len(hw.gates) > max_gates

        # Gate nodes
        for gate in gates_to_show:
            color = "lightyellow"
            if gate.gate_type in ("AND", "NOT_AND"):
                color = "lightgreen"
            elif gate.gate_type in ("OR", "NOT_OR"):
                color = "lightyellow"
            elif gate.gate_type in ("XOR", "NOT_XOR"):
                color = "lightsalmon"
            label = f"{gate.gate_type}\\nid={gate.gate_id}\\nd={gate.depth}"
            lines.append(
                f'    g{gate.gate_id} [label="{label}" style=filled fillcolor={color}];'
            )

        # Edges
        for gate in gates_to_show:
            for inp in gate.input_ids:
                if inp < 0:
                    pi_idx = -(inp + 1)
                    lines.append(f"    in_{pi_idx} -> g{gate.gate_id};")
                else:
                    lines.append(f"    g{inp} -> g{gate.gate_id};")

        # Output nodes
        for bit_idx, gid in enumerate(hw.output_gate_ids):
            lines.append(
                f'    out_{bit_idx} [label="OUT[{bit_idx}]" shape=circle style=filled fillcolor=lightcoral];'
            )
            lines.append(f"    g{gid} -> out_{bit_idx};")

        if truncated:
            lines.append(
                f'    truncated [label="... {len(hw.gates) - max_gates} more gates ..." '
                f'shape=note style=filled fillcolor=lightgrey];'
            )

        lines.append("}")
        return "\n".join(lines)

    def save_dot(self, path: str, max_gates: int = 200) -> None:
        """Save the DOT source to *path*."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.generate_dot(max_gates=max_gates))

    def render_diagram(
        self,
        output_path: str,
        fmt: str = "svg",
        max_gates: int = 200,
    ) -> bool:
        """Render the gate diagram using Graphviz (if available).

        Returns True on success, False if Graphviz is not installed.
        """
        import subprocess
        import tempfile

        dot_src = self.generate_dot(max_gates=max_gates)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dot", delete=False
        ) as tmp:
            tmp.write(dot_src)
            dot_path = tmp.name

        try:
            result = subprocess.run(
                ["dot", f"-T{fmt}", dot_path, "-o", output_path],
                capture_output=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        finally:
            os.unlink(dot_path)

    # ------------------------------------------------------------------
    # Architecture layered diagram (ASCII)
    # ------------------------------------------------------------------

    def generate_architecture_ascii(self) -> str:
        """Return a simple ASCII architecture overview."""
        hw = self.hw
        layers_map = hw.gates_by_layer()
        num_layers = max(layers_map.keys()) + 1 if layers_map else 0

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  DiffLogic Hardware Architecture")
        lines.append("=" * 60)
        lines.append(f"  Inputs      : {hw.num_inputs} bits")
        lines.append(f"  Outputs     : {hw.num_outputs} classes")
        lines.append(f"  Total Gates : {hw.gate_count()}")
        lines.append(f"  Depth       : {hw.max_depth()}")
        lines.append(f"  Layers      : {num_layers}")
        lines.append("-" * 60)
        lines.append(f"  INPUT [{hw.num_inputs}b]")
        lines.append("       |")
        for li in sorted(layers_map.keys()):
            count = len(layers_map[li])
            lines.append(f"  [Layer {li}]  {count} gates")
            lines.append("       |")
        lines.append(f"  OUTPUT [{hw.num_outputs} classes]")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Resource estimation report
    # ------------------------------------------------------------------

    def generate_resource_report(self) -> str:
        """Return a human-readable resource estimation text."""
        hw = self.hw
        summary = hw.resource_summary()
        layers_map = hw.gates_by_layer()

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  DiffLogic Hardware Resource Estimation")
        lines.append("=" * 60)
        lines.append(f"  Primary inputs       : {summary['num_inputs']}")
        lines.append(f"  Primary outputs      : {summary['num_outputs']}")
        lines.append(f"  Total gates          : {summary['total_gates']}")
        lines.append(f"  Critical path depth  : {summary['critical_path_depth']}")
        lines.append(f"  Pipeline stages      : {len(layers_map)}")
        lines.append("")
        lines.append("  Gate type breakdown:")
        for k, v in sorted(summary.items()):
            if k.startswith("gate_"):
                gate_name = k[5:].upper()
                lines.append(f"    {gate_name:<20} {v:>6}")
        lines.append("")
        # Rough LUT estimate: each gate ≈ 1 LUT-4
        lut_estimate = summary["total_gates"]
        lines.append(f"  Estimated LUT-4 usage: ~{lut_estimate}")
        # FF count only if pipelined
        lines.append(f"  Flip-flops (pipelined): ~{summary['total_gates']}")
        lines.append("")
        lines.append("  Note: Estimates are approximate pre-synthesis figures.")
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_resource_report(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.generate_resource_report())

    def save_all(
        self,
        output_dir: str,
        *,
        render_svg: bool = True,
        max_gates: int = 200,
    ) -> None:
        """Write all diagram artefacts to *output_dir*."""
        os.makedirs(output_dir, exist_ok=True)

        dot_path = os.path.join(output_dir, "architecture.dot")
        self.save_dot(dot_path, max_gates=max_gates)

        if render_svg:
            svg_path = os.path.join(output_dir, "architecture.svg")
            ok = self.render_diagram(svg_path, fmt="svg", max_gates=max_gates)
            if not ok:
                # Fall back to writing DOT only
                pass

        report_path = os.path.join(output_dir, "verification_report.txt")
        self.save_resource_report(report_path)

        ascii_path = os.path.join(output_dir, "architecture_ascii.txt")
        with open(ascii_path, "w", encoding="utf-8") as fh:
            fh.write(self.generate_architecture_ascii())
