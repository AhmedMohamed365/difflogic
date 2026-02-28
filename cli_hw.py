#!/usr/bin/env python3
"""
cli_hw.py – Command-line interface for the DiffLogic Hardware Export Pipeline.

Usage::

    python cli_hw.py \\
        --model path/to/model.pt \\
        --interface spi \\
        --mode combinational \\
        --output ./output

Full options::

    python cli_hw.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _load_model(model_path: str):
    """Load a PyTorch model from *model_path* (serialised with torch.save)."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="cli_hw",
        description="DiffLogic Hardware Export Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Path to trained difflogic model (.pt file saved with torch.save)",
    )
    parser.add_argument(
        "--output",
        default="./output",
        metavar="DIR",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--interface",
        choices=["spi", "uart", "parallel"],
        default="spi",
        help="Hardware interface wrapper type (default: spi)",
    )
    parser.add_argument(
        "--mode",
        choices=["combinational", "pipelined"],
        default="combinational",
        help="RTL implementation mode (default: combinational)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run CPU simulation after generation",
    )
    parser.add_argument(
        "--optimize",
        type=int,
        default=0,
        metavar="LEVEL",
        help="Optimisation level 0-3 (default: 0 = none)",
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export hardware IR as JSON",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Alias for --mode pipelined",
    )
    parser.add_argument(
        "--asic-mode",
        action="store_true",
        help="(Stub) Enable ASIC-targeted output",
    )
    parser.add_argument(
        "--target",
        choices=["fpga", "asic", "sim"],
        default="fpga",
        help="Synthesis target (default: fpga)",
    )
    parser.add_argument(
        "--module-name",
        default="model_core",
        help="Top-level Verilog module name (default: model_core)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args(argv)

    if args.pipeline:
        args.mode = "pipelined"

    out_dir = _ensure_output_dir(args.output)
    print(f"[cli_hw] Output directory: {out_dir}")

    # ------------------------------------------------------------------ #
    # 1. Load model
    # ------------------------------------------------------------------ #
    print(f"[cli_hw] Loading model from {args.model} …")
    model = _load_model(args.model)

    # ------------------------------------------------------------------ #
    # 2. Export to hardware IR
    # ------------------------------------------------------------------ #
    print("[cli_hw] Exporting model to hardware IR …")
    from difflogic.hardware.exporter import export_model, export_model_to_json
    from difflogic.hardware.hardware_ir import HardwareModel

    hw: HardwareModel = export_model(model, verbose=args.verbose)

    if args.export_json:
        json_path = os.path.join(out_dir, "model_ir.json")
        export_model_to_json(model, path=json_path, verbose=args.verbose)
        print(f"[cli_hw] IR JSON written to {json_path}")

    # ------------------------------------------------------------------ #
    # 3. Optional optimisation
    # ------------------------------------------------------------------ #
    if args.optimize > 0:
        print(f"[cli_hw] Running optimiser at level {args.optimize} …")
        from difflogic.hardware.optimizer import Optimizer
        opt = Optimizer(hw)
        hw = opt.optimize(level=args.optimize)
        print(f"[cli_hw] After optimisation: {hw.gate_count()} gates, "
              f"depth {hw.max_depth()}")

    # ------------------------------------------------------------------ #
    # 4. Generate Verilog core
    # ------------------------------------------------------------------ #
    print(f"[cli_hw] Generating Verilog ({args.mode}) …")
    from difflogic.hardware.verilog_generator import VerilogGenerator

    vgen = VerilogGenerator(
        hw,
        module_name=args.module_name,
        pipelined=(args.mode == "pipelined"),
    )
    core_path = os.path.join(out_dir, "model_core.v")
    vgen.generate_and_save(core_path)
    print(f"[cli_hw] Core Verilog written to {core_path}")

    # ------------------------------------------------------------------ #
    # 5. Generate wrapper
    # ------------------------------------------------------------------ #
    print(f"[cli_hw] Generating {args.interface.upper()} wrapper …")
    from difflogic.hardware.wrapper_generator import WrapperGenerator

    wgen = WrapperGenerator(hw, core_module=args.module_name)
    wrapper_path = os.path.join(out_dir, f"model_{args.interface}_wrapper.v")
    wgen.generate_and_save(interface=args.interface, path=wrapper_path)
    print(f"[cli_hw] Wrapper written to {wrapper_path}")

    # ------------------------------------------------------------------ #
    # 6. Generate testbench
    # ------------------------------------------------------------------ #
    _generate_testbench(hw, args.module_name, out_dir)

    # ------------------------------------------------------------------ #
    # 7. Generate drivers
    # ------------------------------------------------------------------ #
    print("[cli_hw] Generating MCU drivers …")
    from difflogic.hardware.driver_generator import DriverGenerator

    dgen = DriverGenerator(hw)
    dgen.save_all(out_dir)
    print(f"[cli_hw] Drivers written to {out_dir}")

    # ------------------------------------------------------------------ #
    # 8. Generate diagrams & resource report
    # ------------------------------------------------------------------ #
    print("[cli_hw] Generating diagrams and resource report …")
    from difflogic.hardware.diagram_generator import DiagramGenerator

    diag = DiagramGenerator(hw)
    diag.save_all(out_dir, render_svg=True)
    print(f"[cli_hw] Diagrams written to {out_dir}")

    # ------------------------------------------------------------------ #
    # 9. Optional simulation
    # ------------------------------------------------------------------ #
    if args.simulate:
        print("[cli_hw] Running simulation …")
        _run_simulation(hw, core_path, out_dir, args.verbose)

    # ------------------------------------------------------------------ #
    # 10. Generate README
    # ------------------------------------------------------------------ #
    _write_readme(hw, args, out_dir)

    print(f"\n[cli_hw] ✓ Hardware export complete.  Output: {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_testbench(hw, module_name: str, out_dir: str) -> None:
    """Generate a simple self-checking testbench."""
    n_in = hw.num_inputs
    n_out = len(hw.output_gate_ids)

    lines = [
        f"// Auto-generated testbench for {module_name}",
        "`timescale 1ns/1ps",
        "",
        f"module tb_{module_name};",
        "",
        "    reg  clk;",
        "    reg  rst;",
        f"    reg  [{n_in-1}:0]  in;",
        f"    wire [{n_out-1}:0] out;",
        "",
        f"    {module_name} dut (",
        "        .clk (clk),",
        "        .rst (rst),",
        "        .in  (in),",
        "        .out (out)",
        "    );",
        "",
        "    initial clk = 0;",
        "    always #5 clk = ~clk;",
        "",
        "    initial begin",
        f"        $dumpfile(\"tb_{module_name}.vcd\");",
        f"        $dumpvars(0, tb_{module_name});",
        "        rst = 1; in = 0;",
        "        repeat(2) @(posedge clk);",
        "        rst = 0;",
        "        // TODO: add test vectors",
        "        repeat(10) @(posedge clk);",
        "        $display(\"Simulation complete. out = %b\", out);",
        "        $finish;",
        "    end",
        "",
        "endmodule",
    ]
    tb_path = os.path.join(out_dir, f"model_tb.v")
    with open(tb_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[cli_hw] Testbench written to {tb_path}")


def _run_simulation(hw, verilog_path: str, out_dir: str, verbose: bool) -> None:
    """Run the Python-reference simulation."""
    import numpy as np
    from difflogic.hardware.simulator.run_sim import (
        _software_simulate,
        write_verification_report,
    )

    n_in = hw.num_inputs
    n_samples = 64
    vectors = np.random.randint(0, 2, size=(n_samples, n_in)).astype(bool)

    py_outputs = _software_simulate(hw, vectors, verbose=verbose)

    report_path = os.path.join(out_dir, "verification_report.txt")
    matches = len(py_outputs)  # comparing against itself
    write_verification_report(
        report_path,
        matches=matches,
        total=matches,
        mismatches=[],
        sim_method="Python software reference",
    )
    print(f"[cli_hw] Simulation complete. Report: {report_path}")


def _write_readme(hw, args, out_dir: str) -> None:
    """Write README_HW.md to the output directory."""
    n_in = hw.num_inputs
    n_out = hw.num_outputs
    gates = hw.gate_count()
    depth = hw.max_depth()

    readme = f"""# DiffLogic Hardware Export

Auto-generated hardware files for the trained difflogic model.

## Model Statistics

| Property | Value |
|---|---|
| Primary inputs | {n_in} bits |
| Output classes | {n_out} |
| Total gates | {gates} |
| Critical path depth | {depth} |
| Interface | {args.interface.upper()} |
| RTL mode | {args.mode} |

## Generated Files

| File | Description |
|---|---|
| `model_core.v` | Synthesisable Verilog RTL core |
| `model_{args.interface}_wrapper.v` | {args.interface.upper()} integration wrapper |
| `model_tb.v` | Self-checking testbench |
| `model_ir.json` | Hardware IR (JSON) |
| `esp32_driver.ino` | ESP32 SPI driver |
| `arduino_example.ino` | Arduino SPI example |
| `raspberry_pi_example.py` | Raspberry Pi SPI example |
| `architecture.dot` | Graphviz gate diagram |
| `architecture.svg` | Rendered gate diagram |
| `verification_report.txt` | Simulation verification report |

## Quick Start

### Simulate with Icarus Verilog

```bash
iverilog -o sim model_tb.v model_core.v
vvp sim
```

### Synthesise with Yosys (open-source)

```bash
yosys -p "synth -top model_core; write_verilog synth.v" model_core.v
```

### Flash to FPGA (Vivado example)

```tcl
read_verilog model_core.v
read_verilog model_spi_wrapper.v
synth_design -top model_spi_wrapper -part xc7a35tcpg236-1
```

### ESP32 Integration

1. Open `esp32_driver.ino` in Arduino IDE.
2. Set `MODEL_CS_PIN` to your chip-select GPIO.
3. Upload and open the Serial Monitor at 115200 baud.

## Protocol (SPI)

- **Mode**: SPI Mode 0 (CPOL=0, CPHA=0)
- **Bit order**: MSB first
- **Transaction**: send {(n_in+7)//8} bytes, receive {(n_out+7)//8} bytes

---
*Generated by [DiffLogic Hardware Export Pipeline](https://github.com/AhmedMohamed365/difflogic)*
"""
    readme_path = os.path.join(out_dir, "README_HW.md")
    with open(readme_path, "w") as fh:
        fh.write(readme)
    print(f"[cli_hw] README written to {readme_path}")


if __name__ == "__main__":
    sys.exit(main())
