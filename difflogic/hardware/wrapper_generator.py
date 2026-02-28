"""
Wrapper generator: produces SPI, UART and parallel integration wrappers
around the generated model_core Verilog module.
"""

from __future__ import annotations

from typing import Optional

from .hardware_ir import HardwareModel


class WrapperGenerator:
    """Generate hardware interface wrappers for the difflogic core."""

    def __init__(self, hw: HardwareModel, core_module: str = "model_core") -> None:
        self.hw = hw
        self.core_module = core_module

    # ------------------------------------------------------------------
    # SPI Wrapper
    # ------------------------------------------------------------------

    def generate_spi_wrapper(self) -> str:
        """Generate an SPI slave wrapper that feeds inputs to the core.

        Protocol
        --------
        1. Master sends ``ceil(N/8)`` bytes (input data, LSB first).
        2. Core evaluates combinationally.
        3. Master reads ``ceil(C/8)`` bytes (class scores as raw bit vectors).
        """
        hw = self.hw
        n_in = hw.num_inputs
        n_out = len(hw.output_gate_ids)
        in_bytes = (n_in + 7) // 8
        out_bytes = (n_out + 7) // 8

        lines = [
            f"// Auto-generated SPI wrapper for {self.core_module}",
            f"// Inputs: {n_in} bits ({in_bytes} bytes)",
            f"// Outputs: {n_out} bits ({out_bytes} bytes)",
            "",
            "module model_spi_wrapper (",
            "    input  wire clk,",
            "    input  wire rst,",
            "    // SPI interface",
            "    input  wire spi_clk,",
            "    input  wire spi_mosi,",
            "    output reg  spi_miso,",
            "    input  wire spi_cs_n",
            ");",
            "",
            f"    // Input/output shift registers",
            f"    reg [{n_in-1}:0]  model_in;",
            f"    wire [{n_out-1}:0] model_out;",
            "",
            f"    // SPI state machine",
            f"    reg [7:0]  byte_cnt;",
            f"    reg [2:0]  bit_cnt;",
            f"    reg [7:0]  shift_in;",
            f"    reg [7:0]  shift_out;",
            "",
            f"    localparam IN_BYTES  = {in_bytes};",
            f"    localparam OUT_BYTES = {out_bytes};",
            "",
            "    // Instantiate core",
            f"    {self.core_module} core_inst (",
            "        .clk (clk),",
            "        .rst (rst),",
            "        .in  (model_in),",
            "        .out (model_out)",
            "    );",
            "",
            "    // SPI receive and transmit",
            "    always @(posedge spi_clk or posedge rst) begin",
            "        if (rst) begin",
            "            byte_cnt <= 8'd0;",
            "            bit_cnt  <= 3'd0;",
            "            shift_in <= 8'd0;",
            "        end else if (!spi_cs_n) begin",
            "            shift_in <= {shift_in[6:0], spi_mosi};",
            "            if (bit_cnt == 3'd7) begin",
            "                bit_cnt <= 3'd0;",
            "                if (byte_cnt < IN_BYTES) begin",
            f"                    model_in[byte_cnt*8 +: 8] <= {{shift_in[6:0], spi_mosi}};",
            "                end",
            "                byte_cnt <= byte_cnt + 1;",
            "            end else begin",
            "                bit_cnt <= bit_cnt + 1;",
            "            end",
            "        end",
            "    end",
            "",
            "    always @(negedge spi_clk or posedge rst) begin",
            "        if (rst) begin",
            "            spi_miso  <= 1'b0;",
            "            shift_out <= 8'd0;",
            "        end else if (!spi_cs_n) begin",
            "            if (byte_cnt >= IN_BYTES) begin",
            "                reg [2:0] out_byte_idx;",
            "                out_byte_idx = byte_cnt - IN_BYTES;",
            f"                shift_out <= model_out[out_byte_idx*8 +: 8];",
            "                spi_miso  <= shift_out[7];",
            "                shift_out <= {shift_out[6:0], 1'b0};",
            "            end",
            "        end",
            "    end",
            "",
            "    // Reset byte counter on CS deassert",
            "    always @(posedge spi_cs_n or posedge rst) begin",
            "        byte_cnt <= 8'd0;",
            "        bit_cnt  <= 3'd0;",
            "    end",
            "",
            "endmodule",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # UART Wrapper
    # ------------------------------------------------------------------

    def generate_uart_wrapper(self, baud_divider: int = 868) -> str:
        """Generate a simple UART wrapper (8N1, configurable baud divider).

        The *baud_divider* should equal ``clk_freq / baud_rate``.
        Default is 868 for a 100 MHz clock and 115200 baud.
        """
        hw = self.hw
        n_in = hw.num_inputs
        n_out = len(hw.output_gate_ids)
        in_bytes = (n_in + 7) // 8
        out_bytes = (n_out + 7) // 8

        lines = [
            f"// Auto-generated UART wrapper for {self.core_module}",
            f"// Protocol: send {in_bytes} bytes (8N1), receive {out_bytes} bytes",
            f"// Baud divider: {baud_divider}",
            "",
            "module model_uart_wrapper (",
            "    input  wire clk,",
            "    input  wire rst,",
            "    input  wire uart_rx,",
            "    output reg  uart_tx",
            ");",
            "",
            f"    reg [{n_in-1}:0]  model_in;",
            f"    wire [{n_out-1}:0] model_out;",
            "",
            f"    localparam BAUD_DIV  = {baud_divider};",
            f"    localparam IN_BYTES  = {in_bytes};",
            f"    localparam OUT_BYTES = {out_bytes};",
            "",
            "    // Instantiate core",
            f"    {self.core_module} core_inst (",
            "        .clk (clk),",
            "        .rst (rst),",
            "        .in  (model_in),",
            "        .out (model_out)",
            "    );",
            "",
            "    // ---- RX state machine ----",
            "    reg [15:0] rx_baud_cnt;",
            "    reg [3:0]  rx_bit_cnt;",
            "    reg [7:0]  rx_shift;",
            "    reg [7:0]  rx_byte_idx;",
            "    reg        rx_active;",
            "",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
            "            rx_baud_cnt <= 0; rx_bit_cnt <= 0;",
            "            rx_active   <= 0; rx_byte_idx <= 0;",
            "            model_in    <= 0;",
            "        end else begin",
            "            if (!rx_active && !uart_rx) begin // start bit",
            "                rx_active   <= 1;",
            "                rx_baud_cnt <= BAUD_DIV / 2;",
            "                rx_bit_cnt  <= 0;",
            "            end else if (rx_active) begin",
            "                if (rx_baud_cnt == 0) begin",
            "                    rx_baud_cnt <= BAUD_DIV;",
            "                    if (rx_bit_cnt < 8) begin",
            "                        rx_shift    <= {uart_rx, rx_shift[7:1]};",
            "                        rx_bit_cnt  <= rx_bit_cnt + 1;",
            "                    end else begin // stop bit",
            "                        rx_active <= 0;",
            f"                        model_in[rx_byte_idx*8 +: 8] <= rx_shift;",
            "                        if (rx_byte_idx == IN_BYTES - 1)",
            "                            rx_byte_idx <= 0;",
            "                        else",
            "                            rx_byte_idx <= rx_byte_idx + 1;",
            "                    end",
            "                end else",
            "                    rx_baud_cnt <= rx_baud_cnt - 1;",
            "            end",
            "        end",
            "    end",
            "",
            "    // ---- TX state machine (send results after full frame) ----",
            "    reg [15:0] tx_baud_cnt;",
            "    reg [3:0]  tx_bit_cnt;",
            "    reg [7:0]  tx_shift;",
            "    reg [7:0]  tx_byte_idx;",
            "    reg        tx_active;",
            "",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
            "            tx_baud_cnt <= 0; tx_bit_cnt <= 0;",
            "            tx_active   <= 0; tx_byte_idx <= 0;",
            "            uart_tx     <= 1;",
            "        end else begin",
            "            if (!tx_active) begin",
            "                uart_tx   <= 1;",
            "            end else begin",
            "                if (tx_baud_cnt == 0) begin",
            "                    tx_baud_cnt <= BAUD_DIV;",
            "                    if (tx_bit_cnt == 0) begin // start bit",
            "                        uart_tx    <= 0;",
            "                        tx_bit_cnt <= 1;",
            f"                        tx_shift   <= model_out[tx_byte_idx*8 +: 8];",
            "                    end else if (tx_bit_cnt <= 8) begin",
            "                        uart_tx    <= tx_shift[0];",
            "                        tx_shift   <= {1'b0, tx_shift[7:1]};",
            "                        tx_bit_cnt <= tx_bit_cnt + 1;",
            "                    end else begin // stop bit",
            "                        uart_tx    <= 1;",
            "                        tx_bit_cnt <= 0;",
            "                        if (tx_byte_idx == OUT_BYTES - 1) begin",
            "                            tx_active   <= 0;",
            "                            tx_byte_idx <= 0;",
            "                        end else",
            "                            tx_byte_idx <= tx_byte_idx + 1;",
            "                    end",
            "                end else",
            "                    tx_baud_cnt <= tx_baud_cnt - 1;",
            "            end",
            "        end",
            "    end",
            "",
            "endmodule",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parallel Wrapper
    # ------------------------------------------------------------------

    def generate_parallel_wrapper(self) -> str:
        """Generate a simple parallel (registered I/O) wrapper."""
        hw = self.hw
        n_in = hw.num_inputs
        n_out = len(hw.output_gate_ids)

        lines = [
            f"// Auto-generated parallel wrapper for {self.core_module}",
            "",
            "module model_parallel_wrapper (",
            "    input  wire        clk,",
            "    input  wire        rst,",
            f"    input  wire [{n_in-1}:0]  in,",
            "    input  wire        load,",
            f"    output reg  [{n_out-1}:0] out,",
            "    output reg         valid",
            ");",
            "",
            f"    reg  [{n_in-1}:0]  model_in;",
            f"    wire [{n_out-1}:0] model_out;",
            "",
            "    // Instantiate core",
            f"    {self.core_module} core_inst (",
            "        .clk (clk),",
            "        .rst (rst),",
            "        .in  (model_in),",
            "        .out (model_out)",
            "    );",
            "",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
            "            model_in <= 0;",
            "            out      <= 0;",
            "            valid    <= 0;",
            "        end else begin",
            "            valid <= 0;",
            "            if (load) begin",
            "                model_in <= in;",
            "                out      <= model_out;",
            "                valid    <= 1;",
            "            end",
            "        end",
            "    end",
            "",
            "endmodule",
        ]
        return "\n".join(lines)

    def generate_and_save(
        self,
        interface: str = "spi",
        path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate the requested wrapper and optionally write to *path*."""
        interface = interface.lower()
        if interface == "spi":
            code = self.generate_spi_wrapper()
        elif interface == "uart":
            code = self.generate_uart_wrapper(**kwargs)
        elif interface == "parallel":
            code = self.generate_parallel_wrapper()
        else:
            raise ValueError(f"Unknown interface: {interface}. "
                             "Choose from spi, uart, parallel.")

        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(code)
        return code
