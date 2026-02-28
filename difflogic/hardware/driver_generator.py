"""
Driver generator: produces microcontroller driver code for ESP32 (C/Arduino),
Arduino (.ino sketch) and Raspberry Pi (Python) that match the generated
SPI wrapper protocol exactly.
"""

from __future__ import annotations

from typing import Optional

from .hardware_ir import HardwareModel


class DriverGenerator:
    """Generate MCU driver code for the difflogic SPI interface."""

    def __init__(self, hw: HardwareModel) -> None:
        self.hw = hw
        self.n_in = hw.num_inputs
        self.n_out = len(hw.output_gate_ids)
        self.in_bytes = (self.n_in + 7) // 8
        self.out_bytes = (self.n_out + 7) // 8

    # ------------------------------------------------------------------
    # ESP32 / Arduino C driver
    # ------------------------------------------------------------------

    def generate_esp32_driver(self) -> str:
        """Return a C/C++ ESP32 Arduino driver (.ino) for the SPI wrapper."""
        lines = [
            "// Auto-generated ESP32 SPI driver for difflogic model",
            f"// Inputs : {self.n_in} bits ({self.in_bytes} bytes)",
            f"// Outputs: {self.n_out} bits ({self.out_bytes} bytes)",
            "",
            "#include <Arduino.h>",
            "#include <SPI.h>",
            "",
            "#define MODEL_CS_PIN  5",
            f"#define MODEL_IN_BYTES  {self.in_bytes}",
            f"#define MODEL_OUT_BYTES {self.out_bytes}",
            "",
            "void model_spi_init() {",
            "    SPI.begin();",
            "    pinMode(MODEL_CS_PIN, OUTPUT);",
            "    digitalWrite(MODEL_CS_PIN, HIGH);",
            "    SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));",
            "}",
            "",
            "/**",
            " * Run inference over SPI.",
            " *",
            " * @param input   byte array of length MODEL_IN_BYTES (LSB first)",
            " * @param output  byte array of length MODEL_OUT_BYTES to receive results",
            " */",
            "void model_infer(const uint8_t *input, uint8_t *output) {",
            "    digitalWrite(MODEL_CS_PIN, LOW);",
            "    // Send input bytes",
            "    for (int i = 0; i < MODEL_IN_BYTES; i++) {",
            "        SPI.transfer(input[i]);",
            "    }",
            "    // Read output bytes",
            "    for (int i = 0; i < MODEL_OUT_BYTES; i++) {",
            "        output[i] = SPI.transfer(0x00);",
            "    }",
            "    digitalWrite(MODEL_CS_PIN, HIGH);",
            "}",
            "",
            "// -------- Example sketch --------",
            "void setup() {",
            "    Serial.begin(115200);",
            "    model_spi_init();",
            "}",
            "",
            "void loop() {",
            f"    uint8_t input[MODEL_IN_BYTES]  = {{0}};",
            f"    uint8_t output[MODEL_OUT_BYTES] = {{0}};",
            "",
            "    // TODO: fill input[] with your sensor data",
            "    // Example: set all pixels to 1",
            "    memset(input, 0xFF, MODEL_IN_BYTES);",
            "",
            "    model_infer(input, output);",
            "",
            "    Serial.print(\"Output bytes: \");",
            "    for (int i = 0; i < MODEL_OUT_BYTES; i++) {",
            "        Serial.print(output[i], HEX);",
            "        Serial.print(\" \");",
            "    }",
            "    Serial.println();",
            "",
            "    delay(1000);",
            "}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Arduino (.ino) example – same as ESP32 but uses default SPI pins
    # ------------------------------------------------------------------

    def generate_arduino_example(self) -> str:
        """Return an Arduino sketch example."""
        lines = [
            "// Auto-generated Arduino SPI example for difflogic model",
            f"// Inputs : {self.n_in} bits ({self.in_bytes} bytes)",
            f"// Outputs: {self.n_out} bits ({self.out_bytes} bytes)",
            "",
            "#include <SPI.h>",
            "",
            "const int CS_PIN = 10;",
            f"const int IN_BYTES  = {self.in_bytes};",
            f"const int OUT_BYTES = {self.out_bytes};",
            "",
            "void setup() {",
            "    Serial.begin(9600);",
            "    SPI.begin();",
            "    pinMode(CS_PIN, OUTPUT);",
            "    digitalWrite(CS_PIN, HIGH);",
            "}",
            "",
            "void runInference(const byte* in_buf, byte* out_buf) {",
            "    SPI.beginTransaction(SPISettings(500000, MSBFIRST, SPI_MODE0));",
            "    digitalWrite(CS_PIN, LOW);",
            "    for (int i = 0; i < IN_BYTES; i++)",
            "        SPI.transfer(in_buf[i]);",
            "    for (int i = 0; i < OUT_BYTES; i++)",
            "        out_buf[i] = SPI.transfer(0);",
            "    digitalWrite(CS_PIN, HIGH);",
            "    SPI.endTransaction();",
            "}",
            "",
            "void loop() {",
            f"    byte input[IN_BYTES]  = {{0}};",
            f"    byte output[OUT_BYTES] = {{0}};",
            "",
            "    // TODO: populate input[]",
            "    runInference(input, output);",
            "",
            "    Serial.print(\"Class scores: \");",
            "    for (int i = 0; i < OUT_BYTES; i++) {",
            "        Serial.print(output[i]);",
            "        Serial.print(\" \");",
            "    }",
            "    Serial.println();",
            "    delay(500);",
            "}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Raspberry Pi Python SPI driver
    # ------------------------------------------------------------------

    def generate_raspberry_pi_example(self) -> str:
        """Return a Raspberry Pi Python SPI example using spidev."""
        lines = [
            "#!/usr/bin/env python3",
            "# Auto-generated Raspberry Pi SPI driver for difflogic model",
            f"# Inputs : {self.n_in} bits ({self.in_bytes} bytes)",
            f"# Outputs: {self.n_out} bits ({self.out_bytes} bytes)",
            "",
            "import spidev",
            "import time",
            "",
            f"IN_BYTES  = {self.in_bytes}",
            f"OUT_BYTES = {self.out_bytes}",
            "",
            "spi = spidev.SpiDev()",
            "spi.open(0, 0)          # bus 0, device 0 (CE0)",
            "spi.max_speed_hz = 500000",
            "spi.mode = 0",
            "",
            "",
            "def model_infer(input_bits):",
            "    \"\"\"",
            "    Run inference.",
            "",
            f"    Parameters",
            f"    ----------",
            f"    input_bits : list of int (0/1) of length {self.n_in}",
            "",
            f"    Returns",
            f"    -------",
            f"    list of int (0/1) of length {self.n_out}",
            "    \"\"\"",
            f"    assert len(input_bits) == {self.n_in}",
            "    # Pack bits into bytes (LSB first within each byte)",
            "    in_bytes = []",
            "    for byte_idx in range(IN_BYTES):",
            "        val = 0",
            "        for bit_idx in range(8):",
            "            pos = byte_idx * 8 + bit_idx",
            f"            if pos < {self.n_in} and input_bits[pos]:",
            "                val |= (1 << bit_idx)",
            "        in_bytes.append(val)",
            "",
            "    # SPI transaction: send input, receive output",
            "    tx = in_bytes + [0x00] * OUT_BYTES",
            "    rx = spi.xfer2(tx)",
            "    out_bytes = rx[IN_BYTES:]",
            "",
            "    # Unpack output bytes to bits",
            "    out_bits = []",
            "    for byte_val in out_bytes:",
            "        for bit_idx in range(8):",
            "            out_bits.append((byte_val >> bit_idx) & 1)",
            f"    return out_bits[:{self.n_out}]",
            "",
            "",
            "if __name__ == '__main__':",
            f"    # Example: all-zero input",
            f"    test_input = [0] * {self.n_in}",
            "    result = model_infer(test_input)",
            "    print('Output bits:', result)",
            "    spi.close()",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_all(self, output_dir: str) -> None:
        """Write all driver files to *output_dir*."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        esp32 = self.generate_esp32_driver()
        arduino = self.generate_arduino_example()
        rpi = self.generate_raspberry_pi_example()

        with open(os.path.join(output_dir, "esp32_driver.ino"), "w") as f:
            f.write(esp32)
        with open(os.path.join(output_dir, "arduino_example.ino"), "w") as f:
            f.write(arduino)
        with open(os.path.join(output_dir, "raspberry_pi_example.py"), "w") as f:
            f.write(rpi)
