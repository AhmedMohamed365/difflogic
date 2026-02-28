// Verilator C++ simulation bridge for difflogic model
//
// This file is compiled by Verilator together with the generated Verilog.
// It provides a simple C API that run_sim.py calls via ctypes:
//
//   void sim_init()
//   void sim_reset()
//   void sim_set_input(const uint8_t* bits, int n_bits)
//   void sim_step()
//   void sim_get_output(uint8_t* bits, int n_bits)
//   void sim_close()
//
// Build with:
//   verilator --cc model_core.v --exe verilator_bridge.cpp --build -j4
//   or via run_sim.py which calls verilator automatically.

#include "Vmodel_core.h"
#include "verilated.h"
#include <cstdint>
#include <cstring>

static Vmodel_core* top = nullptr;
static VerilatedContext* ctx = nullptr;

extern "C" {

void sim_init() {
    ctx = new VerilatedContext;
    ctx->commandArgs(0, nullptr);
    top = new Vmodel_core{ctx};
}

void sim_reset() {
    if (!top) return;
    top->rst = 1;
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
    top->rst = 0;
}

void sim_set_input(const uint8_t* bits, int n_bits) {
    if (!top) return;
    // The 'in' port width may exceed 64 bits; Verilator represents wide
    // ports as WData (uint32_t array). We write byte-by-byte using the
    // Verilator VL_ASSIGN_W macro approach – here we use the simple
    // single-word path for ports <= 64 bits and a loop for wider ports.
    //
    // For MNIST 784 inputs the port is 784 bits wide.
    // Verilator exposes `top->in` as `WData in[25]` (784/32 = 25 words).

#if defined(VM_BITS_IN)
    // Wide port: zero fill then pack bits
    memset(top->in, 0, sizeof(top->in));
    for (int i = 0; i < n_bits && i < VM_BITS_IN; i++) {
        int word = i / 32;
        int bit  = i % 32;
        if (bits[i])
            top->in[word] |= (1u << bit);
    }
#else
    // Narrow port (fits in uint64_t or less)
    uint64_t val = 0;
    for (int i = 0; i < n_bits && i < 64; i++) {
        if (bits[i]) val |= (1ULL << i);
    }
    top->in = (decltype(top->in)) val;
#endif
}

void sim_step() {
    if (!top) return;
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
}

void sim_get_output(uint8_t* bits, int n_bits) {
    if (!top) return;
#if defined(VM_BITS_OUT)
    memset(bits, 0, n_bits);
    for (int i = 0; i < n_bits && i < VM_BITS_OUT; i++) {
        int word = i / 32;
        int bit  = i % 32;
        bits[i] = (top->out[word] >> bit) & 1;
    }
#else
    uint64_t val = (uint64_t) top->out;
    for (int i = 0; i < n_bits && i < 64; i++) {
        bits[i] = (val >> i) & 1;
    }
#endif
}

void sim_close() {
    if (top) { top->final(); delete top; top = nullptr; }
    if (ctx) { delete ctx; ctx = nullptr; }
}

} // extern "C"
