// testbench.v.tpl – self-checking testbench template (documentation reference)
//
// Variables:
//   {{ module_name }}  – DUT module name
//   {{ n_in }}         – number of inputs
//   {{ n_out }}        – number of outputs

// Auto-generated testbench for {{ module_name }}
`timescale 1ns/1ps

module tb_{{ module_name }};

    reg         clk;
    reg         rst;
    reg  [{{ n_in - 1 }}:0]  in;
    wire [{{ n_out - 1 }}:0] out;

    // Instantiate DUT
    {{ module_name }} dut (
        .clk (clk),
        .rst (rst),
        .in  (in),
        .out (out)
    );

    // 10 ns clock
    initial clk = 0;
    always #5 clk = ~clk;

    integer i;
    integer errors = 0;

    task apply_and_check;
        input [{{ n_in - 1 }}:0] test_in;
        input [{{ n_out - 1 }}:0] expected_out;
        begin
            in = test_in;
            @(posedge clk); #1;
            if (out !== expected_out) begin
                $display("FAIL: in=%b expected=%b got=%b",
                         test_in, expected_out, out);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("tb_{{ module_name }}.vcd");
        $dumpvars(0, tb_{{ module_name }});

        rst = 1;
        in  = 0;
        repeat(2) @(posedge clk);
        rst = 0;

        // TODO: add test vectors here
        // apply_and_check({{ n_in }}'b0, {{ n_out }}'b0);

        repeat(5) @(posedge clk);

        if (errors == 0)
            $display("PASS: all tests passed.");
        else
            $display("FAIL: %0d errors.", errors);

        $finish;
    end

endmodule
