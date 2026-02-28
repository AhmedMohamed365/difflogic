// spi_wrapper.v.tpl – SPI slave wrapper template (documentation reference)
//
// Variables:
//   {{ core_module }}  – core module name to instantiate
//   {{ n_in }}         – number of primary inputs (bits)
//   {{ n_out }}        – number of primary outputs (bits)
//   {{ in_bytes }}     – ceil(n_in / 8)
//   {{ out_bytes }}    – ceil(n_out / 8)

// Auto-generated SPI wrapper for {{ core_module }}
// Inputs : {{ n_in }} bits ({{ in_bytes }} bytes)
// Outputs: {{ n_out }} bits ({{ out_bytes }} bytes)

module model_spi_wrapper (
    input  wire clk,
    input  wire rst,
    input  wire spi_clk,
    input  wire spi_mosi,
    output reg  spi_miso,
    input  wire spi_cs_n
);

    reg  [{{ n_in - 1 }}:0]  model_in;
    wire [{{ n_out - 1 }}:0] model_out;

    reg [7:0] byte_cnt;
    reg [2:0] bit_cnt;
    reg [7:0] shift_in;
    reg [7:0] shift_out;

    localparam IN_BYTES  = {{ in_bytes }};
    localparam OUT_BYTES = {{ out_bytes }};

    {{ core_module }} core_inst (
        .clk (clk),
        .rst (rst),
        .in  (model_in),
        .out (model_out)
    );

    // SPI receive logic
    always @(posedge spi_clk or posedge rst) begin
        if (rst) begin
            byte_cnt <= 8'd0;
            bit_cnt  <= 3'd0;
        end else if (!spi_cs_n) begin
            shift_in <= {shift_in[6:0], spi_mosi};
            if (bit_cnt == 3'd7) begin
                bit_cnt <= 3'd0;
                if (byte_cnt < IN_BYTES)
                    model_in[byte_cnt*8 +: 8] <= {shift_in[6:0], spi_mosi};
                byte_cnt <= byte_cnt + 1;
            end else
                bit_cnt <= bit_cnt + 1;
        end
    end

    always @(posedge spi_cs_n or posedge rst)
        byte_cnt <= 8'd0;

endmodule
