@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> dims: vec2<i32>; // rows, cols

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = i32(global_id.x); 
    let row = i32(global_id.y); 
    if (col >= dims.y || row >= dims.x) {
        return;
    }
    
    var sum = vec2<f32>(0.0, 0.0);
    let pi = radians(180.0);
    
    // Compute DFT
    for (var x = 0; x < dims.y; x = x + 1) {
        let angle = -2.0 * pi * f32(col * x) / f32(dims.y);
        let euler = vec2<f32>(cos(angle), sin(angle));
        let idx = row * dims.y + x;
        let val = input[idx];

        // Euler rule
        sum = sum + vec2<f32>(
            val.x * euler.x - val.y * euler.y,
            val.x * euler.y + val.y * euler.x
        );
    }
    
    let outIndex = row * dims.y + col;
    output[outIndex] = sum;
}
