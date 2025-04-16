@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> dims: vec2<i32>; // rows & cols

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // u and v represent indices in original matrix
    let u: i32 = i32(global_id.x);
    let v: i32 = i32(global_id.y);

    // dims are the rows/cols
    if (u >= dims.y || v >= dims.x) {
        return;
    }

    var sum: vec2<f32> = vec2<f32>(0.0, 0.0);
    let pi = radians(180.0);

    // loop through all elements
    for (var y: i32 = 0; y < dims.x; y = y + 1) {
        for (var x: i32 = 0; x < dims.y; x = x + 1) {
            // -2π * ( (u*x)/cols + (v*y)/rows )
            let angle: f32 = -2 * pi * ((f32(u * x) / f32(dims.y)) + (f32(v * y) / f32(dims.x)));
            let c: vec2<f32> = vec2<f32>(cos(angle), sin(angle));

            // find original index 
            let index: i32 = y * dims.y + x;
            let a: vec2<f32> = input[index];
            
            // euler
            sum = sum + vec2<f32>(
                a.x * c.x - a.y * c.y,
                a.x * c.y + a.y * c.x
            );
        }
    }

    let outIndex: i32 = v * dims.y + u;
    output[outIndex] = sum;
}
