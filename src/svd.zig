const c = @import("c_api.zig");
const core = @import("core.zig");
const Mat = core.Mat;

pub fn svdCompute(
    src: Mat,
    w: *Mat,
    u: *Mat,
    vt: *Mat,
) void {
    c.SVD_Compute(
        src.ptr,
        w.*.ptr,
        u.*.ptr,
        vt.*.ptr,
    );
}

//*    implementation done
//*    pub extern fn SVD_Compute(src: Mat, w: Mat, u: Mat, vt: Mat) void;

test "svd" {
    const std = @import("std");
    const testing = std.testing;

    const result_w = [2]f32{ 6.167493, 3.8214223 };
    const result_u = [4]f32{ -0.1346676, -0.99089086, 0.9908908, -0.1346676 };
    const result_vt = [4]f32{ 0.01964448, 0.999807, -0.999807, 0.01964448 };

    var src = Mat.initSize(2, 2, .cv32fc1);
    src.set(f32, 0, 0, 3.76956568);
    src.set(f32, 0, 1, -0.90478725);
    src.set(f32, 1, 0, 0.634576);
    src.set(f32, 1, 1, 6.10002347);

    var w = Mat.init();
    defer w.deinit();

    var u = Mat.init();
    defer u.deinit();

    var vt = Mat.init();
    defer vt.deinit();

    svdCompute(src, &w, &u, &vt);

    var data_w = try w.dataPtr(f32);
    var data_u = try u.dataPtr(f32);
    var data_vt = try vt.dataPtr(f32);

    try testing.expectEqual(data_w.len, result_w.len);
    try testing.expectEqual(data_u.len, result_u.len);
    try testing.expectEqual(data_vt.len, result_vt.len);

    for (data_w) |value, i| {
        try testing.expectEqual(value, result_w[i]);
    }

    for (data_u) |value, i| {
        try testing.expectEqual(value, result_u[i]);
    }

    for (data_vt) |value, i| {
        try testing.expectEqual(value, result_vt[i]);
    }
}
