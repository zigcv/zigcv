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
