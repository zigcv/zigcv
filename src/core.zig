const c = @import("c_api.zig");
const std = @import("std");

pub const MatChannels = enum(i32) {
    // MatChannels1 is a single channel Mat.
    MatChannels1 = 0,

    // MatChannels2 is 2 channel Mat.
    MatChannels2 = 8,

    // MatChannels3 is 3 channel Mat.
    MatChannels3 = 16,

    // MatChannels4 is 4 channel Mat.
    MatChannels4 = 24,
};

pub const MatType = enum(i32) {
    // MatTypeCV8U is a Mat of 8-bit unsigned int
    MatTypeCV8U = 0,

    // MatTypeCV8S is a Mat of 8-bit signed int
    MatTypeCV8S = 1,

    // MatTypeCV16U is a Mat of 16-bit unsigned int
    MatTypeCV16U = 2,

    // MatTypeCV16S is a Mat of 16-bit signed int
    MatTypeCV16S = 3,

    // MatTypeCV16SC2 is a Mat of 16-bit signed int with 2 channels
    MatTypeCV16SC2 = MatType.MatTypeCV16S + MatChannels.MatChannels2,

    // MatTypeCV32S is a Mat of 32-bit signed int
    MatTypeCV32S = 4,

    // MatTypeCV32F is a Mat of 32-bit float
    MatTypeCV32F = 5,

    // MatTypeCV64F is a Mat of 64-bit float
    MatTypeCV64F = 6,

    // MatTypeCV8UC1 is a Mat of 8-bit unsigned int with a single channel
    MatTypeCV8UC1 = MatType.MatTypeCV8U + MatChannels.MatChannels1,

    // MatTypeCV8UC2 is a Mat of 8-bit unsigned int with 2 channels
    MatTypeCV8UC2 = MatType.MatTypeCV8U + MatChannels.MatChannels2,

    // MatTypeCV8UC3 is a Mat of 8-bit unsigned int with 3 channels
    MatTypeCV8UC3 = MatType.MatTypeCV8U + MatChannels.MatChannels3,

    // MatTypeCV8UC4 is a Mat of 8-bit unsigned int with 4 channels
    MatTypeCV8UC4 = MatType.MatTypeCV8U + MatChannels.MatChannels4,

    // MatTypeCV8SC1 is a Mat of 8-bit signed int with a single channel
    MatTypeCV8SC1 = MatType.MatTypeCV8S + MatChannels.MatChannels1,

    // MatTypeCV8SC2 is a Mat of 8-bit signed int with 2 channels
    MatTypeCV8SC2 = MatType.MatTypeCV8S + MatChannels.MatChannels2,

    // MatTypeCV8SC3 is a Mat of 8-bit signed int with 3 channels
    MatTypeCV8SC3 = MatType.MatTypeCV8S + MatChannels.MatChannels3,

    // MatTypeCV8SC4 is a Mat of 8-bit signed int with 4 channels
    MatTypeCV8SC4 = MatType.MatTypeCV8S + MatChannels.MatChannels4,

    // MatTypeCV16UC1 is a Mat of 16-bit unsigned int with a single channel
    MatTypeCV16UC1 = MatType.MatTypeCV16U + MatChannels.MatChannels1,

    // MatTypeCV16UC2 is a Mat of 16-bit unsigned int with 2 channels
    MatTypeCV16UC2 = MatType.MatTypeCV16U + MatChannels.MatChannels2,

    // MatTypeCV16UC3 is a Mat of 16-bit unsigned int with 3 channels
    MatTypeCV16UC3 = MatType.MatTypeCV16U + MatChannels.MatChannels3,

    // MatTypeCV16UC4 is a Mat of 16-bit unsigned int with 4 channels
    MatTypeCV16UC4 = MatType.MatTypeCV16U + MatChannels.MatChannels4,

    // MatTypeCV16SC1 is a Mat of 16-bit signed int with a single channel
    MatTypeCV16SC1 = MatType.MatTypeCV16S + MatChannels.MatChannels1,

    // MatTypeCV16SC3 is a Mat of 16-bit signed int with 3 channels
    MatTypeCV16SC3 = MatType.MatTypeCV16S + MatChannels.MatChannels3,

    // MatTypeCV16SC4 is a Mat of 16-bit signed int with 4 channels
    MatTypeCV16SC4 = MatType.MatTypeCV16S + MatChannels.MatChannels4,

    // MatTypeCV32SC1 is a Mat of 32-bit signed int with a single channel
    MatTypeCV32SC1 = MatType.MatTypeCV32S + MatChannels.MatChannels1,

    // MatTypeCV32SC2 is a Mat of 32-bit signed int with 2 channels
    MatTypeCV32SC2 = MatType.MatTypeCV32S + MatChannels.MatChannels2,

    // MatTypeCV32SC3 is a Mat of 32-bit signed int with 3 channels
    MatTypeCV32SC3 = MatType.MatTypeCV32S + MatChannels.MatChannels3,

    // MatTypeCV32SC4 is a Mat of 32-bit signed int with 4 channels
    MatTypeCV32SC4 = MatType.MatTypeCV32S + MatChannels.MatChannels4,

    // MatTypeCV32FC1 is a Mat of 32-bit float int with a single channel
    MatTypeCV32FC1 = MatType.MatTypeCV32F + MatChannels.MatChannels1,

    // MatTypeCV32FC2 is a Mat of 32-bit float int with 2 channels
    MatTypeCV32FC2 = MatType.MatTypeCV32F + MatChannels.MatChannels2,

    // MatTypeCV32FC3 is a Mat of 32-bit float int with 3 channels
    MatTypeCV32FC3 = MatType.MatTypeCV32F + MatChannels.MatChannels3,

    // MatTypeCV32FC4 is a Mat of 32-bit float int with 4 channels
    MatTypeCV32FC4 = MatType.MatTypeCV32F + MatChannels.MatChannels4,

    // MatTypeCV64FC1 is a Mat of 64-bit float int with a single channel
    MatTypeCV64FC1 = MatType.MatTypeCV64F + MatChannels.MatChannels1,

    // MatTypeCV64FC2 is a Mat of 64-bit float int with 2 channels
    MatTypeCV64FC2 = MatType.MatTypeCV64F + MatChannels.MatChannels2,

    // MatTypeCV64FC3 is a Mat of 64-bit float int with 3 channels
    MatTypeCV64FC3 = MatType.MatTypeCV64F + MatChannels.MatChannels3,

    // MatTypeCV64FC4 is a Mat of 64-bit float int with 4 channels
    MatTypeCV64FC4 = MatType.MatTypeCV64F + MatChannels.MatChannels4,
};

pub const CompareType = enum(i32) {
    // CompareEQ src1 is equal to src2.
    CompareEQ = 0,

    // CompareGT src1 is greater than src2.
    CompareGT = 1,

    // CompareGE src1 is greater than or equal to src2.
    CompareGE = 2,

    // CompareLT src1 is less than src2.
    CompareLT = 3,

    // CompareLE src1 is less than or equal to src2.
    CompareLE = 4,

    // CompareNE src1 is unequal to src2.
    CompareNE = 5,
};

pub const SolveDecompositionFlags = enum(i32) {
    // Gaussian elimination with the optimal pivot element chosen.
    SolveDecompositionLu = 0,

    // Singular value decomposition (SVD) method. The system can be over-defined and/or the matrix src1 can be singular.
    SolveDecompositionSvd = 1,

    // Eigenvalue decomposition. The matrix src1 must be symmetrical.
    SolveDecompositionEing = 2,

    // Cholesky LL^T factorization. The matrix src1 must be symmetrical and positively defined.
    SolveDecompositionCholesky = 3,

    // QR factorization. The system can be over-defined and/or the matrix src1 can be singular.
    SolveDecompositionQr = 4,

    // While all the previous flags are mutually exclusive, this flag can be used together with any of the previous.
    // It means that the normal equations 𝚜𝚛𝚌𝟷^T⋅𝚜𝚛𝚌𝟷⋅𝚍𝚜𝚝=𝚜𝚛𝚌𝟷^T𝚜𝚛𝚌𝟸 are solved instead of the original system
    // 𝚜𝚛𝚌𝟷⋅𝚍𝚜𝚝=𝚜𝚛𝚌𝟸.
    SolveDecompositionNormal = 5,
};

pub const TermCriteriaType = enum(i32) {

    // Count is the maximum number of iterations or elements to compute.
    Count = 1,

    // MaxIter is the maximum number of iterations or elements to compute.
    MaxIter = 1,

    // EPS is the desired accuracy or change in parameters at which the
    // iterative algorithm stops.
    EPS = 2,
};

pub const Mat = struct {
    ptr: c.Mat,

    const Self = @This();

    const OperationType = enum {
        Add,
        Subtract,
        Multiply,
        Divide,
    };

    pub fn init() Self {
        return .{ .ptr = c.Mat_New() };
    }

    pub fn initWithSize(n_rows: c_int, n_cols: c_int, mt: MatType) Self {
        return .{ .ptr = c.Mat_NewWithSize(n_rows, n_cols, @enumToInt(mt)) };
    }

    pub fn initFromScalar(s: Scalar) Self {
        return .{c.Mat_NewFromScalar(Scalar.initFromC(s))};
    }

    pub fn initFromC(ptr: c.Mat) !Self {
        if (ptr == null) {
            return error.RuntimeError;
        }
        return Self{ .ptr = ptr };
    }

    pub fn deinit(self: *Self) void {
        _ = c.Mat_Close(self.ptr);
    }

    pub fn toC(self: Self) c.Mat {
        return self.ptr;
    }

    pub fn copy(self: Self, dest: *Mat) void {
        _ = c.Mat_CopyTo(self.ptr, dest.*.ptr);
    }

    pub fn clone(self: Self) Self {
        return .{ .ptr = c.Mat_Clone(self.ptr) };
    }

    pub fn cols(self: Self) i32 {
        return c.Mat_Cols(self.ptr);
    }

    pub fn rows(self: Self) i32 {
        return c.Mat_Rows(self.ptr);
    }

    pub fn channels(self: Self) MatChannels {
        return @intToEnum(MatChannels, c.Mat_Channels(self.ptr));
    }

    pub fn getType(self: Self) MatType {
        var t = c.Mat_Type(self.ptr);
        return @intToEnum(MatType, t);
    }

    pub fn total(self: Self) i32 {
        return c.Mat_Total(self.ptr);
    }

    pub fn size(self: Self) []i32 {
        var return_v: c.IntVector = undefined;
        _ = c.Mat_Size(self.ptr, &return_v);
        return return_v;
    }

    // getAt returns a value from a specific row/col
    // in this Mat expecting it to be of type uchar aka CV_8U.
    // in this Mat expecting it to be of type schar aka CV_8S.
    // in this Mat expecting it to be of type short aka CV_16S.
    // in this Mat expecting it to be of type int aka CV_32S.
    // in this Mat expecting it to be of type float aka CV_32F.
    // in this Mat expecting it to be of type double aka CV_64F.
    pub fn at(self: Self, row: i32, col: i32, comptime T: type) T {
        return switch (T) {
            u8 => c.Mat_GetUChar(self.ptr, row, col),
            i8 => c.Mat_GetSChar(self.ptr, row, col),
            i16 => c.Mat_GetShort(self.ptr, row, col),
            i32 => c.Mat_GetInt(self.ptr, row, col),
            f32 => c.Mat_GetFloat(self.ptr, row, col),
            f64 => c.Mat_GetDouble(self.ptr, row, col),
            else => @compileError("not implemented for " ++ @typeName(T)),
        };
    }

    // getAt3 returns a value from a specific x, y, z coordinate location
    // in this Mat expecting it to be of type uchar aka CV_8U.
    // in this Mat expecting it to be of type schar aka CV_8S.
    // in this Mat expecting it to be of type short aka CV_16S.
    // in this Mat expecting it to be of type int aka CV_32S.
    // in this Mat expecting it to be of type float aka CV_32F.
    // in this Mat expecting it to be of type double aka CV_64F.
    pub fn at3(self: Self, x: i32, y: i32, z: i32, comptime T: type) T {
        return switch (T) {
            u8 => c.Mat_GetUChar3(self.ptr, x, y, z),
            i8 => c.Mat_GetSChar3(self.ptr, x, y, z),
            i16 => c.Mat_GetShort3(self.ptr, x, y, z),
            i32 => c.Mat_GetInt3(self.ptr, x, y, z),
            f32 => c.Mat_GetFloat3(self.ptr, x, y, z),
            f64 => c.Mat_GetDouble3(self.ptr, x, y, z),
            else => @compileError("not implemented for " ++ @typeName(T)),
        };
    }

    // IsContinuous determines if the Mat is continuous.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#aa90cea495029c7d1ee0a41361ccecdf3
    //
    pub fn isContinuous(self: Self) bool {
        return c.Mat_IsContinuous(self.ptr);
    }

    pub fn isEmpty(self: Self) bool {
        return c.Mat_Empty(self.ptr) != 0;
    }

    // Sqrt calculates a square root of array elements.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga186222c3919657890f88df5a1f64a7d7
    //
    pub fn sqrt(self: Self) Mat {
        return .{ .ptr = c.Mat_Sqrt(self.ptr) };
    }

    // Mean calculates the mean value M of array elements, independently for each channel, and return it as Scalar
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga191389f8a0e58180bb13a727782cd461
    //
    pub fn mean(self: Self) Scalar {
        return Scalar.initFromC(c.Mat_Mean(self.ptr));
    }

    pub fn calcValueInplace(self: *Self, v: anytype, op: OperationType) void {
        const T = @TypeOf(v);
        return switch (T) {
            u8 => switch (op) {
                .Add => c.Mat_AddUChar(self.ptr, v, op),
                .Subtract => c.Mat_SubtractUChar(self.ptr, v, op),
                .Multiply => c.Mat_MultiplyUChar(self.ptr, v, op),
                .Divide => c.Mat_DivideUChar(self.ptr, v, op),
            },
            f32 => switch (op) {
                .Add => c.Mat_AddFloat(self.ptr, v, op),
                .Subtract => c.Mat_SubtractFloat(self.ptr, v, op),
                .Multiply => c.Mat_MultiplyFloat(self.ptr, v, op),
                .Divide => c.Mat_DivideFloat(self.ptr, v, op),
            },
            else => @compileError("not implemented for " ++ @typeName(T)),
        };
    }

    // Add calculates the per-element sum of two arrays or an array and a scalar.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6
    //
    // Subtract calculates the per-element subtraction of two arrays or an array and a scalar.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#gaa0f00d98b4b5edeaeb7b8333b2de353b
    //
    // Multiply calculates the per-element scaled product of two arrays.
    // Both input arrays must be of the same size and the same type.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga979d898a58d7f61c53003e162e7ad89f
    //
    // Divide performs the per-element division
    // on two arrays or an array and a scalar.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6db555d30115642fedae0cda05604874
    //
    pub fn calcMat(self: Self, m: Mat, dest: *Mat, op: OperationType) void {
        return switch (op) {
            .Add => c.Mat_Add(self.ptr, m.ptr, dest.*.ptr),
            .Subtract => c.Mat_Subtract(self.ptr, m.ptr, dest.*.ptr),
            .Multiply => c.Mat_Multiply(self.ptr, m.ptr, dest.*.ptr),
            .Divide => c.Mat_Divide(self.ptr, m.ptr, dest.*.ptr),
        };
    }

    // AddWeighted calculates the weighted sum of two arrays.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
    //
    pub fn addMatWeighted(self: Self, alpha: f64, m: Mat, beta: f64) Mat {
        var dest = self.init();
        _ = c.Mat_AddWeighted(self.ptr, alpha, m.ptr, beta, dest.ptr);
        return dest;
    }

    pub fn dataPtr(self: Self, comptime T: type) ![]T {
        if (switch (T) {
            u8 => @enumToInt(self.getType()) & @enumToInt(MatType.MatTypeCV8U) != @enumToInt(MatType.MatTypeCV8U),
            i8 => @enumToInt(self.getType()) & @enumToInt(MatType.MatTypeCV8I) != @enumToInt(MatType.MatTypeCV8I),
            u16 => @enumToInt(self.getType()) & @enumToInt(MatType.MatTypeCV16U) != @enumToInt(MatType.MatTypeCV16U),
            i16 => @enumToInt(self.getType()) & @enumToInt(MatType.MatTypeCV16S) != @enumToInt(MatType.MatTypeCV16S),
            f32 => @enumToInt(self.getType()) & @enumToInt(MatType.MatTypeCV32F) != @enumToInt(MatType.MatTypeCV32F),
            f64 => @enumToInt(self.getType()) & @enumToInt(MatType.MatTypeCV64F) != @enumToInt(MatType.MatTypeCV64F),
        }) {
            return error.RuntimeError;
        }

        if (!self.isContinuous()) {
            return error.RuntimeError;
        }

        var p: c.ByteArray = c.Mat_DataPtr(self.ptr);
        return @ptrCast([*]T, @alignCast(@alignOf(f32), p.data))[0 .. p.length / (@sizeOf(T) / @sizeOf(u8))];
    }

    pub fn randN(self: *Self, mean_: Scalar, stddev: Scalar) void {
        _ = c.Mat_RandN(self.ptr, mean_.toC(), stddev.toC());
    }

    pub fn randShuffle(self: *Self) void {
        _ = c.RandShuffle(self.ptr);
    }

    pub fn randShuffleWithParams(self: *Self, iter_factor: f64, rng: RNG) void {
        _ = c.RandShuffleWithParams(self.ptr, iter_factor, rng.toC());
    }

    pub fn randU(self: *Self, low: Scalar, high: Scalar) void {
        _ = c.RandU(self.ptr, low.toC(), high.toC());
    }

    pub fn cArrayToArrayList(c_mats: c.Mats, allocator: std.mem.Allocator) !std.ArrayList(Self) {
        const len = @intCast(usize, c_mats.length);
        var return_rects = std.ArrayList(Rect).initCapacity(allocator, len);
        for (return_rects) |_, i| return_rects[i] = Rect.initFromC(c_mats.rects[i]);
        return return_rects;
    }
};

pub fn matsToCMats(mats: []const Mat, allocator: std.mem.Allocator) !c.Mats {
    var c_mats = std.ArrayList(c.Mat).init(allocator);
    for (mats) |mat| try c_mats.append(mat.ptr);
    return .{
        .mats = @ptrCast([*]c.Mat, c_mats.items),
        .length = mats.len,
    };
}

pub const Point = struct {
    x: c_int,
    y: c_int,

    const Self = @This();

    pub fn int(x: c_int, y: c_int) Self {
        return .{ .x = x, .y = y };
    }

    pub fn initfromC(p: c.Point) Self {
        return .{ .x = p.x, .y = p.y };
    }

    pub fn toC(self: Self) c.Point {
        return .{ .x = self.x, .y = self.y };
    }

    pub fn deinit(self: *Self) void {
        _ = c.Point_Close(self.*.ptr);
    }
};

pub const Point2f = struct {
    x: f32,
    y: f32,

    const Self = @This();

    pub fn int(x: f32, y: f32) Self {
        return .{ .x = x, .y = y };
    }

    pub fn initfromC(p: c.Point2f) Self {
        return .{ .x = p.x, .y = p.y };
    }

    pub fn toC(self: Self) c.Point2f {
        return .{ .x = self.x, .y = self.y };
    }
};

pub const Point3f = struct {
    x: f32,
    y: f32,
    z: f32,

    const Self = @This();

    pub fn int(x: f32, y: f32, z: f32) Self {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn initfromC(p: c.Point3f) Self {
        return .{ .x = p.x, .y = p.y, .z = p.z };
    }

    pub fn toC(self: Self) c.Point3f {
        return .{ .x = self.x, .y = self.y, .z = self.z };
    }
};

pub const PointVector = struct {
    ptr: c.PointVector,

    const Self = @This();

    pub fn init() Self {
        return .{ .ptr = c.PointVector_New() };
    }

    pub fn initFromMat(mat: Mat) !Self {
        if (mat.ptr == null) {
            return error.RuntimeError;
        }
        return .{ .ptr = c.PointVector_NewFromMat(mat.ptr) };
    }

    pub fn initFromPoints(points: []const Point) Self {
        var c_points: [points.len]c.Point = undefined;
        for (points) |p, i| c_points[i] = p.toC();
        return .{
            .ptr = c.PointVector_NewFromPoints(
                c.Points{
                    .length = points.len,
                    .points = @ptrCast([*]c.Point, c_points),
                },
            ),
        };
    }

    pub fn deinit(self: *Self) void {
        _ = c.PointVector_Close(self.ptr);
    }

    pub fn toC(self: Self) c.PointVector {
        return self.ptr;
    }

    pub fn at(self: Self, idx: c_int) Point {
        return c.PointVector_At(self.ptr, idx);
    }

    pub fn size(self: Self) c_int {
        return c.PointVector_Size(self.ptr);
    }

    pub fn append(self: *Self, p: Point) void {
        c.PointVector_Append(self.ptr, p);
    }
};

pub const Point2fVector = struct {
    ptr: c.Point2fVector,

    const Self = @This();

    pub fn init() Self {
        return .{ .ptr = c.Point2fVector_New() };
    }

    pub fn initFromMat(mat: Mat) !Self {
        if (mat.ptr == null) {
            return error.RuntimeError;
        }
        return .{ .ptr = c.Point2fVector_NewFromMat(mat.ptr) };
    }

    pub fn initFromPoints(points: []const Point) Self {
        var c_points: [points.len]c.Point = undefined;
        for (points) |p, i| c_points[i] = p.toC();
        return .{
            .ptr = c.Point2fVector_NewFromPoints(
                c.Points{
                    .length = points.len,
                    .points = @ptrCast([*]c.Point, c_points),
                },
            ),
        };
    }

    pub fn deinit(self: *Self) void {
        _ = c.Point2fVector_Close(self.ptr);
    }

    pub fn toC(self: Self) c.Point2fVector {
        return self.ptr;
    }

    pub fn at(self: Self, idx: c_int) Point {
        return c.Point2fVector_At(self.ptr, idx);
    }

    pub fn size(self: Self) c_int {
        return c.Point2fVector_Size(self.ptr);
    }

    pub fn append(self: *Self, p: Point) void {
        c.Point2fVector_Append(self.ptr, p);
    }
};

pub const Scalar = struct {
    val1: f64,
    val2: f64,
    val3: f64,
    val4: f64,

    const Self = @This();

    pub fn init(
        val1: f64,
        val2: f64,
        val3: f64,
        val4: f64,
    ) Self {
        return .{
            .val1 = val1,
            .val2 = val2,
            .val3 = val3,
            .val4 = val4,
        };
    }

    pub fn initFromC(s: c.Scalar) Self {
        return .{
            .val1 = s.val1,
            .val2 = s.val2,
            .val3 = s.val3,
            .val4 = s.val4,
        };
    }

    pub fn toC(self: Self) c.Scalar {
        return .{
            .val1 = self.val1,
            .val2 = self.val2,
            .val3 = self.val3,
            .val4 = self.val4,
        };
    }
};

pub const KeyPoint = struct {
    x: f64,
    y: f64,
    size: f64,
    angle: f64,
    response: f64,
    octave: c_int,
    class_id: c_int,

    const Self = @This();

    pub fn init(
        x: f64,
        y: f64,
        size: f64,
        angle: f64,
        response: f64,
        octave: c_int,
        class_id: c_int,
    ) Self {
        return .{
            .x = x,
            .y = y,
            .size = size,
            .angle = angle,
            .response = response,
            .octave = octave,
            .class_id = class_id,
        };
    }

    pub fn initFromC(kp: c.KeyPoint) Self {
        return .{
            .x = kp.x,
            .y = kp.y,
            .size = kp.size,
            .angle = kp.angle,
            .response = kp.response,
            .octave = kp.octave,
            .class_id = kp.class_id,
        };
    }

    pub fn toC(self: Self) c.KeyPoint {
        return .{
            .x = self.x,
            .y = self.y,
            .size = self.size,
            .angle = self.angle,
            .response = self.response,
            .octave = self.octave,
            .class_id = self.class_id,
        };
    }

    pub fn arrayFromC(kps: c.KeyPoints, allocator: std.mem.Allocator) ![]Self {
        var arr = try std.ArrayList(Self).init(allocator);
        {
            var i: usize = 0;
            while (i < kps.length) : (i += 1) {
                try arr.append(Self.initFromC(kps.keypoints[i]));
            }
        }
        return arr;
    }
};

pub const Rect = struct {
    x: c_int,
    y: c_int,
    width: c_int,
    height: c_int,

    const Self = @This();

    pub fn init(
        x: c_int,
        y: c_int,
        width: c_int,
        height: c_int,
    ) Self {
        return .{
            .x = x,
            .y = y,
            .width = width,
            .height = height,
        };
    }

    pub fn initFromC(r: c.Rect) Self {
        return .{
            .x = r.x,
            .y = r.y,
            .width = r.width,
            .height = r.height,
        };
    }

    pub fn toC(self: Self) c.Rect {
        return .{
            .x = self.x,
            .y = self.y,
            .width = self.width,
            .height = self.height,
        };
    }

    pub fn cArrayToArrayList(c_rects: c.Rects, allocator: std.mem.Allocator) !std.ArrayList(Self) {
        const len = @intCast(usize, c_rects.length);
        var return_rects = std.ArrayList(Rect).initCapacity(allocator, len);
        for (return_rects) |_, i| return_rects[i] = Rect.initFromC(c_rects.rects[i]);
        return return_rects;
    }
};

pub const RotatedRect = extern struct {
    pts: c.Points,
    boundingRect: Rect,
    center: Point,
    size: Size,
    angle: f64,

    const Self = @This();

    pub fn init(
        pts: c.Points,
        boundingRect: Rect,
        center: Point,
        size: Size,
        angle: f64,
    ) Self {
        return .{
            .pts = pts,
            .boundingRect = boundingRect,
            .center = center,
            .size = size,
            .angle = angle,
        };
    }

    pub fn initFromC(r: c.RotatedRect) Self {
        return .{
            .pts = r.pts,
            .boundingRect = r.boundingRect,
            .center = Point.initfromC(r.center),
            .size = Size.initFromC(r.size),
            .angle = r.angle,
        };
    }

    pub fn toC(self: Self) c.RotatedRect {
        return .{
            .pts = self.pts,
            .boundingRect = self.boundingRect.toC(),
            .center = self.center.toC(),
            .size = self.size.toC(),
            .angle = self.angle,
        };
    }
};

pub const Size = struct {
    width: c_int,
    height: c_int,

    const Self = @This();

    pub fn init(width: c_int, height: c_int) Self {
        return .{
            .width = width,
            .height = height,
        };
    }

    pub fn initFromC(r: c.Size) Self {
        return .{
            .width = r.width,
            .height = r.height,
        };
    }

    pub fn toC(self: Self) c.Size {
        return .{
            .width = self.width,
            .height = self.height,
        };
    }
};

pub const RNG = struct {
    ptr: c.RNG,

    const Self = @This();

    // TheRNG Returns the default random number generator.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga75843061d150ad6564b5447e38e57722
    //
    pub fn init() Self {
        .{ .ptr = c.TheRNG() };
    }

    pub fn initFromC(ptr: c.RNG) Self {
        return .{ .ptr = ptr };
    }

    pub fn toC(self: Self) c.RNG {
        return self.ptr;
    }

    // TheRNG Sets state of default random number generator.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga757e657c037410d9e19e819569e7de0f
    //
    pub fn setRNGSeed(self: Self, seed: c_int) void {
        _ = self;
        c.SetRNGSeed(seed);
    }

    // Fill Fills arrays with random numbers.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d1/dd6/classcv_1_1RNG.html#ad26f2b09d9868cf108e84c9814aa682d
    //
    pub fn fill(self: *Self, mat: *Mat, dist_type: c_int, a: f64, b: f64, saturate_range: bool) void {
        _ = c.RNG_Fill(self.ptr, mat.*.ptr, dist_type, a, b, saturate_range);
    }

    // Gaussian Returns the next random number sampled from
    // the Gaussian distribution.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d1/dd6/classcv_1_1RNG.html#a8df8ce4dc7d15916cee743e5a884639d
    //
    pub fn gaussian(self: *Self, sigma: f64) f64 {
        return c.RNG_Gaussian(self.ptr, sigma);
    }

    // Gaussian Returns the next random number sampled from
    // the Gaussian distribution.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d1/dd6/classcv_1_1RNG.html#a8df8ce4dc7d15916cee743e5a884639d
    //
    pub fn next(self: *Self) c_uint {
        return c.RNG_Next(self.ptr);
    }
};

//*    implementation done
//     pub extern fn Mats_get(mats: struct_Mats, i: c_int) Mat;
//     pub extern fn MultiDMatches_get(mds: struct_MultiDMatches, index: c_int) struct_DMatches;
//     pub extern fn toByteArray(buf: [*c]const u8, len: c_int) struct_ByteArray;
//     pub extern fn ByteArray_Release(buf: struct_ByteArray) void;
//     pub extern fn Contours_Close(cs: struct_Contours) void;
//     pub extern fn KeyPoints_Close(ks: struct_KeyPoints) void;
//     pub extern fn Rects_Close(rs: struct_Rects) void;
//     pub extern fn Mats_Close(mats: struct_Mats) void;
//*    pub extern fn Point_Close(p: struct_Point) void;
//     pub extern fn Points_Close(ps: struct_Points) void;
//     pub extern fn DMatches_Close(ds: struct_DMatches) void;
//     pub extern fn MultiDMatches_Close(mds: struct_MultiDMatches) void;
//*    pub extern fn Mat_New(...) Mat;
//*    pub extern fn Mat_NewWithSize(rows: c_int, cols: c_int, @"type": c_int) Mat;
//     pub extern fn Mat_NewWithSizes(sizes: struct_IntVector, @"type": c_int) Mat;
//     pub extern fn Mat_NewWithSizesFromScalar(sizes: IntVector, @"type": c_int, ar: Scalar) Mat;
//     pub extern fn Mat_NewWithSizesFromBytes(sizes: IntVector, @"type": c_int, buf: struct_ByteArray) Mat;
//*    pub extern fn Mat_NewFromScalar(ar: Scalar, @"type": c_int) Mat;
//     pub extern fn Mat_NewWithSizeFromScalar(ar: Scalar, rows: c_int, cols: c_int, @"type": c_int) Mat;
//     pub extern fn Mat_NewFromBytes(rows: c_int, cols: c_int, @"type": c_int, buf: struct_ByteArray) Mat;
//     pub extern fn Mat_FromPtr(m: Mat, rows: c_int, cols: c_int, @"type": c_int, prows: c_int, pcols: c_int) Mat;
//*    pub extern fn Mat_Close(m: Mat) void;
//*    pub extern fn Mat_Empty(m: Mat) c_int;
//*    pub extern fn Mat_IsContinuous(m: Mat) bool;
//*    pub extern fn Mat_Clone(m: Mat) Mat;
//*    pub extern fn Mat_CopyTo(m: Mat, dst: Mat) void;
//*    pub extern fn Mat_Total(m: Mat) c_int;
//*    pub extern fn Mat_Size(m: Mat, res: [*c]IntVector) void;
//     pub extern fn Mat_CopyToWithMask(m: Mat, dst: Mat, mask: Mat) void;
//     pub extern fn Mat_ConvertTo(m: Mat, dst: Mat, @"type": c_int) void;
//     pub extern fn Mat_ConvertToWithParams(m: Mat, dst: Mat, @"type": c_int, alpha: f32, beta: f32) void;
//     pub extern fn Mat_ToBytes(m: Mat) struct_ByteArray;
//*    pub extern fn Mat_DataPtr(m: Mat) struct_ByteArray;
//     pub extern fn Mat_Region(m: Mat, r: Rect) Mat;
//     pub extern fn Mat_Reshape(m: Mat, cn: c_int, rows: c_int) Mat;
//     pub extern fn Mat_PatchNaNs(m: Mat) void;
//     pub extern fn Mat_ConvertFp16(m: Mat) Mat;
//*    pub extern fn Mat_Mean(m: Mat) Scalar;
//     pub extern fn Mat_MeanWithMask(m: Mat, mask: Mat) Scalar;
//     pub extern fn Mat_Sqrt(m: Mat) Mat;
//*    pub extern fn Mat_Rows(m: Mat) c_int;
//*    pub extern fn Mat_Cols(m: Mat) c_int;
//*    pub extern fn Mat_Channels(m: Mat) c_int;
//*    pub extern fn Mat_Type(m: Mat) c_int;
//     pub extern fn Mat_Step(m: Mat) c_int;
//     pub extern fn Mat_ElemSize(m: Mat) c_int;
//     pub extern fn Eye(rows: c_int, cols: c_int, @"type": c_int) Mat;
//     pub extern fn Zeros(rows: c_int, cols: c_int, @"type": c_int) Mat;
//     pub extern fn Ones(rows: c_int, cols: c_int, @"type": c_int) Mat;
//*    pub extern fn Mat_GetUChar(m: Mat, row: c_int, col: c_int) u8;
//*    pub extern fn Mat_GetUChar3(m: Mat, x: c_int, y: c_int, z: c_int) u8;
//*    pub extern fn Mat_GetSChar(m: Mat, row: c_int, col: c_int) i8;
//*    pub extern fn Mat_GetSChar3(m: Mat, x: c_int, y: c_int, z: c_int) i8;
//*    pub extern fn Mat_GetShort(m: Mat, row: c_int, col: c_int) i16;
//*    pub extern fn Mat_GetShort3(m: Mat, x: c_int, y: c_int, z: c_int) i16;
//*    pub extern fn Mat_GetInt(m: Mat, row: c_int, col: c_int) i32;
//*    pub extern fn Mat_GetInt3(m: Mat, x: c_int, y: c_int, z: c_int) i32;
//*    pub extern fn Mat_GetFloat(m: Mat, row: c_int, col: c_int) f32;
//*    pub extern fn Mat_GetFloat3(m: Mat, x: c_int, y: c_int, z: c_int) f32;
//*    pub extern fn Mat_GetDouble(m: Mat, row: c_int, col: c_int) f64;
//*    pub extern fn Mat_GetDouble3(m: Mat, x: c_int, y: c_int, z: c_int) f64;
//     pub extern fn Mat_SetTo(m: Mat, value: Scalar) void;
//     pub extern fn Mat_SetUChar(m: Mat, row: c_int, col: c_int, val: u8) void;
//     pub extern fn Mat_SetUChar3(m: Mat, x: c_int, y: c_int, z: c_int, val: u8) void;
//     pub extern fn Mat_SetSChar(m: Mat, row: c_int, col: c_int, val: i8) void;
//     pub extern fn Mat_SetSChar3(m: Mat, x: c_int, y: c_int, z: c_int, val: i8) void;
//     pub extern fn Mat_SetShort(m: Mat, row: c_int, col: c_int, val: i16) void;
//     pub extern fn Mat_SetShort3(m: Mat, x: c_int, y: c_int, z: c_int, val: i16) void;
//     pub extern fn Mat_SetInt(m: Mat, row: c_int, col: c_int, val: i32) void;
//     pub extern fn Mat_SetInt3(m: Mat, x: c_int, y: c_int, z: c_int, val: i32) void;
//     pub extern fn Mat_SetFloat(m: Mat, row: c_int, col: c_int, val: f32) void;
//     pub extern fn Mat_SetFloat3(m: Mat, x: c_int, y: c_int, z: c_int, val: f32) void;
//     pub extern fn Mat_SetDouble(m: Mat, row: c_int, col: c_int, val: f64) void;
//     pub extern fn Mat_SetDouble3(m: Mat, x: c_int, y: c_int, z: c_int, val: f64) void;
//*    pub extern fn Mat_AddUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_SubtractUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_MultiplyUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_DivideUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_AddFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_SubtractFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_MultiplyFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_DivideFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_MultiplyMatrix(x: Mat, y: Mat) Mat;
//     pub extern fn Mat_T(x: Mat) Mat;
//     pub extern fn LUT(src: Mat, lut: Mat, dst: Mat) void;
//     pub extern fn Mat_AbsDiff(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_Add(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_AddWeighted(src1: Mat, alpha: f64, src2: Mat, beta: f64, gamma: f64, dst: Mat) void;
//     pub extern fn Mat_BitwiseAnd(src1: Mat, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_BitwiseAndWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
//     pub extern fn Mat_BitwiseNot(src1: Mat, dst: Mat) void;
//     pub extern fn Mat_BitwiseNotWithMask(src1: Mat, dst: Mat, mask: Mat) void;
//     pub extern fn Mat_BitwiseOr(src1: Mat, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_BitwiseOrWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
//     pub extern fn Mat_BitwiseXor(src1: Mat, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_BitwiseXorWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
//     pub extern fn Mat_Compare(src1: Mat, src2: Mat, dst: Mat, ct: c_int) void;
//     pub extern fn Mat_BatchDistance(src1: Mat, src2: Mat, dist: Mat, dtype: c_int, nidx: Mat, normType: c_int, K: c_int, mask: Mat, update: c_int, crosscheck: bool) void;
//     pub extern fn Mat_BorderInterpolate(p: c_int, len: c_int, borderType: c_int) c_int;
//     pub extern fn Mat_CalcCovarMatrix(samples: Mat, covar: Mat, mean: Mat, flags: c_int, ctype: c_int) void;
//     pub extern fn Mat_CartToPolar(x: Mat, y: Mat, magnitude: Mat, angle: Mat, angleInDegrees: bool) void;
//     pub extern fn Mat_CheckRange(m: Mat) bool;
//     pub extern fn Mat_CompleteSymm(m: Mat, lowerToUpper: bool) void;
//     pub extern fn Mat_ConvertScaleAbs(src: Mat, dst: Mat, alpha: f64, beta: f64) void;
//     pub extern fn Mat_CopyMakeBorder(src: Mat, dst: Mat, top: c_int, bottom: c_int, left: c_int, right: c_int, borderType: c_int, value: Scalar) void;
//     pub extern fn Mat_CountNonZero(src: Mat) c_int;
//     pub extern fn Mat_DCT(src: Mat, dst: Mat, flags: c_int) void;
//     pub extern fn Mat_Determinant(m: Mat) f64;
//     pub extern fn Mat_DFT(m: Mat, dst: Mat, flags: c_int) void;
//*    pub extern fn Mat_Divide(src1: Mat, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_Eigen(src: Mat, eigenvalues: Mat, eigenvectors: Mat) bool;
//     pub extern fn Mat_EigenNonSymmetric(src: Mat, eigenvalues: Mat, eigenvectors: Mat) void;
//     pub extern fn Mat_Exp(src: Mat, dst: Mat) void;
//     pub extern fn Mat_ExtractChannel(src: Mat, dst: Mat, coi: c_int) void;
//     pub extern fn Mat_FindNonZero(src: Mat, idx: Mat) void;
//     pub extern fn Mat_Flip(src: Mat, dst: Mat, flipCode: c_int) void;
//     pub extern fn Mat_Gemm(src1: Mat, src2: Mat, alpha: f64, src3: Mat, beta: f64, dst: Mat, flags: c_int) void;
//     pub extern fn Mat_GetOptimalDFTSize(vecsize: c_int) c_int;
//     pub extern fn Mat_Hconcat(src1: Mat, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_Vconcat(src1: Mat, src2: Mat, dst: Mat) void;
//     pub extern fn Rotate(src: Mat, dst: Mat, rotationCode: c_int) void;
//     pub extern fn Mat_Idct(src: Mat, dst: Mat, flags: c_int) void;
//     pub extern fn Mat_Idft(src: Mat, dst: Mat, flags: c_int, nonzeroRows: c_int) void;
//     pub extern fn Mat_InRange(src: Mat, lowerb: Mat, upperb: Mat, dst: Mat) void;
//     pub extern fn Mat_InRangeWithScalar(src: Mat, lowerb: Scalar, upperb: Scalar, dst: Mat) void;
//     pub extern fn Mat_InsertChannel(src: Mat, dst: Mat, coi: c_int) void;
//     pub extern fn Mat_Invert(src: Mat, dst: Mat, flags: c_int) f64;
//     pub extern fn KMeans(data: Mat, k: c_int, bestLabels: Mat, criteria: TermCriteria, attempts: c_int, flags: c_int, centers: Mat) f64;
//     pub extern fn KMeansPoints(pts: PointVector, k: c_int, bestLabels: Mat, criteria: TermCriteria, attempts: c_int, flags: c_int, centers: Mat) f64;
//     pub extern fn Mat_Log(src: Mat, dst: Mat) void;
//     pub extern fn Mat_Magnitude(x: Mat, y: Mat, magnitude: Mat) void;
//     pub extern fn Mat_Max(src1: Mat, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_MeanStdDev(src: Mat, dstMean: Mat, dstStdDev: Mat) void;
//     pub extern fn Mat_Merge(mats: struct_Mats, dst: Mat) void;
//     pub extern fn Mat_Min(src1: Mat, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_MinMaxIdx(m: Mat, minVal: [*c]f64, maxVal: [*c]f64, minIdx: [*c]c_int, maxIdx: [*c]c_int) void;
//     pub extern fn Mat_MinMaxLoc(m: Mat, minVal: [*c]f64, maxVal: [*c]f64, minLoc: [*c]Point, maxLoc: [*c]Point) void;
//     pub extern fn Mat_MixChannels(src: struct_Mats, dst: struct_Mats, fromTo: struct_IntVector) void;
//     pub extern fn Mat_MulSpectrums(a: Mat, b: Mat, c: Mat, flags: c_int) void;
//*    pub extern fn Mat_Multiply(src1: Mat, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_MultiplyWithParams(src1: Mat, src2: Mat, dst: Mat, scale: f64, dtype: c_int) void;
//*    pub extern fn Mat_Subtract(src1: Mat, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_Normalize(src: Mat, dst: Mat, alpha: f64, beta: f64, typ: c_int) void;
//     pub extern fn Norm(src1: Mat, normType: c_int) f64;
//     pub extern fn NormWithMats(src1: Mat, src2: Mat, normType: c_int) f64;
//     pub extern fn Mat_PerspectiveTransform(src: Mat, dst: Mat, tm: Mat) void;
//     pub extern fn Mat_Solve(src1: Mat, src2: Mat, dst: Mat, flags: c_int) bool;
//     pub extern fn Mat_SolveCubic(coeffs: Mat, roots: Mat) c_int;
//     pub extern fn Mat_SolvePoly(coeffs: Mat, roots: Mat, maxIters: c_int) f64;
//     pub extern fn Mat_Reduce(src: Mat, dst: Mat, dim: c_int, rType: c_int, dType: c_int) void;
//     pub extern fn Mat_Repeat(src: Mat, nY: c_int, nX: c_int, dst: Mat) void;
//     pub extern fn Mat_ScaleAdd(src1: Mat, alpha: f64, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_SetIdentity(src: Mat, scalar: f64) void;
//     pub extern fn Mat_Sort(src: Mat, dst: Mat, flags: c_int) void;
//     pub extern fn Mat_SortIdx(src: Mat, dst: Mat, flags: c_int) void;
//     pub extern fn Mat_Split(src: Mat, mats: [*c]struct_Mats) void;
//     pub extern fn Mat_Trace(src: Mat) Scalar;
//     pub extern fn Mat_Transform(src: Mat, dst: Mat, tm: Mat) void;
//     pub extern fn Mat_Transpose(src: Mat, dst: Mat) void;
//     pub extern fn Mat_PolarToCart(magnitude: Mat, degree: Mat, x: Mat, y: Mat, angleInDegrees: bool) void;
//     pub extern fn Mat_Pow(src: Mat, power: f64, dst: Mat) void;
//     pub extern fn Mat_Phase(x: Mat, y: Mat, angle: Mat, angleInDegrees: bool) void;
//     pub extern fn Mat_Sum(src1: Mat) Scalar;
//     pub extern fn TermCriteria_New(typ: c_int, maxCount: c_int, epsilon: f64) TermCriteria;
//     pub extern fn GetCVTickCount(...) i64;
//     pub extern fn GetTickFrequency(...) f64;
//     pub extern fn Mat_rowRange(m: Mat, startrow: c_int, endrow: c_int) Mat;
//     pub extern fn Mat_colRange(m: Mat, startrow: c_int, endrow: c_int) Mat;
//*    pub extern fn PointVector_New(...) PointVector;
//     pub extern fn PointVector_NewFromPoints(points: Contour) PointVector;
//*    pub extern fn PointVector_NewFromMat(mat: Mat) PointVector;
//*    pub extern fn PointVector_At(pv: PointVector, idx: c_int) Point;
//*    pub extern fn PointVector_Append(pv: PointVector, p: Point) void;
//*    pub extern fn PointVector_Size(pv: PointVector) c_int;
//*    pub extern fn PointVector_Close(pv: PointVector) void;
//     pub extern fn PointsVector_New(...) PointsVector;
//     pub extern fn PointsVector_NewFromPoints(points: Contours) PointsVector;
//     pub extern fn PointsVector_At(psv: PointsVector, idx: c_int) PointVector;
//     pub extern fn PointsVector_Append(psv: PointsVector, pv: PointVector) void;
//     pub extern fn PointsVector_Size(psv: PointsVector) c_int;
//     pub extern fn PointsVector_Close(psv: PointsVector) void;
//*    pub extern fn Point2fVector_New(...) Point2fVector;
//*    pub extern fn Point2fVector_Close(pfv: Point2fVector) void;
//*    pub extern fn Point2fVector_NewFromPoints(pts: Contour2f) Point2fVector;
//*    pub extern fn Point2fVector_NewFromMat(mat: Mat) Point2fVector;
//*    pub extern fn Point2fVector_At(pfv: Point2fVector, idx: c_int) Point2f;
//*    pub extern fn Point2fVector_Size(pfv: Point2fVector) c_int;
//     pub extern fn IntVector_Close(ivec: struct_IntVector) void;
//     pub extern fn CStrings_Close(cstrs: struct_CStrings) void;
//*    pub extern fn TheRNG(...) RNG;
//*    pub extern fn SetRNGSeed(seed: c_int) void;
//*    pub extern fn RNG_Fill(rng: RNG, mat: Mat, distType: c_int, a: f64, b: f64, saturateRange: bool) void;
//*    pub extern fn RNG_Gaussian(rng: RNG, sigma: f64) f64;
//*    pub extern fn RNG_Next(rng: RNG) c_uint;
//*    pub extern fn RandN(mat: Mat, mean: Scalar, stddev: Scalar) void;
//*    pub extern fn RandShuffle(mat: Mat) void;
//*    pub extern fn RandShuffleWithParams(mat: Mat, iterFactor: f64, rng: RNG) void;
//*    pub extern fn RandU(mat: Mat, low: Scalar, high: Scalar) void;
//     pub extern fn copyPointVectorToPoint2fVector(src: PointVector, dest: Point2fVector) void;
//     pub extern fn StdByteVectorInitialize(data: ?*anyopaque) void;
//     pub extern fn StdByteVectorFree(data: ?*anyopaque) void;
//     pub extern fn StdByteVectorLen(data: ?*anyopaque) usize;
//     pub extern fn StdByteVectorData(data: ?*anyopaque) [*c]u8;
//     pub extern fn Points2fVector_New(...) Points2fVector;
//     pub extern fn Points2fVector_NewFromPoints(points: Contours2f) Points2fVector;
//     pub extern fn Points2fVector_Size(ps: Points2fVector) c_int;
//     pub extern fn Points2fVector_At(ps: Points2fVector, idx: c_int) Point2fVector;
//     pub extern fn Points2fVector_Append(psv: Points2fVector, pv: Point2fVector) void;
//     pub extern fn Points2fVector_Close(ps: Points2fVector) void;
//     pub extern fn Point3fVector_New(...) Point3fVector;
//     pub extern fn Point3fVector_NewFromPoints(points: Contour3f) Point3fVector;
//     pub extern fn Point3fVector_NewFromMat(mat: Mat) Point3fVector;
//     pub extern fn Point3fVector_Append(pfv: Point3fVector, point: Point3f) void;
//     pub extern fn Point3fVector_At(pfv: Point3fVector, idx: c_int) Point3f;
//     pub extern fn Point3fVector_Size(pfv: Point3fVector) c_int;
//     pub extern fn Point3fVector_Close(pv: Point3fVector) void;
//     pub extern fn Points3fVector_New(...) Points3fVector;
//     pub extern fn Points3fVector_NewFromPoints(points: Contours3f) Points3fVector;
//     pub extern fn Points3fVector_Size(ps: Points3fVector) c_int;
//     pub extern fn Points3fVector_At(ps: Points3fVector, idx: c_int) Point3fVector;
//     pub extern fn Points3fVector_Append(psv: Points3fVector, pv: Point3fVector) void;
//     pub extern fn Points3fVector_Close(ps: Points3fVector) void;

// pub const struct_CStrings = extern struct {
//     strs: [*c][*c]const u8,
//     length: c_int,
// };
// pub const CStrings = struct_CStrings;
// pub const struct_ByteArray = extern struct {
//     data: [*c]u8,
//     length: c_int,
// };
// pub const ByteArray = struct_ByteArray;
// pub const struct_IntVector = extern struct {
//     val: [*c]c_int,
//     length: c_int,
// };
// pub const IntVector = struct_IntVector;
// pub const struct_FloatVector = extern struct {
//     val: [*c]f32,
//     length: c_int,
// };
// pub const FloatVector = struct_FloatVector;
// pub const struct_RawData = extern struct {
//     width: c_int,
//     height: c_int,
//     data: struct_ByteArray,
// };
// pub const RawData = struct_RawData;
// pub const struct_Point2f = extern struct {
//     x: f32,
//     y: f32,
// };
// pub const Point2f = struct_Point2f;
// pub const struct_Point3f = extern struct {
//     x: f32,
//     y: f32,
//     z: f32,
// };
// pub const Point3f = struct_Point3f;
// pub const struct_Point = extern struct {
//     x: c_int,
//     y: c_int,
// };
// pub const Point = struct_Point;
// pub const struct_Points = extern struct {
//     points: [*c]Point,
//     length: c_int,
// };
// pub const Points = struct_Points;
// pub const struct_Points2f = extern struct {
//     points: [*c]Point2f,
//     length: c_int,
// };
// pub const Points2f = struct_Points2f;
// pub const struct_Points3f = extern struct {
//     points: [*c]Point3f,
//     length: c_int,
// };
// pub const Points3f = struct_Points3f;
// pub const Contour = Points;
// pub const Contour2f = Points2f;
// pub const struct_Contours2f = extern struct {
//     contours: [*c]Contour2f,
//     length: c_int,
// };
// pub const Contours2f = struct_Contours2f;
// pub const Contour3f = Points3f;
// pub const struct_Contours3f = extern struct {
//     contours: [*c]Contour3f,
//     length: c_int,
// };
// pub const Contours3f = struct_Contours3f;
// pub const struct_Contours = extern struct {
//     contours: [*c]Contour,
//     length: c_int,
// };
// pub const Contours = struct_Contours;
// pub const struct_Rect = extern struct {
//     x: c_int,
//     y: c_int,
//     width: c_int,
//     height: c_int,
// };
// pub const Rect = struct_Rect;
// pub const struct_Rects = extern struct {
//     rects: [*c]Rect,
//     length: c_int,
// };
// pub const Rects = struct_Rects;
// pub const struct_Size = extern struct {
//     width: c_int,
//     height: c_int,
// };
// pub const Size = struct_Size;
// pub const struct_RotatedRect = extern struct {
//     pts: Points,
//     boundingRect: Rect,
//     center: Point,
//     size: Size,
//     angle: f64,
// };
// pub const RotatedRect = struct_RotatedRect;
// pub const struct_Scalar = extern struct {
//     val1: f64,
//     val2: f64,
//     val3: f64,
//     val4: f64,
// };
// pub const Scalar = struct_Scalar;
// pub const struct_KeyPoint = extern struct {
//     x: f64,
//     y: f64,
//     size: f64,
//     angle: f64,
//     response: f64,
//     octave: c_int,
//     classID: c_int,
// };
// pub const KeyPoint = struct_KeyPoint;
// pub const struct_KeyPoints = extern struct {
//     keypoints: [*c]KeyPoint,
//     length: c_int,
// };
// pub const KeyPoints = struct_KeyPoints;
// pub const struct_SimpleBlobDetectorParams = extern struct {
//     blobColor: u8,
//     filterByArea: bool,
//     filterByCircularity: bool,
//     filterByColor: bool,
//     filterByConvexity: bool,
//     filterByInertia: bool,
//     maxArea: f32,
//     maxCircularity: f32,
//     maxConvexity: f32,
//     maxInertiaRatio: f32,
//     maxThreshold: f32,
//     minArea: f32,
//     minCircularity: f32,
//     minConvexity: f32,
//     minDistBetweenBlobs: f32,
//     minInertiaRatio: f32,
//     minRepeatability: usize,
//     minThreshold: f32,
//     thresholdStep: f32,
// };
// pub const SimpleBlobDetectorParams = struct_SimpleBlobDetectorParams;
// pub const struct_DMatch = extern struct {
//     queryIdx: c_int,
//     trainIdx: c_int,
//     imgIdx: c_int,
//     distance: f32,
// };
// pub const DMatch = struct_DMatch;
// pub const struct_DMatches = extern struct {
//     dmatches: [*c]DMatch,
//     length: c_int,
// };
// pub const DMatches = struct_DMatches;
// pub const struct_MultiDMatches = extern struct {
//     dmatches: [*c]DMatches,
//     length: c_int,
// };
// pub const MultiDMatches = struct_MultiDMatches;
// pub const struct_Moment = extern struct {
//     m00: f64,
//     m10: f64,
//     m01: f64,
//     m20: f64,
//     m11: f64,
//     m02: f64,
//     m30: f64,
//     m21: f64,
//     m12: f64,
//     m03: f64,
//     mu20: f64,
//     mu11: f64,
//     mu02: f64,
//     mu30: f64,
//     mu21: f64,
//     mu12: f64,
//     mu03: f64,
//     nu20: f64,
//     nu11: f64,
//     nu02: f64,
//     nu30: f64,
//     nu21: f64,
//     nu12: f64,
//     nu03: f64,
// };
// pub const Moment = struct_Moment;
// pub const Mat = ?*anyopaque;
// pub const TermCriteria = ?*anyopaque;
// pub const RNG = ?*anyopaque;
// pub const PointVector = ?*anyopaque;
// pub const PointsVector = ?*anyopaque;
// pub const Point2fVector = ?*anyopaque;
// pub const Points2fVector = ?*anyopaque;
// pub const Point3fVector = ?*anyopaque;
// pub const Points3fVector = ?*anyopaque;
// pub const struct_Mats = extern struct {
//     mats: [*c]Mat,
//     length: c_int,
// };
