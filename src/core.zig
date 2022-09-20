const c = @import("c_api.zig");
const std = @import("std");
const utils = @import("utils.zig");
const epnn = utils.ensurePtrNotNull;

pub const Mat = @import("core/mat.zig");
pub const Mats = Mat.Mats;

pub const Point = struct {
    x: i32,
    y: i32,

    const Self = @This();

    pub fn init(x: i32, y: i32) Self {
        return .{ .x = x, .y = y };
    }

    pub fn initFromC(p: c.Point) Self {
        return .{ .x = p.x, .y = p.y };
    }

    pub fn toC(self: Self) c.Point {
        return .{ .x = self.x, .y = self.y };
    }
};

pub const Point2f = struct {
    x: f32,
    y: f32,

    const Self = @This();

    pub fn int(x: f32, y: f32) Self {
        return .{ .x = x, .y = y };
    }

    pub fn initFromC(p: c.Point2f) Self {
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

    pub fn initFromC(p: c.Point3f) Self {
        return .{ .x = p.x, .y = p.y, .z = p.z };
    }

    pub fn toC(self: Self) c.Point3f {
        return .{ .x = self.x, .y = self.y, .z = self.z };
    }
};

pub const PointVector = struct {
    ptr: c.PointVector,
    allocator: ?std.mem.Allocator = null,

    const Self = @This();

    pub fn init() !Self {
        const ptr = c.PointVector_New();
        const nn_ptr = try epnn(ptr);
        return .{ .ptr = nn_ptr };
    }

    pub fn initFromMat(mat: Mat) !Self {
        const mat_ptr = try epnn(mat.ptr);
        const ptr = c.PointVector_NewFromMat(mat_ptr);
        const nn_ptr = try epnn(ptr);
        return .{ .ptr = nn_ptr };
    }

    pub fn fromPoints(points: []const Point, allocator: std.mem.Allocator) !Self {
        const len = @intCast(usize, points.len);
        var arr = try std.ArrayList(c.Point).initCapacity(allocator, len);
        {
            var i: usize = 0;
            while (i < len) : (i += 1) {
                arr.items[i] = points[i].toC();
            }
        }
        return .{
            .ptr = c.PointVector_NewFromPoints(
                c.Points{
                    .length = points.len,
                    .points = arr.toOwnedSliceSentinel(0),
                },
            ),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| {
            allocator.free(self.ptr.points);
        } else {
            _ = c.PointVector_Close(self.ptr);
            self.ptr = null;
        }
    }

    pub fn at(self: Self, idx: i32) Point {
        return c.PointVector_At(self.ptr, idx);
    }

    pub fn size(self: Self) i32 {
        return c.PointVector_Size(self.ptr);
    }

    pub fn append(self: *Self, p: Point) void {
        c.PointVector_Append(self.ptr, p);
    }

    pub fn toC(self: Self) c.PointVector {
        return self.ptr;
    }

    pub fn toArrayList(self: Self, allocator: std.mem.Allocator) !std.ArrayList(Point) {
        return try utils.fromCStructsToArrayList(self.ptr, self.size(), Point, allocator);
    }
};

pub const Point2fVector = struct {
    ptr: c.Point2fVector,
    allocator: ?std.mem.Allocator = null,

    const Self = @This();

    pub fn init() !Self {
        const ptr = c.Point2fVector_New();
        const nn_ptr = try epnn(ptr);
        return .{ .ptr = nn_ptr };
    }

    pub fn initFromMat(mat: Mat) !Self {
        const mat_ptr = try epnn(mat.ptr);
        const ptr = c.Point2fVector_NewFromMat(mat_ptr);
        const nn_ptr = try epnn(ptr);
        return .{ .ptr = nn_ptr };
    }

    pub fn initFromPoints(points: []const Point, allocator: std.mem.Allocator) !Self {
        const len = @intCast(usize, points.len);
        var arr = try std.ArrayList(c.Point).initCapacity(allocator, len);
        {
            var i: usize = 0;
            while (i < len) : (i += 1) {
                arr.items[i] = points[i].toC();
            }
        }
        return .{
            .ptr = c.PointVector_NewFromPoints(
                c.Points{
                    .length = points.len,
                    .points = try arr.toOwnedSliceSentinel(0),
                },
            ),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| {
            allocator.free(self.ptr.points);
        } else {
            _ = c.PointVector_Close(self.ptr);
            self.ptr = null;
        }
    }

    pub fn at(self: Self, idx: i32) Point {
        return c.Point2fVector_At(self.ptr, idx);
    }

    pub fn size(self: Self) i32 {
        return c.Point2fVector_Size(self.ptr);
    }

    pub fn append(self: *Self, p: Point) void {
        c.Point2fVector_Append(self.ptr, p);
    }

    pub fn toC(self: Self) c.Point2fVector {
        return self.ptr;
    }

    pub fn toArrayList(self: Self, allocator: std.mem.Allocator) !std.ArrayList(Point2f) {
        return try utils.fromCStructsToArrayList(self.ptr, self.size(), Point2f, allocator);
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

    pub fn toArray(self: Self) [4]f64 {
        return .{
            self.val1,
            self.val2,
            self.val3,
            self.val4,
        };
    }
};

pub const Color = struct {
    r: f64 = 0,
    g: f64 = 0,
    b: f64 = 0,
    a: f64 = 0,

    const Self = @This();

    pub fn init(
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) Self {
        return .{ .r = r, .g = g, .b = b, .a = a };
    }

    pub fn toScalar(self: Self) Scalar {
        return .{
            .val1 = self.b,
            .val2 = self.g,
            .val3 = self.r,
            .val4 = self.a,
        };
    }
};

pub const KeyPoint = struct {
    x: f64,
    y: f64,
    size: f64,
    angle: f64,
    response: f64,
    octave: i32,
    class_id: i32,

    const Self = @This();

    pub fn init(
        x: f64,
        y: f64,
        size: f64,
        angle: f64,
        response: f64,
        octave: i32,
        class_id: i32,
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
            .class_id = kp.classID,
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
            .classID = self.class_id,
        };
    }

    pub fn toArrayList(c_kps: c.KeyPoints, allocator: std.mem.Allocator) !KeyPoints {
        return try utils.fromCStructsToArrayList(c_kps.keypoints, c_kps.length, Self, allocator);
    }
};

pub const KeyPoints = std.ArrayList(KeyPoint);

pub const Rect = struct {
    x: i32,
    y: i32,
    width: i32,
    height: i32,

    const Self = @This();

    pub fn init(
        x: i32,
        y: i32,
        width: i32,
        height: i32,
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

    pub fn toArrayList(c_rects: c.Rects, allocator: std.mem.Allocator) !Rects {
        return try utils.fromCStructsToArrayList(c_rects.rects, c_rects.length, Self, allocator);
    }

    pub fn deinitRects(rects: c.Rects) void {
        _ = c.Rects_Close(rects);
    }
};

pub const Rects = std.ArrayList(Rect);

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
            .center = Point.fromC(r.center),
            .size = Size.fromC(r.size),
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
    width: u31,
    height: u31,

    const Self = @This();

    pub fn init(width: i32, height: i32) Self {
        return .{
            .width = @intCast(u31, width),
            .height = @intCast(u31, height),
        };
    }

    pub fn initFromC(r: c.Size) Self {
        return .{
            .width = @intCast(u31, r.width),
            .height = @intCast(u31, r.height),
        };
    }

    pub fn toC(self: Self) c.Size {
        return .{
            .width = @intCast(c_int, self.width),
            .height = @intCast(c_int, self.height),
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
    pub fn setRNGSeed(self: Self, seed: i32) void {
        _ = self;
        c.SetRNGSeed(seed);
    }

    // Fill Fills arrays with random numbers.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d1/dd6/classcv_1_1RNG.html#ad26f2b09d9868cf108e84c9814aa682d
    //
    pub fn fill(self: *Self, mat: *Mat, dist_type: i32, a: f64, b: f64, saturate_range: bool) void {
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

pub const STDVector = packed struct {
    value: usize,
    before: usize,
    after: usize,

    const self = @This();
    pub fn init(ptr: *self) void {
        c.StdByteVectorInitialize(ptr);
    }

    pub fn deinit(ptr: *self) void {
        c.StdByteVectorFree(ptr);
    }

    pub fn len(ptr: *self) usize {
        return c.StdByteVectorLen(ptr);
    }

    pub fn data(ptr: *self) [*c]u8 {
        return c.StdByteVectorData(ptr);
    }
};

pub const NormTypes = enum(u6) {
    inf = 1,
    l1 = 2,
    l2 = 4,
    l2sqr = 5,
    hamming = 6,
    hamming2 = 7,
    // type_mask = 7,
    relative = 8,
    min_max = 32,
};

pub const TermCriteria = struct {
    ptr: c.TermCriteria,

    const Self = @This();

    // TermCriteriaType for TermCriteria.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d9/d5d/classcv_1_1TermCriteria.html#a56fecdc291ccaba8aad27d67ccf72c57
    //
    pub const Type = enum(u2) {
        // Count is the maximum number of iterations or elements to compute.
        count = 1,

        // MaxIter is the maximum number of iterations or elements to compute.
        max_iter = 1,

        // EPS is the desired accuracy or change in parameters at which the
        // iterative algorithm stops.
        eps = 2,
    };

    pub fn init(type_: Type, max_count: i32, epsilon: f64) !Self {
        const ptr = c.TermCriteria_New(@enumToInt(type_), max_count, epsilon);
        return try initFromC(ptr);
    }

    fn initFromC(ptr: c.TermCriteria) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        self.ptr = null;
    }

    pub fn toC(self: Self) c.TermCriteria {
        return self.ptr;
    }
};

//*    implementation done
//     pub extern fn Mats_get(mats: struct_Mats, i: c_int) Mat;
//     pub extern fn MultiDMatches_get(mds: struct_MultiDMatches, index: c_int) struct_DMatches;
//     pub extern fn toByteArray(buf: [*c]const u8, len: c_int) struct_ByteArray;
//     pub extern fn ByteArray_Release(buf: struct_ByteArray) void;
//     pub extern fn Contours_Close(cs: struct_Contours) void;
//     pub extern fn KeyPoints_Close(ks: struct_KeyPoints) void;
//*    pub extern fn Rects_Close(rs: struct_Rects) void;
//     pub extern fn Mats_Close(mats: struct_Mats) void;
//*    pub extern fn Point_Close(p: struct_Point) void;
//     pub extern fn Points_Close(ps: struct_Points) void;
//     pub extern fn DMatches_Close(ds: struct_DMatches) void;
//     pub extern fn MultiDMatches_Close(mds: struct_MultiDMatches) void;
//*    pub extern fn Mat_New(...) Mat;
//*    pub extern fn Mat_NewWithSize(rows: c_int, cols: c_int, @"type": c_int) Mat;
//*    pub extern fn Mat_NewWithSizes(sizes: struct_IntVector, @"type": c_int) Mat;
//*    pub extern fn Mat_NewWithSizesFromScalar(sizes: IntVector, @"type": c_int, ar: Scalar) Mat;
//     pub extern fn Mat_NewWithSizesFromBytes(sizes: IntVector, @"type": c_int, buf: struct_ByteArray) Mat;
//*    pub extern fn Mat_NewFromScalar(ar: Scalar, @"type": c_int) Mat;
//*    pub extern fn Mat_NewWithSizeFromScalar(ar: Scalar, rows: c_int, cols: c_int, @"type": c_int) Mat;
//     pub extern fn Mat_NewFromBytes(rows: c_int, cols: c_int, @"type": c_int, buf: struct_ByteArray) Mat;
//*    pub extern fn Mat_FromPtr(m: Mat, rows: c_int, cols: c_int, @"type": c_int, prows: c_int, pcols: c_int) Mat;
//*    pub extern fn Mat_Close(m: Mat) void;
//*    pub extern fn Mat_Empty(m: Mat) c_int;
//*    pub extern fn Mat_IsContinuous(m: Mat) bool;
//*    pub extern fn Mat_Clone(m: Mat) Mat;
//*    pub extern fn Mat_CopyTo(m: Mat, dst: Mat) void;
//*    pub extern fn Mat_Total(m: Mat) c_int;
//*    pub extern fn Mat_Size(m: Mat, res: [*c]IntVector) void;
//*    pub extern fn Mat_CopyToWithMask(m: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_ConvertTo(m: Mat, dst: Mat, @"type": c_int) void;
//*    pub extern fn Mat_ConvertToWithParams(m: Mat, dst: Mat, @"type": c_int, alpha: f32, beta: f32) void;
//     pub extern fn Mat_ToBytes(m: Mat) struct_ByteArray;
//*    pub extern fn Mat_DataPtr(m: Mat) struct_ByteArray;
//*    pub extern fn Mat_Region(m: Mat, r: Rect) Mat;
//*    pub extern fn Mat_Reshape(m: Mat, cn: c_int, rows: c_int) Mat;
//*    pub extern fn Mat_PatchNaNs(m: Mat) void;
//*    pub extern fn Mat_ConvertFp16(m: Mat) Mat;
//*    pub extern fn Mat_Mean(m: Mat) Scalar;
//*    pub extern fn Mat_MeanWithMask(m: Mat, mask: Mat) Scalar;
//*    pub extern fn Mat_Sqrt(m: Mat) Mat;
//*    pub extern fn Mat_Rows(m: Mat) c_int;
//*    pub extern fn Mat_Cols(m: Mat) c_int;
//*    pub extern fn Mat_Channels(m: Mat) c_int;
//*    pub extern fn Mat_Type(m: Mat) c_int;
//*    pub extern fn Mat_Step(m: Mat) c_int;
//*    pub extern fn Mat_ElemSize(m: Mat) c_int;
//*    pub extern fn Eye(rows: c_int, cols: c_int, @"type": c_int) Mat;
//*    pub extern fn Zeros(rows: c_int, cols: c_int, @"type": c_int) Mat;
//*    pub extern fn Ones(rows: c_int, cols: c_int, @"type": c_int) Mat;
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
//*    pub extern fn Mat_SetTo(m: Mat, value: Scalar) void;
//*    pub extern fn Mat_SetUChar(m: Mat, row: c_int, col: c_int, val: u8) void;
//*    pub extern fn Mat_SetUChar3(m: Mat, x: c_int, y: c_int, z: c_int, val: u8) void;
//*    pub extern fn Mat_SetSChar(m: Mat, row: c_int, col: c_int, val: i8) void;
//*    pub extern fn Mat_SetSChar3(m: Mat, x: c_int, y: c_int, z: c_int, val: i8) void;
//*    pub extern fn Mat_SetShort(m: Mat, row: c_int, col: c_int, val: i16) void;
//*    pub extern fn Mat_SetShort3(m: Mat, x: c_int, y: c_int, z: c_int, val: i16) void;
//*    pub extern fn Mat_SetInt(m: Mat, row: c_int, col: c_int, val: i32) void;
//*    pub extern fn Mat_SetInt3(m: Mat, x: c_int, y: c_int, z: c_int, val: i32) void;
//*    pub extern fn Mat_SetFloat(m: Mat, row: c_int, col: c_int, val: f32) void;
//*    pub extern fn Mat_SetFloat3(m: Mat, x: c_int, y: c_int, z: c_int, val: f32) void;
//*    pub extern fn Mat_SetDouble(m: Mat, row: c_int, col: c_int, val: f64) void;
//*    pub extern fn Mat_SetDouble3(m: Mat, x: c_int, y: c_int, z: c_int, val: f64) void;
//*    pub extern fn Mat_AddUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_SubtractUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_MultiplyUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_DivideUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_AddFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_SubtractFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_MultiplyFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_DivideFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_MultiplyMatrix(x: Mat, y: Mat) Mat;
//*    pub extern fn Mat_T(x: Mat) Mat;
//*    pub extern fn LUT(src: Mat, lut: Mat, dst: Mat) void;
//*    pub extern fn Mat_AbsDiff(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_Add(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_AddWeighted(src1: Mat, alpha: f64, src2: Mat, beta: f64, gamma: f64, dst: Mat) void;
//*    pub extern fn Mat_BitwiseAnd(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_BitwiseAndWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_BitwiseNot(src1: Mat, dst: Mat) void;
//*    pub extern fn Mat_BitwiseNotWithMask(src1: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_BitwiseOr(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_BitwiseOrWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_BitwiseXor(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_BitwiseXorWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
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
//*    pub extern fn Mat_Eigen(src: Mat, eigenvalues: Mat, eigenvectors: Mat) bool;
//*    pub extern fn Mat_EigenNonSymmetric(src: Mat, eigenvalues: Mat, eigenvectors: Mat) void;
//*    pub extern fn Mat_Exp(src: Mat, dst: Mat) void;
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
//*    pub extern fn Mat_Solve(src1: Mat, src2: Mat, dst: Mat, flags: c_int) bool;
//*    pub extern fn Mat_SolveCubic(coeffs: Mat, roots: Mat) c_int;
//*    pub extern fn Mat_SolvePoly(coeffs: Mat, roots: Mat, maxIters: c_int) f64;
//     pub extern fn Mat_Reduce(src: Mat, dst: Mat, dim: c_int, rType: c_int, dType: c_int) void;
//     pub extern fn Mat_Repeat(src: Mat, nY: c_int, nX: c_int, dst: Mat) void;
//     pub extern fn Mat_ScaleAdd(src1: Mat, alpha: f64, src2: Mat, dst: Mat) void;
//     pub extern fn Mat_SetIdentity(src: Mat, scalar: f64) void;
//     pub extern fn Mat_Sort(src: Mat, dst: Mat, flags: c_int) void;
//     pub extern fn Mat_SortIdx(src: Mat, dst: Mat, flags: c_int) void;
//     pub extern fn Mat_Split(src: Mat, mats: [*c]struct_Mats) void;
//     pub extern fn Mat_Trace(src: Mat) Scalar;
//     pub extern fn Mat_Transform(src: Mat, dst: Mat, tm: Mat) void;
//*    pub extern fn Mat_Transpose(src: Mat, dst: Mat) void;
//     pub extern fn Mat_PolarToCart(magnitude: Mat, degree: Mat, x: Mat, y: Mat, angleInDegrees: bool) void;
//     pub extern fn Mat_Pow(src: Mat, power: f64, dst: Mat) void;
//     pub extern fn Mat_Phase(x: Mat, y: Mat, angle: Mat, angleInDegrees: bool) void;
//*    pub extern fn Mat_Sum(src1: Mat) Scalar;
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
//*    pub extern fn StdByteVectorInitialize(data: ?*anyopaque) void;
//*    pub extern fn StdByteVectorFree(data: ?*anyopaque) void;
//*    pub extern fn StdByteVectorLen(data: ?*anyopaque) usize;
//*    pub extern fn StdByteVectorData(data: ?*anyopaque) [*c]u8;
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
