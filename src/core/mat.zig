const c = @import("../c_api.zig");
const std = @import("std");
const utils = @import("../utils.zig");
const core = @import("../core.zig");
const epnn = utils.ensurePtrNotNull;
const Rect = core.Rect;
const Scalar = core.Scalar;
const RNG = core.RNG;

ptr: CSelf,

const Self = @This();
const CSelf = c.Mat;

const OperationType = enum {
    add,
    subtract,
    multiply,
    divide,
};

const BitOperationType = enum {
    and_,
    or_,
    xor_,
    not_,
};

// TODO:apply packed struct like this
// pub const MatType = packed struct {
//     type: u3,
//     channel: u2,
// };
pub const MatType = enum(u5) {
    // MatTypeCV8UC1 is a Mat of 8-bit unsigned int with a single channel
    cv8uc1 = cv8u + channels1,

    // MatTypeCV8UC2 is a Mat of 8-bit unsigned int with 2 channels
    cv8uc2 = cv8u + channels2,

    // MatTypeCV8UC3 is a Mat of 8-bit unsigned int with 3 channels
    cv8uc3 = cv8u + channels3,

    // MatTypeCV8UC4 is a Mat of 8-bit unsigned int with 4 channels
    cv8uc4 = cv8u + channels4,

    // MatTypeCV8SC1 is a Mat of 8-bit signed int with a single channel
    cv8sc1 = cv8s + channels1,

    // MatTypeCV8SC2 is a Mat of 8-bit signed int with 2 channels
    cv8sc2 = cv8s + channels2,

    // MatTypeCV8SC3 is a Mat of 8-bit signed int with 3 channels
    cv8sc3 = cv8s + channels3,

    // MatTypeCV8SC4 is a Mat of 8-bit signed int with 4 channels
    cv8sc4 = cv8s + channels4,

    // MatTypeCV16UC1 is a Mat of 16-bit unsigned int with a single channel
    cv16uc1 = cv16u + channels1,

    // MatTypeCV16UC2 is a Mat of 16-bit unsigned int with 2 channels
    cv16uc2 = cv16u + channels2,

    // MatTypeCV16UC3 is a Mat of 16-bit unsigned int with 3 channels
    cv16uc3 = cv16u + channels3,

    // MatTypeCV16UC4 is a Mat of 16-bit unsigned int with 4 channels
    cv16uc4 = cv16u + channels4,

    // MatTypeCV16SC1 is a Mat of 16-bit signed int with a single channel
    cv16sc1 = cv16s + channels1,

    // MatTypeCV16SC2 is a Mat of 16-bit signed int with 2 channels
    cv16sc2 = cv16s + channels2,

    // MatTypeCV16SC3 is a Mat of 16-bit signed int with 3 channels
    cv16sc3 = cv16s + channels3,

    // MatTypeCV16SC4 is a Mat of 16-bit signed int with 4 channels
    cv16sc4 = cv16s + channels4,

    // MatTypeCV32SC1 is a Mat of 32-bit signed int with a single channel
    cv32sc1 = cv32s + channels1,

    // MatTypeCV32SC2 is a Mat of 32-bit signed int with 2 channels
    cv32sc2 = cv32s + channels2,

    // MatTypeCV32SC3 is a Mat of 32-bit signed int with 3 channels
    cv32sc3 = cv32s + channels3,

    // MatTypeCV32SC4 is a Mat of 32-bit signed int with 4 channels
    cv32sc4 = cv32s + channels4,

    // MatTypeCV32FC1 is a Mat of 32-bit float int with a single channel
    cv32fc1 = cv32f + channels1,

    // MatTypeCV32FC2 is a Mat of 32-bit float int with 2 channels
    cv32fc2 = cv32f + channels2,

    // MatTypeCV32FC3 is a Mat of 32-bit float int with 3 channels
    cv32fc3 = cv32f + channels3,

    // MatTypeCV32FC4 is a Mat of 32-bit float int with 4 channels
    cv32fc4 = cv32f + channels4,

    // MatTypeCV64FC1 is a Mat of 64-bit float int with a single channel
    cv64fc1 = cv64f + channels1,

    // MatTypeCV64FC2 is a Mat of 64-bit float int with 2 channels
    cv64fc2 = cv64f + channels2,

    // MatTypeCV64FC3 is a Mat of 64-bit float int with 3 channels
    cv64fc3 = cv64f + channels3,

    // MatTypeCV64FC4 is a Mat of 64-bit float int with 4 channels
    cv64fc4 = cv64f + channels4,

    const channels1 = 0 << 3;
    const channels2 = 1 << 3;
    const channels3 = 2 << 3;
    const channels4 = 3 << 3;
    // MatTypeCV8U is a Mat of 8-bit unsigned int
    const cv8u = 0;
    // MatTypeCV8S is a Mat of 8-bit signed int
    const cv8s = 1;
    // MatTypeCV16U is a Mat of 16-bit unsigned int
    const cv16u = 2;
    // MatTypeCV16S is a Mat of 16-bit signed int
    const cv16s = 3;
    // MatTypeCV32S is a Mat of 32-bit signed int
    const cv32s = 4;
    // MatTypeCV32F is a Mat of 32-bit float
    const cv32f = 5;
    // MatTypeCV64F is a Mat of 64-bit float
    const cv64f = 6;
};

pub const CompareType = enum(u3) {
    // CompareEQ src1 is equal to src2.
    eq = 0,

    // CompareGT src1 is greater than src2.
    gt = 1,

    // CompareGE src1 is greater than or equal to src2.
    ge = 2,

    // CompareLT src1 is less than src2.
    lt = 3,

    // CompareLE src1 is less than or equal to src2.
    le = 4,

    // CompareNE src1 is unequal to src2.
    ne = 5,
};

pub const SolveDecompositionFlag = enum(u3) {
    // Gaussian elimination with the optimal pivot element chosen.
    lu = 0,

    // Singular value decomposition (SVD) method. The system can be over-defined and/or the matrix src1 can be singular.
    svd = 1,

    // Eigenvalue decomposition. The matrix src1 must be symmetrical.
    eing = 2,

    // Cholesky LL^T factorization. The matrix src1 must be symmetrical and positively defined.
    cholesky = 3,

    // QR factorization. The system can be over-defined and/or the matrix src1 can be singular.
    qr = 4,

    // While all the previous flags are mutually exclusive, this flag can be used together with any of the previous.
    // It means that the normal equations 𝚜𝚛𝚌𝟷^T⋅𝚜𝚛𝚌𝟷⋅𝚍𝚜𝚝=𝚜𝚛𝚌𝟷^T𝚜𝚛𝚌𝟸 are solved instead of the original system
    // 𝚜𝚛𝚌𝟷⋅𝚍𝚜𝚝=𝚜𝚛𝚌𝟸.
    normal = 5,
};

pub const ReduceType = enum(u3) {
    // The output is the sum of all rows/columns of the matrix.
    sum = 0,

    // The output is the mean vector of all rows/columns of the matrix.
    avg = 1,

    // The output is the maximum (column/row-wise) of all rows/columns of the matrix.
    max = 2,

    // The output is the minimum (column/row-wise) of all rows/columns of the matrix.
    min = 3,
};

pub fn initFromC(ptr: CSelf) !Self {
    const nn_ptr = try epnn(ptr);
    return Self{ .ptr = nn_ptr };
}

/// init Mat
pub fn init() !Self {
    const ptr = c.Mat_New();
    return try Self.initFromC(ptr);
}

/// init Mat with size and type
pub fn initSize(n_rows: i32, n_cols: i32, mt: MatType) !Self {
    const ptr = c.Mat_NewWithSize(n_rows, n_cols, @enumToInt(mt));
    return try Self.initFromC(ptr);
}

/// init multidimentional Mat with sizes and type
pub fn initSizes(size_array: []const i32, mt: MatType) !Self {
    const c_size_vector = c.IntVector{
        .val = @ptrCast([*]i32, size_array),
        .length = @intCast(i32, size_array.len),
    };

    const ptr = c.Mat_NewWithSizes(c_size_vector, @enumToInt(mt));
    return try Self.initFromC(ptr);
}

pub fn initFromMat(self: *Self, n_rows: i32, n_cols: i32, mt: MatType, prows: i32, pcols: i32) !Self {
    const mat_ptr = try epnn(self.*.ptr);
    const ptr = c.Mat_FromPtr(mat_ptr, n_rows, n_cols, @enumToInt(mt), prows, pcols);
    return try Self.initFromC(ptr);
}

pub fn initFromScalar(s: Scalar) !Self {
    const ptr = c.Mat_NewFromScalar(s.toC());
    return try Self.initFromC(ptr);
}

pub fn initSizeFromScalar(s: Scalar, n_rows: i32, n_cols: i32, mt: MatType) !Self {
    const ptr = c.Mat_NewWithSizeFromScalar(s.toC(), n_rows, n_cols, @enumToInt(mt));
    return try Self.initFromC(ptr);
}

pub fn initSizesFromScalar(size_array: []const i32, s: Scalar, mt: MatType) !Self {
    const c_size_vector = c.IntVector{
        .val = @ptrCast([*]i32, size_array),
        .length = @intCast(i32, size_array.len),
    };
    const ptr = c.Mat_NewWithSizesFromScalar(c_size_vector, @enumToInt(mt), s.toC());
    return try Self.initFromC(ptr);
}

/// Returns an identity matrix of the specified size and type.
///
/// The method returns a Matlab-style identity matrix initializer, similarly to Mat::zeros. Similarly to Mat::ones.
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a2cf9b9acde7a9852542bbc20ef851ed2
pub fn initEye(rows_: i32, cols_: i32, mt: MatType) !Self {
    const ptr = c.Eye(rows_, cols_, @enumToInt(mt));
    return try Self.initFromC(ptr);
}

/// Returns a zero array of the specified size and type.
///
/// The method returns a Matlab-style zero array initializer.
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a0b57b6a326c8876d944d188a46e0f556
pub fn initZeros(rows_: i32, cols_: i32, mt: MatType) !Self {
    const ptr = c.Zeros(rows_, cols_, @enumToInt(mt));
    return try Self.initFromC(ptr);
}

/// Returns an array of all 1's of the specified size and type.
///
/// The method returns a Matlab-style 1's array initializer
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a69ae0402d116fc9c71908d8508dc2f09
pub fn initOnes(rows_: i32, cols_: i32, mt: MatType) !Self {
    const ptr = c.Ones(rows_, cols_, @enumToInt(mt));
    return try Self.initFromC(ptr);
}

pub fn deinit(self: *Self) void {
    c.Mat_Close(self.ptr);
    self.ptr = null;
}

pub fn toC(self: Self) CSelf {
    return self.ptr;
}

/// CopyTo copies Mat into destination Mat.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a33fd5d125b4c302b0c9aa86980791a77
///
pub fn copyTo(self: Self, dest: *Self) void {
    _ = c.Mat_CopyTo(self.ptr, dest.*.ptr);
}

/// CopyToWithMask copies Mat into destination Mat after applying the mask Mat.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a626fe5f96d02525e2604d2ad46dd574f
///
pub fn copyToWithMask(self: Self, dest: *Self, mask: Self) void {
    _ = c.Mat_CopyToWithMask(self.ptr, dest.*.ptr, mask.ptr);
}

pub fn clone(self: Self) !Self {
    const ptr = c.Mat_Clone(self.ptr);
    return try Self.initFromC(ptr);
}

// ConvertTo converts Mat into destination Mat.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#adf88c60c5b4980e05bb556080916978b
//
pub fn converTo(self: Self, dst: *Self, mt: MatType) void {
    _ = c.Mat_ConvertTo(self.ptr, dst.*.ptr, @enumToInt(mt));
}
pub fn convertToWithParams(self: Self, dst: *Self, mt: MatType, alpha: f32, beta: f32) void {
    _ = c.Mat_ConvertToWithParams(self.ptr, dst.*.ptr, @enumToInt(mt), alpha, beta);
}

pub fn cols(self: Self) i32 {
    return c.Mat_Cols(self.ptr);
}

pub fn rows(self: Self) i32 {
    return c.Mat_Rows(self.ptr);
}

pub fn channels(self: Self) i32 {
    return c.Mat_Channels(self.ptr);
}

pub fn getType(self: Self) MatType {
    var type_ = c.Mat_Type(self.ptr);
    return @intToEnum(MatType, type_);
}

pub fn step(self: Self) u32 {
    return @intCast(u32, c.Mat_Step(self.ptr));
}

pub fn elemSize(self: Self) u32 {
    return @intCast(u32, c.Mat_ElemSize(self.ptr));
}

// Total returns the total number of array elements.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#aa4d317d43fb0cba9c2503f3c61b866c8
//
pub fn total(self: Self) i32 {
    return c.Mat_Total(self.ptr);
}

// Size returns an array with one element for each dimension containing the size of that dimension for the Mat.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#aa4d317d43fb0cba9c2503f3c61b866c8
//
pub fn size(self: Self) []const i32 {
    var v: c.IntVector = undefined;
    _ = c.Mat_Size(self.ptr, &v);
    return v.val[0..@intCast(usize, v.length)];
}

// getAt returns a value from a specific row/col
// in this Mat expecting it to be of type uchar aka CV_8U.
// in this Mat expecting it to be of type schar aka CV_8S.
// in this Mat expecting it to be of type short aka CV_16S.
// in this Mat expecting it to be of type int aka CV_32S.
// in this Mat expecting it to be of type float aka CV_32F.
// in this Mat expecting it to be of type double aka CV_64F.
pub fn get(self: Self, comptime T: type, row: usize, col: usize) T {
    const row_ = @intCast(u31, row);
    const col_ = @intCast(u31, col);
    return switch (T) {
        u8 => c.Mat_GetUChar(self.ptr, row_, col_),
        i8 => c.Mat_GetSChar(self.ptr, row_, col_),
        i16 => c.Mat_GetShort(self.ptr, row_, col_),
        i32 => c.Mat_GetInt(self.ptr, row_, col_),
        f32 => c.Mat_GetFloat(self.ptr, row_, col_),
        f64 => c.Mat_GetDouble(self.ptr, row_, col_),
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
pub fn get3(self: Self, comptime T: type, x: usize, y: usize, z: usize) T {
    const x_ = @intCast(u31, x);
    const y_ = @intCast(u31, y);
    const z_ = @intCast(u31, z);
    return switch (T) {
        u8 => c.Mat_GetUChar3(self.ptr, x_, y_, z_),
        i8 => c.Mat_GetSChar3(self.ptr, x_, y_, z_),
        i16 => c.Mat_GetShort3(self.ptr, x_, y_, z_),
        i32 => c.Mat_GetInt3(self.ptr, x_, y_, z_),
        f32 => c.Mat_GetFloat3(self.ptr, x_, y_, z_),
        f64 => c.Mat_GetDouble3(self.ptr, x_, y_, z_),
        else => @compileError("not implemented for " ++ @typeName(T)),
    };
}

pub fn setTo(self: *Self, value: Scalar) void {
    _ = c.Mat_SetTo(self.ptr, value.toC());
}

pub fn set(self: *Self, comptime T: type, row: usize, col: usize, val: T) void {
    const row_ = @intCast(u31, row);
    const col_ = @intCast(u31, col);
    _ = switch (T) {
        u8 => c.Mat_SetUChar(self.ptr, row_, col_, val),
        i8 => c.Mat_SetSChar(self.ptr, row_, col_, val),
        i16 => c.Mat_SetShort(self.ptr, row_, col_, val),
        i32 => c.Mat_SetInt(self.ptr, row_, col_, val),
        f32 => c.Mat_SetFloat(self.ptr, row_, col_, val),
        f64 => c.Mat_SetDouble(self.ptr, row_, col_, val),
        else => @compileError("not implemented for " ++ @typeName(T)),
    };
}

pub fn set3(self: *Self, comptime T: type, x: usize, y: usize, z: usize, val: T) void {
    const x_ = @intCast(u31, x);
    const y_ = @intCast(u31, y);
    const z_ = @intCast(u31, z);
    _ = switch (T) {
        u8 => c.Mat_SetUChar3(self.ptr, x_, y_, z_, val),
        i8 => c.Mat_SetSChar3(self.ptr, x_, y_, z_, val),
        i16 => c.Mat_SetShort3(self.ptr, x_, y_, z_, val),
        i32 => c.Mat_SetInt3(self.ptr, x_, y_, z_, val),
        f32 => c.Mat_SetFloat3(self.ptr, x_, y_, z_, val),
        f64 => c.Mat_SetDouble3(self.ptr, x_, y_, z_, val),
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
pub fn sqrt(self: Self) !Self {
    const ptr = c.Mat_Sqrt(self.ptr);
    return try initFromC(ptr);
}

// Mean calculates the mean value M of array elements, independently for each channel, and return it as Scalar
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga191389f8a0e58180bb13a727782cd461
//
pub fn mean(self: Self) Scalar {
    return Scalar.fromC(c.Mat_Mean(self.ptr));
}

// ConvertFp16 converts a Mat to half-precision floating point.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9c25d9ef44a2a48ecc3774b30cb80082
//
pub fn convertFp16(self: Self) Self {
    return .{ .ptr = c.Mat_ConvertFp16(self.ptr) };
}

// MeanWithMask calculates the mean value M of array elements,independently for each channel,
// and returns it as Scalar vector while applying the mask.
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga191389f8a0e58180bb13a727782cd461
//
pub fn meanWithMask(self: Self, mask: Self) Scalar {
    return Scalar.initFromC(c.Mat_MeanWithMask(self.ptr, mask.ptr));
}

pub fn calcValueInplace(self: *Self, comptime op: OperationType, v: anytype) void {
    const T = @TypeOf(v);
    return switch (T) {
        u8 => switch (op) {
            .add => c.Mat_AddUChar(self.ptr, v, op),
            .subtract => c.Mat_SubtractUChar(self.ptr, v, op),
            .multiply => c.Mat_MultiplyUChar(self.ptr, v, op),
            .divide => c.Mat_DivideUChar(self.ptr, v, op),
        },
        f32 => switch (op) {
            .add => c.Mat_AddFloat(self.ptr, v, op),
            .subtract => c.Mat_SubtractFloat(self.ptr, v, op),
            .multiply => c.Mat_MultiplyFloat(self.ptr, v, op),
            .divide => c.Mat_DivideFloat(self.ptr, v, op),
        },
        else => @compileError("not implemented for " ++ @typeName(T)),
    };
}

pub fn addValueInplace(self: *Self, v: anytype) void {
    return self.calcValueInplace(.add, v);
}

pub fn subtractValueInplace(self: *Self, v: anytype) void {
    return self.calcValueInplace(.subtract, v);
}

pub fn multiplyValueInplace(self: *Self, v: anytype) void {
    return self.calcValueInplace(.multiply, v);
}

pub fn divideValueInplace(self: *Self, v: anytype) void {
    return self.calcValueInplace(.divide, v);
}

pub fn calcMat(self: Self, op: OperationType, m: Self, dest: *Self) void {
    return switch (op) {
        .add => c.Mat_Add(self.ptr, m.ptr, dest.*.ptr),
        .subtract => c.Mat_Subtract(self.ptr, m.ptr, dest.*.ptr),
        .multiply => c.Mat_Multiply(self.ptr, m.ptr, dest.*.ptr),
        .divide => c.Mat_Divide(self.ptr, m.ptr, dest.*.ptr),
    };
}

// Add calculates the per-element sum of two arrays or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6
//
pub fn addMat(self: Self, m: Self, dest: *Self) void {
    return self.calcMat(.add, m, dest);
}

// Subtract calculates the per-element subtraction of two arrays or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaa0f00d98b4b5edeaeb7b8333b2de353b
//
pub fn subtractMat(self: Self, m: Self, dest: *Self) void {
    return self.calcMat(.subtract, m, dest);
}

// Multiply calculates the per-element scaled product of two arrays.
// Both input arrays must be of the same size and the same type.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga979d898a58d7f61c53003e162e7ad89f
//
pub fn multiplyMat(self: Self, m: Self, dest: *Self) void {
    return self.calcMat(.multiply, m, dest);
}

// Divide performs the per-element division
// on two arrays or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6db555d30115642fedae0cda05604874
//
pub fn divideMat(self: Self, m: Self, dest: *Self) void {
    return self.calcMat(.divide, m, dest);
}

// AbsDiff calculates the per-element absolute difference between two arrays
// or between an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6fef31bc8c4071cbc114a758a2b79c14
//
pub fn absDiff(self: Self, m: Self, dest: *Self) void {
    _ = c.Mat_AbsDiff(self.ptr, m.ptr, dest.*.ptr);
}

// Eigen calculates eigenvalues and eigenvectors of a symmetric matrix.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9fa0d58657f60eaa6c71f6fbb40456e3
//
pub fn eigen(self: Self, eigenvalues: *Self, eigenvectors: *Self) bool {
    return c.Mat_Eigen(self.ptr, eigenvalues.*.ptr, eigenvectors.*.ptr);
}

// EigenNonSymmetric calculates eigenvalues and eigenvectors of a non-symmetric matrix (real eigenvalues only).
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaf51987e03cac8d171fbd2b327cf966f6
//
pub fn eigenNonSymmetric(self: Self, eigenvalues: *Self, eigenvectors: *Self) void {
    return c.Mat_EigenNonSymmetric(self.ptr, eigenvalues.*.ptr, eigenvectors.*.ptr);
}

// Exp calculates the exponent of every array element.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga3e10108e2162c338f1b848af619f39e5
//
pub fn exp(self: Self, dest: *Self) void {
    return c.Mat_Exp(self.ptr, dest.*.ptr);
}

// AddWeighted calculates the weighted sum of two arrays.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
//
pub fn addMatWeighted(self: Self, alpha: f64, m: Self, beta: f64) Self {
    var dest = self.init();
    _ = c.Mat_AddWeighted(self.ptr, alpha, m.ptr, beta, dest.ptr);
    return dest;
}

pub fn bitwise(self: Self, m: Self, dest: *Self, comptime op: BitOperationType) void {
    _ = switch (op) {
        .and_ => _ = c.Mat_BitwiseAnd(self.ptr, m.ptr, dest.*.ptr),
        .not_ => _ = c.Mat_BitwiseNot(self.ptr, dest.*.ptr),
        .or_ => _ = c.Mat_BitwiseOr(self.ptr, m.ptr, dest.*.ptr),
        .xor_ => _ = c.Mat_BitwiseXor(self.ptr, m.ptr, dest.*.ptr),
    };
}

// BitwiseAnd computes bitwise conjunction of the two arrays (dst = src1 & src2).
// Calculates the per-element bit-wise conjunction of two arrays
// or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14
//
pub fn bitwiseAnd(self: Self, m: Self, dest: *Self) void {
    return self.bitwise(m, dest, .and_);
}

// BitwiseNot inverts every bit of an array.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga0002cf8b418479f4cb49a75442baee2f
//
pub fn bitwiseNot(self: Self, dest: *Self) void {
    return self.bitwise(self, dest, .not_);
}

// BitwiseOr calculates the per-element bit-wise disjunction of two arrays
// or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gab85523db362a4e26ff0c703793a719b4
//

pub fn bitwiseOr(self: Self, m: Self, dest: *Self) void {
    return self.bitwise(m, dest, .or_);
}

// BitwiseXor calculates the per-element bit-wise "exclusive or" operation
// on two arrays or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga84b2d8188ce506593dcc3f8cd00e8e2c
//
pub fn bitwiseXor(self: Self, m: Self, dest: *Self) void {
    return self.bitwise(m, dest, .xor_);
}

pub fn bitwiseWithMask(self: Self, comptime op: BitOperationType, m: Self, dest: *Self, mask: Self) void {
    _ = switch (op) {
        .and_ => _ = c.Mat_BitwiseAndWithMask(self.ptr, m.ptr, dest.*.ptr, mask.ptr),
        .not_ => _ = c.Mat_BitwiseNotWithMask(self.ptr, dest.*.ptr, mask.ptr),
        .or_ => _ = c.Mat_BitwiseOrWithMask(self.ptr, m.ptr, dest.*.ptr, mask.ptr),
        .xor_ => _ = c.Mat_BitwiseXorWithMask(self.ptr, m.ptr, dest.*.ptr, mask.ptr),
    };
}

// BitwiseAndWithMask computes bitwise conjunction of the two arrays (dst = src1 & src2).
// Calculates the per-element bit-wise conjunction of two arrays
// or an array and a scalar. It has an additional parameter for a mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14
//
pub fn bitwiseAndWithMask(self: Self, m: Self, dest: *Self, mask: Self) void {
    return self.bitwiseWithMask(self, .and_, m, dest, mask);
}

// BitwiseNotWithMask inverts every bit of an array. It has an additional parameter for a mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga0002cf8b418479f4cb49a75442baee2f
//
pub fn bitwiseNotWithMask(self: Self, m: Self, dest: *Self, mask: Self) void {
    return self.bitwiseWithMask(self, .not_, m, dest, mask);
}

// BitwiseOrWithMask calculates the per-element bit-wise disjunction of two arrays
// or an array and a scalar. It has an additional parameter for a mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gab85523db362a4e26ff0c703793a719b4
//
pub fn bitwiseOrWithMask(self: Self, m: Self, dest: *Self, mask: Self) void {
    return self.bitwiseWithMask(self, .or_, m, dest, mask);
}

// BitwiseXorWithMask calculates the per-element bit-wise "exclusive or" operation
// on two arrays or an array and a scalar. It has an additional parameter for a mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga84b2d8188ce506593dcc3f8cd00e8e2c
//
pub fn bitwiseXorWithMask(self: Self, m: Self, dest: *Self, mask: Self) void {
    return self.bitwiseWithMask(self, .xor_, m, dest, mask);
}

pub fn dataPtr(self: Self, comptime T: type) ![]T {
    if (switch (T) {
        u8 => @enumToInt(self.getType()) & MatType.cv8u != MatType.cv8u,
        i8 => @enumToInt(self.getType()) & MatType.cv8s != MatType.cv8s,
        u16 => @enumToInt(self.getType()) & MatType.cv16u != MatType.cv16u,
        i16 => @enumToInt(self.getType()) & MatType.cv16s != MatType.cv16s,
        f32 => @enumToInt(self.getType()) & MatType.cv32f != MatType.cv32f,
        f64 => @enumToInt(self.getType()) & MatType.cv64f != MatType.cv64f,
        else => @compileError("Unsupported type"),
    }) {
        return error.RuntimeError;
    }

    if (!self.isContinuous()) {
        return error.RuntimeError;
    }

    var p: c.ByteArray = c.Mat_DataPtr(self.ptr);
    var len = @intCast(usize, p.length);
    const bit_scale = @sizeOf(T) / @sizeOf(u8);
    return @ptrCast([*]T, @alignCast(@alignOf(T), p.data))[0 .. len / bit_scale];
}

// Reshape changes the shape and/or the number of channels of a 2D matrix without copying the data.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a4eb96e3251417fa88b78e2abd6cfd7d8
//
pub fn reshape(self: Self, cn: i32, rows_: usize) !Self {
    const ptr = c.Mat_Reshape(self.ptr, @intCast(u31, cn), @intCast(i32, rows_));
    return try initFromC(ptr);
}

// Region returns a new Mat that points to a region of this Mat. Changes made to the
// region Mat will affect the original Mat, since they are pointers to the underlying
// OpenCV Mat object.
pub fn region(self: Self, r: Rect) Self {
    const ptr = c.Mat_Region(self.ptr, r.toC());
    return initFromC(ptr);
}

// T  transpose matrix
// https://docs.opencv.org/4.1.2/d3/d63/classcv_1_1Mat.html#aaa428c60ccb6d8ea5de18f63dfac8e11
pub fn t(self: Self) Self {
    const ptr = c.Mat_T(self.ptr);
    return initFromC(ptr);
}

// Transpose transposes a matrix.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga46630ed6c0ea6254a35f447289bd7404
//
pub fn transpose(self: Self, dst: *Self) Self {
    _ = c.Mat_Transpose(self.ptr, dst.*.ptr);
}

// LUT performs a look-up table transform of an array.
//
// The function LUT fills the output array with values from the look-up table.
// Indices of the entries are taken from the input array.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gab55b8d062b7f5587720ede032d34156f
pub fn lut(src: Self, lut_: Self, dst: *Self) void {
    c.LUT(src.ptr, lut_.ptr, dst.*.ptr);
}

// Sum calculates the per-channel pixel sum of an image.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga716e10a2dd9e228e4d3c95818f106722
//
//     pub extern fn Mat_Sum(src1: Mat) Scalar;
pub fn sum(self: Self) Scalar {
    return Scalar.initFromC(c.Mat_Sum(self.ptr));
}

// PatchNaNs converts NaN's to zeros.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga62286befb7cde3568ff8c7d14d5079da
//
pub fn patchNaNs(self: Self) void {
    c.Mat_PatchNaNs(self.ptr);
}

// RandN Fills the array with normally distributed random numbers.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaeff1f61e972d133a04ce3a5f81cf6808
//
pub fn randN(self: *Self, mean_: Scalar, stddev: Scalar) void {
    _ = c.RandN(self.ptr, mean_.toC(), stddev.toC());
}

// RandShuffle Shuffles the array elements randomly.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6a789c8a5cb56c6dd62506179808f763
//
pub fn randShuffle(self: *Self) void {
    _ = c.RandShuffle(self.ptr);
}

// RandShuffleWithParams Shuffles the array elements randomly.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6a789c8a5cb56c6dd62506179808f763
//
pub fn randShuffleWithParams(self: *Self, iter_factor: f64, rng: RNG) void {
    _ = c.RandShuffleWithParams(self.ptr, iter_factor, rng.toC());
}

// RandU Generates a single uniformly-distributed random
// number or an array of random numbers.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga1ba1026dca0807b27057ba6a49d258c0
//
pub fn randU(self: *Self, low: Scalar, high: Scalar) void {
    _ = c.RandU(self.ptr, low.toC(), high.toC());
}

// Solve solves one or more linear systems or least-squares problems.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga12b43690dbd31fed96f213eefead2373
//
pub fn solve(self: Self, src2: Self, dst: *Self, flag: SolveDecompositionFlag) bool {
    return c.Mat_Solve(self.ptr, src2.ptr, dst.*.ptr, @enumToInt(flag));
}
// SolveCubic finds the real roots of a cubic equation.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga1c3b0b925b085b6e96931ee309e6a1da
//
pub fn solveCubic(self: Self, roots: *Self) bool {
    return c.Mat_SolveCubic(self.ptr, roots.*.ptr);
}

// SolvePoly finds the real or complex roots of a polynomial equation.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gac2f5e953016fabcdf793d762f4ec5dce
//
pub fn solvePoly(self: Self, roots: *Self, max_iters: i32) bool {
    return c.Mat_SolvePoly(self.ptr, roots.*.ptr, max_iters);
}

pub fn toArrayList(c_mats: c.Mats, allocator: std.mem.Allocator) !Mats {
    return try utils.fromCStructsToArrayList(c_mats.mats, c_mats.length, Self, allocator);
}

pub fn deinitArrayList(mats: *Mats) void {
    mats.deinit();
}

pub fn toCStructs(mats: []const Self) !c.Mats {
    const len = @intCast(i32, mats.len);
    var c_mats = c.Mats_New(len);
    if (c_mats.length != len) return error.AllocationError;
    _ = try epnn(c_mats.mats);
    {
        var i: usize = 0;
        while (i < mats.len) : (i += 1) {
            c_mats.mats[i] = mats[i].toC();
        }
    }
    return c_mats;
}

pub fn deinitCStructs(c_mats: c.Mats) void {
    c.Mats_Close(c_mats);
}

pub const Mats = std.ArrayList(Self);

test "core" {
    _ = @import("mat_test.zig");
}
