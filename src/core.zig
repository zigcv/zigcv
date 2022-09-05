const c = @import("c_api.zig");
const std = @import("std");
const utils = @import("utils.zig");

// TermCriteriaType for TermCriteria.
//
// For further details, please see:
// https://docs.opencv.org/master/d9/d5d/classcv_1_1TermCriteria.html#a56fecdc291ccaba8aad27d67ccf72c57
//
pub const TermCriteriaType = enum(u2) {

    // Count is the maximum number of iterations or elements to compute.
    count = 1,

    // MaxIter is the maximum number of iterations or elements to compute.
    max_iter = 1,

    // EPS is the desired accuracy or change in parameters at which the
    // iterative algorithm stops.
    eps = 2,
};

pub const Mat = struct {
    ptr: c.Mat,

    const Self = @This();

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

    pub fn init() Self {
        return .{ .ptr = c.Mat_New() };
    }

    pub fn initSize(n_rows: c_int, n_cols: c_int, mt: MatType) Self {
        return .{ .ptr = c.Mat_NewWithSize(n_rows, n_cols, @enumToInt(mt)) };
    }

    pub fn fromScalar(s: Scalar) Self {
        return .{ .ptr = c.Mat_NewFromScalar(Scalar.fromC(s)) };
    }

    pub fn fromC(ptr: c.Mat) !Self {
        if (ptr == null) {
            return error.RuntimeError;
        }
        return Self{ .ptr = ptr };
    }

    pub fn eye(rows_: c_int, cols_: c_int, mt: MatType) Self {
        return .{ .ptr = c.Eye(rows_, cols_, @enumToInt(mt)) };
    }
    pub fn zeros(rows_: c_int, cols_: c_int, mt: MatType) Self {
        return .{ .ptr = c.Zeros(rows_, cols_, @enumToInt(mt)) };
    }
    pub fn ones(rows_: c_int, cols_: c_int, mt: MatType) Self {
        return .{ .ptr = c.Ones(rows_, cols_, @enumToInt(mt)) };
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

    pub fn copyToWithMask(self: Self, dest: *Mat, mask: Mat) void {
        _ = c.Mat_CopyToWithMask(self.ptr, dest.*.ptr, mask.ptr);
    }

    pub fn clone(self: Self) Self {
        return .{ .ptr = c.Mat_Clone(self.ptr) };
    }

    pub fn converTo(self: Self, dst: *Mat, mt: MatType) void {
        _ = c.Mat_ConvertTo(self.ptr, dst.*.ptr, @enumToInt(mt));
    }
    pub fn convertToWithParams(self: Self, dst: *Mat, mt: MatType, alpha: f32, beta: f32) void {
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

    pub fn step(self: Self) i32 {
        return c.Mat_Step(self.ptr);
    }

    pub fn elemSize(self: Self) c_int {
        return c.Mat_ElemSize(self.ptr);
    }

    pub fn total(self: Self) i32 {
        return c.Mat_Total(self.ptr);
    }

    pub fn size(self: Self) []c_int {
        var return_v: c.IntVector = undefined;
        _ = c.Mat_Size(self.ptr, &return_v);
        return std.mem.span(return_v.val);
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

    pub fn setTo(self: *Self, value: Scalar) void {
        _ = c.Mat_SetTo(self.ptr, value.toC());
    }

    pub fn set(self: *Self, row_: c_int, col_: c_int, val: anytype, comptime T: type) void {
        _ = switch (T) {
            u8 => c.Mat_SetUChar(self.ptr, row_, col_, @intCast(T, val)),
            i8 => c.Mat_SetSChar(self.ptr, row_, col_, @intCast(T, val)),
            i16 => c.Mat_SetShort(self.ptr, row_, col_, @intCast(T, val)),
            i32 => c.Mat_SetInt(self.ptr, row_, col_, @intCast(T, val)),
            f32 => c.Mat_SetFloat(self.ptr, row_, col_, @floatCast(T, val)),
            f64 => c.Mat_SetDouble(self.ptr, row_, col_, @floatCast(T, val)),
            else => @compileError("not implemented for " ++ @typeName(T)),
        };
    }

    pub fn set3(self: *Self, x: c_int, y: c_int, z: c_int, val: anytype, comptime T: type) void {
        _ = switch (T) {
            u8 => c.Mat_SetUChar3(self.ptr, x, y, z, @intCast(T, val)),
            i8 => c.Mat_SetSChar3(self.ptr, x, y, z, @intCast(T, val)),
            i16 => c.Mat_SetShort3(self.ptr, x, y, z, @intCast(T, val)),
            i32 => c.Mat_SetInt3(self.ptr, x, y, z, @intCast(T, val)),
            f32 => c.Mat_SetFloat3(self.ptr, x, y, z, @floatCast(T, val)),
            f64 => c.Mat_SetDouble3(self.ptr, x, y, z, @floatCast(T, val)),
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
        return Scalar.fromC(c.Mat_Mean(self.ptr));
    }

    // ConvertFp16 converts a Mat to half-precision floating point.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9c25d9ef44a2a48ecc3774b30cb80082
    //
    pub fn convertFp16(self: Self) Mat {
        return .{ .ptr = c.Mat_ConvertFp16(self.ptr) };
    }

    // MeanWithMask calculates the mean value M of array elements,independently for each channel,
    // and returns it as Scalar vector while applying the mask.
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga191389f8a0e58180bb13a727782cd461
    //
    pub fn meanWithMask(self: Self, mask: Mat) Scalar {
        return Scalar.fromC(c.Mat_MeanWithMask(self.ptr, mask.ptr));
    }

    pub fn calcValueInplace(self: *Self, v: anytype, comptime op: OperationType) void {
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
        return self.calcValueInplace(v, .add);
    }

    pub fn subtractValueInplace(self: *Self, v: anytype) void {
        return self.calcValueInplace(v, .subtract);
    }

    pub fn multiplyValueInplace(self: *Self, v: anytype) void {
        return self.calcValueInplace(v, .multiply);
    }

    pub fn divideValueInplace(self: *Self, v: anytype) void {
        return self.calcValueInplace(v, .divide);
    }

    pub fn calcMat(self: Self, m: Mat, dest: *Mat, op: OperationType) void {
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
    pub fn addMat(self: Self, m: Mat, dest: *Mat) void {
        return self.calcMat(m, dest, .add);
    }

    // Subtract calculates the per-element subtraction of two arrays or an array and a scalar.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#gaa0f00d98b4b5edeaeb7b8333b2de353b
    //
    pub fn subtractMat(self: Self, m: Mat, dest: *Mat) void {
        return self.calcMat(m, dest, .subtract);
    }

    // Multiply calculates the per-element scaled product of two arrays.
    // Both input arrays must be of the same size and the same type.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga979d898a58d7f61c53003e162e7ad89f
    //
    pub fn multiplyMat(self: Self, m: Mat, dest: *Mat) void {
        return self.calcMat(m, dest, .multiply);
    }

    // Divide performs the per-element division
    // on two arrays or an array and a scalar.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6db555d30115642fedae0cda05604874
    //
    pub fn divideMat(self: Self, m: Mat, dest: *Mat) void {
        return self.calcMat(m, dest, .divide);
    }

    // AbsDiff calculates the per-element absolute difference between two arrays
    // or between an array and a scalar.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6fef31bc8c4071cbc114a758a2b79c14
    //
    pub fn absDiff(self: Self, m: Mat, dest: *Mat) void {
        _ = c.Mat_AbsDiff(self.ptr, m.ptr, dest.*.ptr);
    }

    // Eigen calculates eigenvalues and eigenvectors of a symmetric matrix.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9fa0d58657f60eaa6c71f6fbb40456e3
    //
    pub fn eigen(self: Self, eigenvalues: *Mat, eigenvectors: *Mat) bool {
        return c.Mat_Eigen(self.ptr, eigenvalues.*.ptr, eigenvectors.*.ptr);
    }

    // EigenNonSymmetric calculates eigenvalues and eigenvectors of a non-symmetric matrix (real eigenvalues only).
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#gaf51987e03cac8d171fbd2b327cf966f6
    //
    pub fn eigenNonSymmetric(self: Self, eigenvalues: *Mat, eigenvectors: *Mat) void {
        return c.Mat_EigenNonSymmetric(self.ptr, eigenvalues.*.ptr, eigenvectors.*.ptr);
    }

    // Exp calculates the exponent of every array element.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga3e10108e2162c338f1b848af619f39e5
    //
    pub fn exp(self: Self, dest: *Mat) void {
        return c.Mat_Exp(self.ptr, dest.*.ptr);
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

    pub fn bitwise(self: Self, m: Mat, dest: *Mat, comptime op: BitOperationType) void {
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
    pub fn bitwiseAnd(self: Self, m: Mat, dest: *Mat) void {
        return self.bitwise(m, dest, .and_);
    }

    // BitwiseNot inverts every bit of an array.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga0002cf8b418479f4cb49a75442baee2f
    //
    pub fn bitwiseNot(self: Self, dest: *Mat) void {
        return self.bitwise(self, dest, .not_);
    }

    // BitwiseOr calculates the per-element bit-wise disjunction of two arrays
    // or an array and a scalar.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#gab85523db362a4e26ff0c703793a719b4
    //

    pub fn bitwiseOr(self: Self, m: Mat, dest: *Mat) void {
        return self.bitwise(m, dest, .or_);
    }

    // BitwiseXor calculates the per-element bit-wise "exclusive or" operation
    // on two arrays or an array and a scalar.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga84b2d8188ce506593dcc3f8cd00e8e2c
    //
    pub fn bitwiseXor(self: Self, m: Mat, dest: *Mat) void {
        return self.bitwise(m, dest, .xor_);
    }

    pub fn bitwiseWithMask(self: Self, m: Mat, dest: *Mat, mask: Mat, comptime op: BitOperationType) void {
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
    pub fn bitwiseAndWithMask(self: Self, m: Mat, dest: *Mat, mask: Mat) void {
        return self.bitwiseWithMask(m, dest, mask, .and_);
    }

    // BitwiseNotWithMask inverts every bit of an array. It has an additional parameter for a mask.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga0002cf8b418479f4cb49a75442baee2f
    //
    pub fn bitwiseNotWithMask(self: Self, dest: *Mat, mask: Mat) void {
        return self.bitwiseWithMask(self, dest, mask, .not_);
    }

    // BitwiseOrWithMask calculates the per-element bit-wise disjunction of two arrays
    // or an array and a scalar. It has an additional parameter for a mask.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#gab85523db362a4e26ff0c703793a719b4
    //
    pub fn bitwiseOrWithMask(self: Self, m: Mat, dest: *Mat, mask: Mat) void {
        return self.bitwiseWithMask(m, dest, mask, .or_);
    }

    // BitwiseXorWithMask calculates the per-element bit-wise "exclusive or" operation
    // on two arrays or an array and a scalar. It has an additional parameter for a mask.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga84b2d8188ce506593dcc3f8cd00e8e2c
    //
    pub fn bitwiseXorWithMask(self: Self, m: Mat, dest: *Mat, mask: Mat) void {
        return self.bitwiseWithMask(m, dest, mask, .xor_);
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
    pub fn reshape(self: Self, cn: c_int, rows_: c_int) Self {
        return .{ .ptr = c.Mat_Reshape(self.ptr, cn, rows_) };
    }

    // Region returns a new Mat that points to a region of this Mat. Changes made to the
    // region Mat will affect the original Mat, since they are pointers to the underlying
    // OpenCV Mat object.
    pub fn region(self: Self, r: Rect) Self {
        return .{ .ptr = c.Mat_Region(self.ptr, r.toC()) };
    }

    // T  transpose matrix
    // https://docs.opencv.org/4.1.2/d3/d63/classcv_1_1Mat.html#aaa428c60ccb6d8ea5de18f63dfac8e11
    pub fn t(self: Self) Self {
        return .{ .ptr = c.Mat_T(self.ptr) };
    }

    // Transpose transposes a matrix.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga46630ed6c0ea6254a35f447289bd7404
    //
    pub fn transpose(self: Self, dst: *Mat) Self {
        _ = c.Mat_Transpose(self.ptr, dst.*.ptr);
    }

    // LUT performs a look-up table transform of an array.
    //
    // The function LUT fills the output array with values from the look-up table.
    // Indices of the entries are taken from the input array.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#gab55b8d062b7f5587720ede032d34156f
    pub fn lut(src: Mat, lut_: Mat, dst: *Mat) void {
        c.LUT(src.ptr, lut_.ptr, dst.*.ptr);
    }

    // Sum calculates the per-channel pixel sum of an image.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga716e10a2dd9e228e4d3c95818f106722
    //
    //     pub extern fn Mat_Sum(src1: Mat) Scalar;
    pub fn sum(self: Self) Scalar {
        return Scalar.fromC(c.Mat_Sum(self.ptr));
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
    pub fn solve(self: Self, src2: Mat, dst: *Mat, flag: SolveDecompositionFlag) bool {
        return c.Mat_Solve(self.ptr, src2.ptr, dst.*.ptr, @enumToInt(flag));
    }
    // SolveCubic finds the real roots of a cubic equation.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga1c3b0b925b085b6e96931ee309e6a1da
    //
    pub fn solveCubic(self: Self, roots: *Mat) bool {
        return c.Mat_SolveCubic(self.ptr, roots.*.ptr);
    }

    // SolvePoly finds the real or complex roots of a polynomial equation.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#gac2f5e953016fabcdf793d762f4ec5dce
    //
    pub fn solvePoly(self: Self, roots: *Mat, max_iters: c_int) bool {
        return c.Mat_SolvePoly(self.ptr, roots.*.ptr, max_iters);
    }

    pub fn toArrayList(c_mats: c.Mats, allocator: std.mem.Allocator) !Mats {
        return try utils.fromCStructsToArrayList(c_mats.mats, c_mats.length, Self, allocator);
    }

    pub fn deinitArrayList(mats: *Mats) void {
        mats.deinit();
    }

    pub fn toCStructs(mats: []const Mat, allocator: std.mem.Allocator) !c.Mats {
        const len = @intCast(usize, mats.len);
        var c_mats = try std.ArrayList(c.Mat).initCapacity(allocator, len);
        {
            var i: usize = 0;
            while (i < len) : (i += 1) {
                c_mats[i] = mats[i].ptr;
            }
        }
        return .{
            .mats = c_mats.toOwnedSliceSentinel(0),
            .length = mats.len,
        };
    }

    pub fn deinitCStructs(c_mats: c.Mats, allocator: std.mem.Allocator) void {
        allocator.free(c_mats.mats);
    }
};

pub const Mats = std.ArrayList(Mat);

pub const Point = struct {
    x: c_int,
    y: c_int,

    const Self = @This();

    pub fn int(x: c_int, y: c_int) Self {
        return .{ .x = x, .y = y };
    }

    pub fn fromC(p: c.Point) Self {
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

    pub fn fromC(p: c.Point2f) Self {
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

    pub fn fromC(p: c.Point3f) Self {
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

    pub fn init() Self {
        return .{ .ptr = c.PointVector_New() };
    }

    pub fn fromMat(mat: Mat) !Self {
        if (mat.ptr == null) {
            return error.RuntimeError;
        }
        return .{ .ptr = c.PointVector_NewFromMat(mat.ptr) };
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
        }
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

    pub fn init() Self {
        return .{ .ptr = c.Point2fVector_New() };
    }

    pub fn fromMat(mat: Mat) !Self {
        if (mat.ptr == null) {
            return error.RuntimeError;
        }
        return .{ .ptr = c.Point2fVector_NewFromMat(mat.ptr) };
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
        }
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

    pub fn fromC(s: c.Scalar) Self {
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

    pub fn fromC(kp: c.KeyPoint) Self {
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

    pub fn toArrayList(c_kps: c.KeyPoints, allocator: std.mem.Allocator) !KeyPoints {
        return try utils.fromCStructsToArrayList(c_kps.keypoints, c_kps.length, Self, allocator);
    }
};

pub const KeyPoints = std.ArrayList(KeyPoint);

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

    pub fn fromC(r: c.Rect) Self {
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

    pub fn fromC(r: c.RotatedRect) Self {
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
    width: c_int,
    height: c_int,

    const Self = @This();

    pub fn init(width: c_int, height: c_int) Self {
        return .{
            .width = width,
            .height = height,
        };
    }

    pub fn fromC(r: c.Size) Self {
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

    pub fn fromC(ptr: c.RNG) Self {
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

test "core" {
    _ = @import("core_test.zig");
}

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
