const c = @import("c_api.zig");

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

pub const OperationType = enum {
    Add,
    Subtract,
    Multiply,
    Divide,
};

pub const Scalar = struct {
    val1: f64,
    val2: f64,
    val3: f64,
    val4: f64,

    const Self = @This();

    pub fn init(val1: f64, val2: f64, val3: f64, val4: f64) Self {
        return Self{
            .val1 = val1,
            .val2 = val2,
            .val3 = val3,
            .val4 = val4,
        };
    }

    pub fn initFromCScalar(s: c.Scalar) Scalar {
        return Scalar{
            .val1 = s.val1,
            .val2 = s.val2,
            .val3 = s.val3,
            .val4 = s.val4,
        };
    }

    pub fn toCScalar(self: Self) c.Scalar {
        return c.Scalar{
            .val1 = self.val1,
            .val2 = self.val2,
            .val3 = self.val3,
            .val4 = self.val4,
        };
    }
};

pub const Mat = struct {
    ptr: c.Mat,

    const Self = @This();

    fn newMat(m: c.Mat) Self {
        return Self{
            .ptr = m,
        };
    }

    pub fn init() Self {
        return Self.newMat(c.Mat_New());
    }

    pub fn initWithSize(n_rows: c_int, n_cols: c_int, mt: MatType) Self {
        return Self.newMat(c.Mat_NewWithSize(n_rows, n_cols, @enumToInt(mt)));
    }

    pub fn initFromScalar(s: Scalar) Self {
        return Self.newMat(c.Mat_NewFromScalar(Scalar.initFromCScalar(s)));
    }

    pub fn deinit(self: *Self) void {
        _ = c.Mat_Close(self.ptr);
    }

    pub fn copy(self: *Self, dest: *Mat) void {
        _ = c.Mat_CopyTo(self.ptr, dest.*.ptr);
    }

    pub fn cols(self: *Self) i32 {
        return c.Mat_Cols(self.ptr);
    }

    pub fn rows(self: *Self) i32 {
        return c.Mat_Rows(self.ptr);
    }

    pub fn channels(self: *Self) MatChannels {
        return @intToEnum(MatChannels, c.Mat_Channels(self.ptr));
    }

    pub fn getType(self: *Self) MatType {
        var t = c.Mat_Type(self.ptr);
        return @intToEnum(MatType, t);
    }

    pub fn total(self: *Self) i32 {
        return c.Mat_Total(self.ptr);
    }

    pub fn size(self: *Self) []i32 {
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
    pub fn getAt(self: *Self, row: i32, col: i32, comptime T: type) T {
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
    pub fn getAt3(self: *Self, x: i32, y: i32, z: i32, comptime T: type) T {
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
    pub fn isContinuous(self: *Self) bool {
        return c.Mat_IsContinuous(self.ptr);
    }

    pub fn isEmpty(self: *Self) bool {
        return c.Mat_Empty(self.ptr) != 0;
    }

    // Sqrt calculates a square root of array elements.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga186222c3919657890f88df5a1f64a7d7
    //
    pub fn sqrt(self: *Self) Mat {
        return self.newMat(c.Mat_Sqrt(self.ptr));
    }

    // Mean calculates the mean value M of array elements, independently for each channel, and return it as Scalar
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga191389f8a0e58180bb13a727782cd461
    //
    pub fn mean(self: *Self) Scalar {
        return Scalar.initFromCScalar(c.Mat_Mean(self.ptr));
    }

    pub fn calcValueInplace(self: *Self, op: OperationType, v: anytype) void {
        const T = @TypeOf(v);
        return switch (op) {
            .Add => switch (T) {
                u8 => c.Mat_AddUChar(self.ptr, v, op),
                f32 => c.Mat_AddFloat(self.ptr, v, op),
                else => @compileError("not implemented for " ++ @typeName(T)),
            },
            .Subtract => switch (T) {
                u8 => c.Mat_SubtractUChar(self.ptr, v, op),
                f32 => c.Mat_SubtractFloat(self.ptr, v, op),
                else => @compileError("not implemented for " ++ @typeName(T)),
            },
            .Multiply => switch (T) {
                u8 => c.Mat_MultiplyUChar(self.ptr, v, op),
                f32 => c.Mat_MultiplyFloat(self.ptr, v, op),
                else => @compileError("not implemented for " ++ @typeName(T)),
            },
            .Divide => switch (T) {
                u8 => c.Mat_DivideUChar(self.ptr, v, op),
                f32 => c.Mat_DivideFloat(self.ptr, v, op),
                else => @compileError("not implemented for " ++ @typeName(T)),
            },
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
    pub fn calcMat(self: *Self, m: Mat, dest: *Mat, op: OperationType) void {
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
    pub fn addMatWeighted(self: *Self, alpha: f64, m: Mat, beta: f64) Mat {
        var dest = self.init();
        _ = c.Mat_AddWeighted(self.ptr, alpha, m.ptr, beta, dest.ptr);
        return dest;
    }

    pub fn dataPtr(self: *Self, comptime T: type) ![]T {
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
};

// will not implement them. you can call them directly from C-headers
// pub extern fn Mats_get(mats: struct_Mats, i: c_int) Mat;

// pub extern fn Mat_NewWithSizes(sizes: struct_IntVector, @"type": c_int) Mat;
// pub extern fn Mat_NewWithSizesFromScalar(sizes: IntVector, @"type": c_int, ar: Scalar) Mat;
// pub extern fn Mat_NewWithSizesFromBytes(sizes: IntVector, @"type": c_int, buf: struct_ByteArray) Mat;

// pub extern fn Mats_Close(mats: struct_Mats) void;
// pub extern fn Mat_Clone(m: Mat) Mat;
// pub extern fn KeyPoints_Close(ks: struct_KeyPoints) void;
// pub extern fn Points_Close(ps: struct_Points) void;
// pub extern fn MultiDMatches_get(mds: struct_MultiDMatches, index: c_int) struct_DMatches;

// will implement in the future.
// pub extern fn toByteArray(buf: [*c]const u8, len: c_int) struct_ByteArray;
// pub extern fn ByteArray_Release(buf: struct_ByteArray) void;
// pub extern fn Contours_Close(cs: struct_Contours) void;
// pub extern fn Rects_Close(rs: struct_Rects) void;
// pub extern fn Point_Close(p: struct_Point) void;
// pub extern fn DMatches_Close(ds: struct_DMatches) void;
// pub extern fn MultiDMatches_Close(mds: struct_MultiDMatches) void;
// pub extern fn Mat_NewWithSizeFromScalar(ar: Scalar, rows: c_int, cols: c_int, @"type": c_int) Mat;
// pub extern fn Mat_NewFromBytes(rows: c_int, cols: c_int, @"type": c_int, buf: struct_ByteArray) Mat;
// pub extern fn Mat_FromPtr(m: Mat, rows: c_int, cols: c_int, @"type": c_int, prows: c_int, pcols: c_int) Mat;
// pub extern fn Mat_CopyToWithMask(m: Mat, dst: Mat, mask: Mat) void;
// pub extern fn Mat_ConvertTo(m: Mat, dst: Mat, @"type": c_int) void;
// pub extern fn Mat_ConvertToWithParams(m: Mat, dst: Mat, @"type": c_int, alpha: f32, beta: f32) void;
// pub extern fn Mat_ToBytes(m: Mat) struct_ByteArray;
// pub extern fn Mat_DataPtr(m: Mat) struct_ByteArray;
// pub extern fn Mat_Region(m: Mat, r: Rect) Mat;
// pub extern fn Mat_Reshape(m: Mat, cn: c_int, rows: c_int) Mat;
// pub extern fn Mat_PatchNaNs(m: Mat) void;
// pub extern fn Mat_ConvertFp16(m: Mat) Mat;
// pub extern fn Mat_MeanWithMask(m: Mat, mask: Mat) Scalar;
// pub extern fn Mat_Step(m: Mat) c_int;
// pub extern fn Mat_ElemSize(m: Mat) c_int;
// pub extern fn Eye(rows: c_int, cols: c_int, @"type": c_int) Mat;
// pub extern fn Zeros(rows: c_int, cols: c_int, @"type": c_int) Mat;
// pub extern fn Ones(rows: c_int, cols: c_int, @"type": c_int) Mat;
// pub extern fn Mat_SetTo(m: Mat, value: Scalar) void;
// pub extern fn Mat_SetUChar(m: Mat, row: c_int, col: c_int, val: u8) void;
// pub extern fn Mat_SetUChar3(m: Mat, x: c_int, y: c_int, z: c_int, val: u8) void;
// pub extern fn Mat_SetSChar(m: Mat, row: c_int, col: c_int, val: i8) void;
// pub extern fn Mat_SetSChar3(m: Mat, x: c_int, y: c_int, z: c_int, val: i8) void;
// pub extern fn Mat_SetShort(m: Mat, row: c_int, col: c_int, val: i16) void;
// pub extern fn Mat_SetShort3(m: Mat, x: c_int, y: c_int, z: c_int, val: i16) void;
// pub extern fn Mat_SetInt(m: Mat, row: c_int, col: c_int, val: i32) void;
// pub extern fn Mat_SetInt3(m: Mat, x: c_int, y: c_int, z: c_int, val: i32) void;
// pub extern fn Mat_SetFloat(m: Mat, row: c_int, col: c_int, val: f32) void;
// pub extern fn Mat_SetFloat3(m: Mat, x: c_int, y: c_int, z: c_int, val: f32) void;
// pub extern fn Mat_SetDouble(m: Mat, row: c_int, col: c_int, val: f64) void;
// pub extern fn Mat_SetDouble3(m: Mat, x: c_int, y: c_int, z: c_int, val: f64) void;
// pub extern fn Mat_MultiplyMatrix(x: Mat, y: Mat) Mat;
// pub extern fn Mat_T(x: Mat) Mat;
// pub extern fn LUT(src: Mat, lut: Mat, dst: Mat) void;
// pub extern fn Mat_AbsDiff(src1: Mat, src2: Mat, dst: Mat) void;
// pub extern fn Mat_BitwiseAnd(src1: Mat, src2: Mat, dst: Mat) void;
// pub extern fn Mat_BitwiseAndWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
// pub extern fn Mat_BitwiseNot(src1: Mat, dst: Mat) void;
// pub extern fn Mat_BitwiseNotWithMask(src1: Mat, dst: Mat, mask: Mat) void;
// pub extern fn Mat_BitwiseOr(src1: Mat, src2: Mat, dst: Mat) void;
// pub extern fn Mat_BitwiseOrWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
// pub extern fn Mat_BitwiseXor(src1: Mat, src2: Mat, dst: Mat) void;
// pub extern fn Mat_BitwiseXorWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
// pub extern fn Mat_Compare(src1: Mat, src2: Mat, dst: Mat, ct: c_int) void;
// pub extern fn Mat_BatchDistance(src1: Mat, src2: Mat, dist: Mat, dtype: c_int, nidx: Mat, normType: c_int, K: c_int, mask: Mat, update: c_int, crosscheck: bool) void;
// pub extern fn Mat_BorderInterpolate(p: c_int, len: c_int, borderType: c_int) c_int;
// pub extern fn Mat_CalcCovarMatrix(samples: Mat, covar: Mat, mean: Mat, flags: c_int, ctype: c_int) void;
// pub extern fn Mat_CartToPolar(x: Mat, y: Mat, magnitude: Mat, angle: Mat, angleInDegrees: bool) void;
// pub extern fn Mat_CheckRange(m: Mat) bool;
// pub extern fn Mat_CompleteSymm(m: Mat, lowerToUpper: bool) void;
// pub extern fn Mat_ConvertScaleAbs(src: Mat, dst: Mat, alpha: f64, beta: f64) void;
// pub extern fn Mat_CopyMakeBorder(src: Mat, dst: Mat, top: c_int, bottom: c_int, left: c_int, right: c_int, borderType: c_int, value: Scalar) void;
// pub extern fn Mat_CountNonZero(src: Mat) c_int;
// pub extern fn Mat_DCT(src: Mat, dst: Mat, flags: c_int) void;
// pub extern fn Mat_Determinant(m: Mat) f64;
// pub extern fn Mat_DFT(m: Mat, dst: Mat, flags: c_int) void;
// pub extern fn Mat_Eigen(src: Mat, eigenvalues: Mat, eigenvectors: Mat) bool;
// pub extern fn Mat_EigenNonSymmetric(src: Mat, eigenvalues: Mat, eigenvectors: Mat) void;
// pub extern fn Mat_Exp(src: Mat, dst: Mat) void;
// pub extern fn Mat_ExtractChannel(src: Mat, dst: Mat, coi: c_int) void;
// pub extern fn Mat_FindNonZero(src: Mat, idx: Mat) void;
// pub extern fn Mat_Flip(src: Mat, dst: Mat, flipCode: c_int) void;
// pub extern fn Mat_Gemm(src1: Mat, src2: Mat, alpha: f64, src3: Mat, beta: f64, dst: Mat, flags: c_int) void;
// pub extern fn Mat_GetOptimalDFTSize(vecsize: c_int) c_int;
// pub extern fn Mat_Hconcat(src1: Mat, src2: Mat, dst: Mat) void;
// pub extern fn Mat_Vconcat(src1: Mat, src2: Mat, dst: Mat) void;
// pub extern fn Rotate(src: Mat, dst: Mat, rotationCode: c_int) void;
// pub extern fn Mat_Idct(src: Mat, dst: Mat, flags: c_int) void;
// pub extern fn Mat_Idft(src: Mat, dst: Mat, flags: c_int, nonzeroRows: c_int) void;
// pub extern fn Mat_InRange(src: Mat, lowerb: Mat, upperb: Mat, dst: Mat) void;
// pub extern fn Mat_InRangeWithScalar(src: Mat, lowerb: Scalar, upperb: Scalar, dst: Mat) void;
// pub extern fn Mat_InsertChannel(src: Mat, dst: Mat, coi: c_int) void;
// pub extern fn Mat_Invert(src: Mat, dst: Mat, flags: c_int) f64;
// pub extern fn KMeans(data: Mat, k: c_int, bestLabels: Mat, criteria: TermCriteria, attempts: c_int, flags: c_int, centers: Mat) f64;
// pub extern fn KMeansPoints(pts: PointVector, k: c_int, bestLabels: Mat, criteria: TermCriteria, attempts: c_int, flags: c_int, centers: Mat) f64;
// pub extern fn Mat_Log(src: Mat, dst: Mat) void;
// pub extern fn Mat_Magnitude(x: Mat, y: Mat, magnitude: Mat) void;
// pub extern fn Mat_Max(src1: Mat, src2: Mat, dst: Mat) void;
// pub extern fn Mat_MeanStdDev(src: Mat, dstMean: Mat, dstStdDev: Mat) void;
// pub extern fn Mat_Merge(mats: struct_Mats, dst: Mat) void;
// pub extern fn Mat_Min(src1: Mat, src2: Mat, dst: Mat) void;
// pub extern fn Mat_MinMaxIdx(m: Mat, minVal: [*c]f64, maxVal: [*c]f64, minIdx: [*c]c_int, maxIdx: [*c]c_int) void;
// pub extern fn Mat_MinMaxLoc(m: Mat, minVal: [*c]f64, maxVal: [*c]f64, minLoc: [*c]Point, maxLoc: [*c]Point) void;
// pub extern fn Mat_MixChannels(src: struct_Mats, dst: struct_Mats, fromTo: struct_IntVector) void;
// pub extern fn Mat_MulSpectrums(a: Mat, b: Mat, c: Mat, flags: c_int) void;
// pub extern fn Mat_MultiplyWithParams(src1: Mat, src2: Mat, dst: Mat, scale: f64, dtype: c_int) void;
// pub extern fn Mat_Normalize(src: Mat, dst: Mat, alpha: f64, beta: f64, typ: c_int) void;
// pub extern fn Norm(src1: Mat, normType: c_int) f64;
// pub extern fn NormWithMats(src1: Mat, src2: Mat, normType: c_int) f64;
// pub extern fn Mat_PerspectiveTransform(src: Mat, dst: Mat, tm: Mat) void;
// pub extern fn Mat_Solve(src1: Mat, src2: Mat, dst: Mat, flags: c_int) bool;
// pub extern fn Mat_SolveCubic(coeffs: Mat, roots: Mat) c_int;
// pub extern fn Mat_SolvePoly(coeffs: Mat, roots: Mat, maxIters: c_int) f64;
// pub extern fn Mat_Reduce(src: Mat, dst: Mat, dim: c_int, rType: c_int, dType: c_int) void;
// pub extern fn Mat_Repeat(src: Mat, nY: c_int, nX: c_int, dst: Mat) void;
// pub extern fn Mat_ScaleAdd(src1: Mat, alpha: f64, src2: Mat, dst: Mat) void;
// pub extern fn Mat_SetIdentity(src: Mat, scalar: f64) void;
// pub extern fn Mat_Sort(src: Mat, dst: Mat, flags: c_int) void;
// pub extern fn Mat_SortIdx(src: Mat, dst: Mat, flags: c_int) void;
// pub extern fn Mat_Split(src: Mat, mats: [*c]struct_Mats) void;
// pub extern fn Mat_Trace(src: Mat) Scalar;
// pub extern fn Mat_Transform(src: Mat, dst: Mat, tm: Mat) void;
// pub extern fn Mat_Transpose(src: Mat, dst: Mat) void;
// pub extern fn Mat_PolarToCart(magnitude: Mat, degree: Mat, x: Mat, y: Mat, angleInDegrees: bool) void;
// pub extern fn Mat_Pow(src: Mat, power: f64, dst: Mat) void;
// pub extern fn Mat_Phase(x: Mat, y: Mat, angle: Mat, angleInDegrees: bool) void;
// pub extern fn Mat_Sum(src1: Mat) Scalar;
// pub extern fn TermCriteria_New(typ: c_int, maxCount: c_int, epsilon: f64) TermCriteria;
// pub extern fn GetCVTickCount(...) i64;
// pub extern fn GetTickFrequency(...) f64;
// pub extern fn Mat_rowRange(m: Mat, startrow: c_int, endrow: c_int) Mat;
// pub extern fn Mat_colRange(m: Mat, startrow: c_int, endrow: c_int) Mat;
// pub extern fn PointVector_New(...) PointVector;
// pub extern fn PointVector_NewFromPoints(points: Contour) PointVector;
// pub extern fn PointVector_NewFromMat(mat: Mat) PointVector;
// pub extern fn PointVector_At(pv: PointVector, idx: c_int) Point;
// pub extern fn PointVector_Append(pv: PointVector, p: Point) void;
// pub extern fn PointVector_Size(pv: PointVector) c_int;
// pub extern fn PointVector_Close(pv: PointVector) void;
// pub extern fn PointsVector_New(...) PointsVector;
// pub extern fn PointsVector_NewFromPoints(points: Contours) PointsVector;
// pub extern fn PointsVector_At(psv: PointsVector, idx: c_int) PointVector;
// pub extern fn PointsVector_Append(psv: PointsVector, pv: PointVector) void;
// pub extern fn PointsVector_Size(psv: PointsVector) c_int;
// pub extern fn PointsVector_Close(psv: PointsVector) void;
// pub extern fn Point2fVector_New(...) Point2fVector;
// pub extern fn Point2fVector_Close(pfv: Point2fVector) void;
// pub extern fn Point2fVector_NewFromPoints(pts: Contour2f) Point2fVector;
// pub extern fn Point2fVector_NewFromMat(mat: Mat) Point2fVector;
// pub extern fn Point2fVector_At(pfv: Point2fVector, idx: c_int) Point2f;
// pub extern fn Point2fVector_Size(pfv: Point2fVector) c_int;
// pub extern fn IntVector_Close(ivec: struct_IntVector) void;
// pub extern fn CStrings_Close(cstrs: struct_CStrings) void;
// pub extern fn TheRNG(...) RNG;
// pub extern fn SetRNGSeed(seed: c_int) void;
// pub extern fn RNG_Fill(rng: RNG, mat: Mat, distType: c_int, a: f64, b: f64, saturateRange: bool) void;
// pub extern fn RNG_Gaussian(rng: RNG, sigma: f64) f64;
// pub extern fn RNG_Next(rng: RNG) c_uint;
// pub extern fn RandN(mat: Mat, mean: Scalar, stddev: Scalar) void;
// pub extern fn RandShuffle(mat: Mat) void;
// pub extern fn RandShuffleWithParams(mat: Mat, iterFactor: f64, rng: RNG) void;
// pub extern fn RandU(mat: Mat, low: Scalar, high: Scalar) void;
// pub extern fn copyPointVectorToPoint2fVector(src: PointVector, dest: Point2fVector) void;
// pub extern fn StdByteVectorInitialize(data: ?*anyopaque) void;
// pub extern fn StdByteVectorFree(data: ?*anyopaque) void;
// pub extern fn StdByteVectorLen(data: ?*anyopaque) usize;
// pub extern fn StdByteVectorData(data: ?*anyopaque) [*c]u8;
// pub extern fn Points2fVector_New(...) Points2fVector;
// pub extern fn Points2fVector_NewFromPoints(points: Contours2f) Points2fVector;
// pub extern fn Points2fVector_Size(ps: Points2fVector) c_int;
// pub extern fn Points2fVector_At(ps: Points2fVector, idx: c_int) Point2fVector;
// pub extern fn Points2fVector_Append(psv: Points2fVector, pv: Point2fVector) void;
// pub extern fn Points2fVector_Close(ps: Points2fVector) void;
// pub extern fn Point3fVector_New(...) Point3fVector;
// pub extern fn Point3fVector_NewFromPoints(points: Contour3f) Point3fVector;
// pub extern fn Point3fVector_NewFromMat(mat: Mat) Point3fVector;
// pub extern fn Point3fVector_Append(pfv: Point3fVector, point: Point3f) void;
// pub extern fn Point3fVector_At(pfv: Point3fVector, idx: c_int) Point3f;
// pub extern fn Point3fVector_Size(pfv: Point3fVector) c_int;
// pub extern fn Point3fVector_Close(pv: Point3fVector) void;
// pub extern fn Points3fVector_New(...) Points3fVector;
// pub extern fn Points3fVector_NewFromPoints(points: Contours3f) Points3fVector;
// pub extern fn Points3fVector_Size(ps: Points3fVector) c_int;
// pub extern fn Points3fVector_At(ps: Points3fVector, idx: c_int) Point3fVector;
// pub extern fn Points3fVector_Append(psv: Points3fVector, pv: Point3fVector) void;
// pub extern fn Points3fVector_Close(ps: Points3fVector) void;
