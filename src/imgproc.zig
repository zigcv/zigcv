const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const epnn = utils.ensurePtrNotNull;
const Mat = core.Mat;
const Size = core.Size;
const Scalar = core.Scalar;
const Color = core.Color;
const Rect = core.Rect;
const Point = core.Point;
const Point2f = core.Point2f;
const PointVector = core.PointVector;
const PointsVector = core.PointsVector;
const Point2fVector = core.Point2fVector;
const RotatedRect = core.RotatedRect;
const MatType = Mat.MatType;
const TermCriteria = core.TermCriteria;
const ColorConversionCode = @import("imgproc/color_codes.zig").ColorConversionCode;

pub const ConnectedComponentsAlgorithmType = enum(u2) {
    /// SAUF algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity.
    wu = 0,

    /// BBDT algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity.
    default = 1,

    /// BBDT algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity
    grana = 2,
};

pub const ConnectedComponentsType = enum(u3) {
    ///The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
    stat_left = 0,

    ///The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
    stat_top = 1,

    /// The horizontal size of the bounding box.
    stat_width = 2,

    /// The vertical size of the bounding box.
    stat_height = 3,

    /// The total area (in pixels) of the connected component.
    stat_area = 4,

    stat_max = 5,
};

pub const BorderType = enum(u5) {
    /// BorderConstant border type
    constant = 0,

    /// BorderReplicate border type
    replicate = 1,

    /// BorderReflect border type
    reflect = 2,

    /// BorderWrap border type
    wrap = 3,

    /// BorderReflect101 border type
    reflect101 = 4,

    /// BorderTransparent border type
    transparent = 5,

    /// BorderIsolated border type
    isolated = 16,

    pub const default = BorderType.reflect101;
};

pub const MorphType = enum(u3) {
    /// MorphErode operation
    erode = 0,

    /// MorphDilate operation
    dilate = 1,

    /// MorphOpen operation
    open = 2,

    /// MorphClose operation
    close = 3,

    /// MorphGradient operation
    gradient = 4,

    /// MorphTophat operation
    tophat = 5,

    /// MorphBlackhat operation
    blackhat = 6,

    /// MorphHitmiss operation
    hitmiss = 7,
};

pub const HoughMode = enum(u2) {
    /// HoughStandard is the classical or standard Hough transform.
    standard = 0,
    /// HoughProbabilistic is the probabilistic Hough transform (more efficient
    /// in case if the picture contains a few long linear segments).
    probabilistic = 1,
    /// HoughMultiScale is the multi-scale variant of the classical Hough
    /// transform.
    multi_scale = 2,
    /// HoughGradient is basically 21HT, described in: HK Yuen, John Princen,
    /// John Illingworth, and Josef Kittler. Comparative study of hough
    /// transform methods for circle finding. Image and Vision Computing,
    /// 8(1):71–77, 1990.
    gradient = 3,
};

pub const GrabCutMode = enum(u2) {
    /// GCInitWithRect makes the function initialize the state and the mask using the provided rectangle.
    /// After that it runs the itercount iterations of the algorithm.
    init_with_rect = 0,
    /// GCInitWithMask makes the function initialize the state using the provided mask.
    /// GCInitWithMask and GCInitWithRect can be combined.
    /// Then all the pixels outside of the ROI are automatically initialized with GC_BGD.
    init_with_mask = 1,
    /// GCEval means that the algorithm should just resume.
    eval = 2,
    /// GCEvalFreezeModel means that the algorithm should just run a single iteration of the GrabCut algorithm
    /// with the fixed model
    eval_freeze_model = 3,
};

pub const LineType = enum(i6) {
    /// Filled line
    filled = -1,
    /// Line4 4-connected line
    line4 = 4,
    /// Line8 8-connected line
    line8 = 8,
    /// LineAA antialiased line
    line_aa = 16,
};

pub const HersheyFont = enum(u5) {
    /// FontHersheySimplex is normal size sans-serif font.
    simplex = 0,
    /// FontHersheyPlain issmall size sans-serif font.
    plain = 1,
    /// FontHersheyDuplex normal size sans-serif font
    /// (more complex than FontHersheySIMPLEX).
    duplex = 2,
    /// FontHersheyComplex i a normal size serif font.
    complex = 3,
    /// FontHersheyTriplex is a normal size serif font
    /// (more complex than FontHersheyCOMPLEX).
    triplex = 4,
    /// FontHersheyComplexSmall is a smaller version of FontHersheyCOMPLEX.
    complex_small = 5,
    /// FontHersheyScriptSimplex is a hand-writing style font.
    script_simplex = 6,
    /// FontHersheyScriptComplex is a more complex variant of FontHersheyScriptSimplex.
    script_complex = 7,
    /// FontItalic is the flag for italic font.
    italic = 16,
};

pub const InterpolationFlag = enum(u5) {
    /// InterpolationNearestNeighbor is nearest neighbor. (fast but low quality)
    nearest_neighbor = 0,

    /// InterpolationLinear is bilinear interpolation.
    linear = 1,

    /// InterpolationCubic is bicube interpolation.
    cubic = 2,

    /// InterpolationArea uses pixel area relation. It is preferred for image
    /// decimation as it gives moire-free results.
    area = 3,

    /// InterpolationLanczos4 is Lanczos interpolation over 8x8 neighborhood.
    lanczos4 = 4,

    /// InterpolationMax indicates use maximum interpolation.
    max = 7,

    /// WarpFillOutliers fills all of the destination image pixels. If some of them correspond to outliers in the source image, they are set to zero.
    warp_fill_outliers = 8,

    /// WarpInverseMap, inverse transformation.
    warp_inverse_map = 16,

    pub const default = InterpolationFlag.linear;
};

pub const ColormapType = enum(u4) {
    autumn = 0,
    bone = 1,
    jet = 2,
    winter = 3,
    rainbow = 4,
    ocean = 5,
    summer = 6,
    spring = 7,
    cool = 8,
    hsv = 9,
    pink = 10,
    hot = 11,
    parula = 12,
};

pub const HomographyMethod = enum(u4) {
    all_points = 0,
    lmeds = 4,
    ransac = 8,
};

/// DistanceType types for Distance Transform and M-estimatorss
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gaa2bfbebbc5c320526897996aafa1d8eb
pub const DistanceType = enum(u3) {
    user = 0,
    l1 = 1,
    l2 = 2,
    c = 3,
    l12 = 4,
    fair = 5,
    welsch = 6,
    huber = 7,
};

pub const DistanceTransformLabelType = enum(u1) {
    c_comp = 0,
    pixel = 1,
};

pub const DistanceTransformMask = enum(u1) {
    mask_3 = 3,
    mask_5 = 5,
    mask_precise = 0,
};

pub const ThresholdType = enum(u5) {
    /// ThresholdBinary threshold type
    binary = 0,

    /// ThresholdBinaryInv threshold type
    binary_inv = 1,

    /// ThresholdTrunc threshold type
    trunc = 2,

    /// ThresholdToZero threshold type
    to_zero = 3,

    /// ThresholdToZeroInv threshold type
    to_zero_inv = 4,

    /// ThresholdMask threshold type
    mask = 7,

    /// ThresholdOtsu threshold type
    otsu = 8,

    /// ThresholdTriangle threshold type
    triangle = 16,
};

/// AdaptiveThresholdType type of adaptive threshold operation.
pub const AdaptiveThresholdType = enum(u1) {
    //// AdaptiveThresholdMean threshold type
    mean = 0,

    //// AdaptiveThresholdGaussian threshold type
    gaussian = 1,
};

/// RetrievalMode is the mode of the contour retrieval algorithm.
pub const RetrievalMode = enum(u3) {
    /// RetrievalExternal retrieves only the extreme outer contours.
    /// It sets `hierarchy[i][2]=hierarchy[i][3]=-1` for all the contours.
    external = 0,

    /// RetrievalList retrieves all of the contours without establishing
    /// any hierarchical relationships.
    list = 1,

    /// RetrievalCComp retrieves all of the contours and organizes them into
    /// a two-level hierarchy. At the top level, there are external boundaries
    /// of the components. At the second level, there are boundaries of the holes.
    /// If there is another contour inside a hole of a connected component, it
    /// is still put at the top level.
    c_comp = 2,

    /// RetrievalTree retrieves all of the contours and reconstructs a full
    /// hierarchy of nested contours.
    tree = 3,

    /// RetrievalFloodfill lacks a description in the original header.
    floodfill = 4,
};

/// ContourApproximationMode is the mode of the contour approximation algorithm.
pub const ContourApproximationMode = enum(u3) {
    /// ChainApproxNone stores absolutely all the contour points. That is,
    /// any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be
    /// either horizontal, vertical or diagonal neighbors, that is,
    /// max(abs(x1-x2),abs(y2-y1))==1.
    none = 1,

    /// ChainApproxSimple compresses horizontal, vertical, and diagonal segments
    /// and leaves only their end points.
    /// For example, an up-right rectangular contour is encoded with 4 points.
    simple = 2,

    /// ChainApproxTC89L1 applies one of the flavors of the Teh-Chin chain
    /// approximation algorithms.
    tc89L1 = 3,

    /// ChainApproxTC89KCOS applies one of the flavors of the Teh-Chin chain
    /// approximation algorithms.
    tc89kcos = 4,
};

/// CLAHE is a wrapper around the cv::CLAHE algorithm.
pub const CLAHE = struct {
    ptr: c.CLAHE,

    const Self = @This();

    /// NewCLAHE returns a new CLAHE algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/db6/classcv_1_1CLAHE.html
    ///
    pub fn init() !Self {
        const ptr = c.CLAHE_Create();
        return initFromC(ptr);
    }

    /// NewCLAHEWithParams returns a new CLAHE algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/db6/classcv_1_1CLAHE.html
    ///
    pub fn initWithParams(clip_limit: f64, tile_grid_size: Size) !Self {
        const ptr = c.CLAHE_CreateWithParams(clip_limit, tile_grid_size.toC());
        return initFromC(ptr);
    }

    fn initFromC(ptr: c.CLAHE) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close the CLAHE algorithm
    pub fn deinit(self: *Self) void {
        c.CLAHE_Close(self.ptr);
        self.ptr = null;
    }

    pub fn toC(self: Self) c.CLANE {
        return self.ptr;
    }

    /// Apply CLAHE.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/db6/classcv_1_1CLAHE.html#a4e92e0e427de21be8d1fae8dcd862c5e
    ///
    pub fn apply(self: Self, src: Mat, dst: *Mat) void {
        c.CLAHE_Apply(self.ptr, src.toC(), dst.toC());
    }
};

pub fn arcLength(curve: PointVector, is_closed: bool) f64 {
    return c.ArcLength(curve.toC(), is_closed);
}

pub fn approxPolyDP(curve: PointVector, epsilon: f64, closed: bool) !PointVector {
    const ptr = c.ApproxPolyDP(curve.toC(), epsilon, closed);
    return try PointVector.initFromC(ptr);
}

/// CvtColor converts an image from one color space to another.
/// It converts the src Mat image to the dst Mat using the
/// code param containing the desired ColorConversionCode color space.
///
/// For further details, please see:
/// http://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga4e0972be5de079fed4e3a10e24ef5ef0
///
pub fn cvtColor(src: Mat, dst: *Mat, code: ColorConversionCode) void {
    c.CvtColor(src.ptr, dst.*.ptr, @enumToInt(code));
}

pub fn equalizeHist(src: Mat, dst: *Mat) void {
    return c.equalizeHist(src.ptr, dst.*.ptr);
}

/// pub fn calcHist(mats: struct_Mats, chans: IntVector, mask: Mat, hist: Mat, sz: IntVector, rng: FloatVector, acc: bool) void;
/// pub fn calcBackProject(mats: struct_Mats, chans: IntVector, hist: Mat, backProject: Mat, rng: FloatVector, uniform: bool) void;
pub fn compareHist(hist1: Mat, hist2: Mat, method: i32) f64 {
    return c.CompareHist(hist1.ptr, hist2.ptr, method);
}

pub fn convexHull(points: PointVector, hull: *Mat, clockwise: bool, return_points: bool) void {
    _ = c.ConvexHull(points.toC(), hull.*.ptr, clockwise, return_points);
}

pub fn convexityDefects(points: PointVector, hull: Mat, result: *Mat) void {
    _ = c.ConvexityDefects(points.toC(), hull.ptr, result.*.ptr);
}

pub fn bilateralFilter(src: Mat, dst: *Mat, d: i32, sc: f64, ss: f64) void {
    _ = c.BilateralFilter(src.ptr, dst.*.ptr, d, sc, ss);
}

pub fn blur(src: Mat, dst: *Mat, ps: Size) void {
    _ = c.Blur(src.ptr, dst.*.ptr, ps.toC());
}

pub fn boxFilter(src: Mat, dst: *Mat, ddepth: i32, ps: Size) void {
    _ = c.BoxFilter(src.ptr, dst.*.ptr, ddepth, ps.toC());
}

pub fn sqBoxFilter(src: Mat, dst: *Mat, ddepth: i32, ps: Size) void {
    _ = c.SqBoxFilter(src.ptr, dst.*.ptr, ddepth, ps);
}

pub fn dilate(src: Mat, dst: *Mat, kernel: Mat) void {
    _ = c.Dilate(src.ptr, dst.*.ptr, kernel.ptr);
}

pub fn dilateWithParams(src: Mat, dst: *Mat, kernel: Mat, anchor: Point, iterations: BorderType, border_type: BorderType, border_value: Color) void {
    _ = c.DilateWithParams(src.ptr, dst.*.ptr, kernel.ptr, anchor.toC(), @enumToInt(iterations), @enumToInt(border_type), border_value.toScalar.toC());
}

pub fn distanceTransform(src: Mat, dst: *Mat, labels: Mat, distance_type: DistanceType, mask_size: DistanceTransformMask, label_type: DistanceTransformLabelType) void {
    _ = c.DistanceTransform(src.ptr, dst.*.ptr, labels.ptr, @enumToInt(distance_type), @enumToInt(mask_size), @enumToInt(label_type));
}
pub fn erode(src: Mat, dst: *Mat, kernel: Mat) void {
    _ = c.Erode(src.ptr, dst.*.ptr, kernel.ptr);
}

pub fn erodeWithParams(src: Mat, dst: *Mat, kernel: Mat, anchor: Point, iterations: i32, border_type: i32) void {
    _ = c.ErodeWithParams(src.ptr, dst.*.ptr, kernel.ptr, anchor.toC(), iterations, border_type);
}

pub fn matchTemplate(image: Mat, templ: Mat, result: *Mat, method: i32, mask: Mat) void {
    _ = c.MatchTemplate(image.ptr, templ.ptr, result.*.ptr, method, mask.ptr);
}

pub fn pyrDown(src: Mat, dst: *Mat, dstsize: Size, border_type: BorderType) void {
    _ = c.PyrDown(src.ptr, dst.*.ptr, dstsize.toC(), @enumToInt(border_type));
}

pub fn pyrUp(src: Mat, dst: *Mat, dstsize: Size, border_type: BorderType) void {
    _ = c.PyrUp(src.ptr, dst.*.ptr, dstsize.toC(), @enumToInt(border_type));
}

pub fn boundingRect(pts: PointVector) Rect {
    return Rect.fromC(c.BoundingRect(pts.toC()));
}
pub fn boxPoints(rect: RotatedRect, box_pts: *Mat) void {
    _ = c.BoxPoints(rect.toC(), box_pts.*.ptr);
}

pub fn contourArea(pts: PointVector) f64 {
    return c.ContourArea(pts.toC());
}
pub fn minAreaRect(pts: PointVector) RotatedRect {
    return RotatedRect.fromC(c.MinAreaRect(pts.toC()));
}

pub fn fitEllipse(pts: PointVector) RotatedRect {
    return RotatedRect.fromC(c.FitEllipse(pts.toC()));
}

/// FindContours finds contours in a binary image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga95f5b48d01abc7c2e0732db24689837b
///
//     pub extern fn FindContours(src: Mat, hierarchy: Mat, mode: c_int, method: c_int) PointsVector;
pub fn findContours(src: Mat, mode: RetrievalMode, method: ContourApproximationMode) !PointsVector {
    var hierarchy = try Mat.init();
    defer hierarchy.deinit();
    return try findContoursWithParams(src, &hierarchy, mode, method);
}

/// FindContoursWithParams finds contours in a binary image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
///
pub fn findContoursWithParams(src: Mat, hierarchy: *Mat, mode: RetrievalMode, method: ContourApproximationMode) !PointsVector {
    return try PointsVector.initFromC(c.FindContours(src.ptr, hierarchy.*.ptr, @enumToInt(mode), @enumToInt(method)));
}

pub fn pointPolygonTest(pts: PointVector, pt: Point, measure_dist: bool) f64 {
    return c.PointPolygonTest(pts.toC(), pt.toC(), measure_dist);
}

/// ConnectedComponents computes the connected components labeled image of boolean image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5
///
pub fn connectedComponents(src: Mat, labels: *Mat) i32 {
    return c.ConnectedComponents(
        src.ptr,
        labels.*.ptr,
        8,
        @enumToInt(MatType.cv32sc1),
        @enumToInt(ConnectedComponentsAlgorithmType.default),
    );
}

// ConnectedComponents computes the connected components labeled image of boolean image.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5
//
pub fn connectedComponentsWithParams(src: Mat, labels: *Mat, connectivity: i32, ltype: MatType, ccltype: ConnectedComponentsAlgorithmType) i32 {
    return c.ConnectedComponents(src.ptr, labels.*.ptr, connectivity, @enumToInt(ltype), @enumToInt(ccltype));
}

// ConnectedComponentsWithStats computes the connected components labeled image of boolean
// image and also produces a statistics output for each label.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
//
pub fn connectedComponentsWithStats(src: Mat, labels: *Mat, stats: *Mat, centroids: *Mat, connectivity: i32, ltype: MatType, ccltype: ConnectedComponentsType) i32 {
    return c.ConnectedComponentsWithStats(src.ptr, labels.*.ptr, stats.*.ptr, centroids.*.ptr, connectivity, @enumToInt(ltype), @enumToInt(ccltype));
}

// GaussianBlur blurs an image Mat using a Gaussian filter.
// The function convolves the src Mat image into the dst Mat using
// the specified Gaussian kernel params.
//
// For further details, please see:
// http://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
//
pub fn gaussianBlur(src: Mat, dst: *Mat, ps: Size, sigma_x: f64, sigma_y: f64, border_type: BorderType) void {
    _ = c.GaussianBlur(src.ptr, dst.*.ptr, ps.toC(), sigma_x, sigma_y, @enumToInt(border_type));
}

// GetGaussianKernel returns Gaussian filter coefficients.
//
// For further details, please see:
// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
pub fn getGaussianKernel(ksize: i32, sigma: f64) !Mat {
    return try Mat.initFromC(c.GetGaussianKernel(ksize, sigma, @enumToInt(MatType.cv64fc1)));
}

pub fn getGaussianKernelWithParams(ksize: i32, sigma: f64, ktype: MatType) !Mat {
    return try Mat.initFromC(c.GetGaussianKernel(ksize, sigma, @enumToInt(ktype)));
}

// Laplacian calculates the Laplacian of an image.
//
// For further details, please see:
// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6
//
pub fn laplacian(src: Mat, dst: *Mat, d_depth: i32, k_size: i32, scale: f64, delta: f64, border_type: BorderType) void {
    _ = c.Laplacian(src.ptr, dst.*.ptr, d_depth, k_size, scale, delta, @enumToInt(border_type));
}

// Scharr calculates the first x- or y- image derivative using Scharr operator.
//
// For further details, please see:
// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaa13106761eedf14798f37aa2d60404c9
//
pub fn scharr(src: Mat, dst: *Mat, d_depth: i32, dx: i32, dy: i32, scale: f64, delta: f64, border_type: BorderType) void {
    _ = c.Scharr(src.ptr, dst.*.ptr, d_depth, dx, dy, scale, delta, @enumToInt(border_type));
}

pub fn getStructuringElement(shape: i32, ksize: Size) !Mat {
    return try Mat.initFromC(c.GetStructuringElement(shape, ksize.toC()));
}
pub fn morphologyDefaultBorderValue() Scalar {
    return Scalar.fromC(c.MorphologyDefaultBorderValue());
}

pub fn morphologyEx(src: Mat, dst: *Mat, op: i32, kernel: Mat) void {
    _ = c.MorphologyEx(src.ptr, dst.*.ptr, op, kernel.ptr);
}

pub fn morphologyExWithParams(src: Mat, dst: *Mat, op: i32, kernel: Mat, pt: Point, iterations: i32, border_type: BorderType) void {
    _ = c.MorphologyExWithParams(src.ptr, dst.*.ptr, op, kernel.ptr, pt.toC(), iterations, @enumToInt(border_type));
}

pub fn medianBlur(src: Mat, dst: *Mat, ksize: i32) void {
    _ = c.MedianBlur(src.ptr, dst.*.ptr, ksize);
}

pub fn canny(src: Mat, edges: *Mat, t1: f64, t2: f64) void {
    _ = c.Canny(src.ptr, edges.*.ptr, t1, t2);
}

pub fn cornerSubPix(img: Mat, corners: *Mat, winSize: Size, zeroZone: Size, criteria: TermCriteria) void {
    _ = c.CornerSubPix(img.ptr, corners.*.ptr, winSize.toC(), zeroZone.toC(), @enumToInt(criteria));
}
pub fn goodFeaturesToTrack(img: Mat, corners: Mat, maxCorners: i32, quality: f64, minDist: f64) void {
    _ = c.GoodFeaturesToTrack(img.ptr, corners.*.ptr, maxCorners, quality, minDist);
}

pub fn grabCut(img: Mat, mask: Mat, rect: Rect, bgd_model: *Mat, fgd_model: *Mat, iter_count: i32, mode: GrabCutMode) void {
    _ = c.GrabCut(img.ptr, mask.*.ptr, rect.toC(), bgd_model.*.ptr, fgd_model.*.ptr, iter_count, @enumToInt(mode));
}

pub fn houghCircles(src: Mat, circles: *Mat, method: HoughMode, dp: f64, min_dist: f64) void {
    _ = c.HoughCircles(src.ptr, circles.*.ptr, @enumToInt(method), dp, min_dist);
}

pub fn houghCirclesWithParams(src: Mat, circles: *Mat, method: HoughMode, dp: f64, min_dist: f64, param1: f64, param2: f64, min_radius: i32, max_radius: i32) void {
    _ = c.HoughCirclesWithParams(src.ptr, circles.*.ptr, @enumToInt(method), dp, min_dist, param1, param2, min_radius, max_radius);
}

pub fn houghLines(src: Mat, lines: *Mat, rho: f64, theta: f64, threshold_int: i32) void {
    _ = c.HoughLines(src.ptr, lines.*.ptr, rho, theta, threshold_int);
}

pub fn houghLinesP(src: Mat, lines: *Mat, rho: f64, theta: f64, threshold_int: i32) void {
    _ = c.HoughLinesP(src.ptr, lines.*.ptr, rho, theta, threshold_int);
}

// Integral calculates one or more integral images for the source image.
// For further details, please see:
// https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga97b87bec26908237e8ba0f6e96d23e28
//
pub fn integral(src: Mat, sum: *Mat, sqsum: Mat, tilted: Mat) void {
    _ = c.Integral(src.ptr, sum.*.ptr, sqsum.ptr, tilted.ptr);
}

pub fn threshold(src: Mat, dst: *Mat, thresh: f64, maxvalue: f64, typ: i32) f64 {
    return c.Threshold(src.ptr, dst.*.ptr, thresh, maxvalue, typ);
}

pub fn adaptiveThreshold(src: Mat, dst: *Mat, max_value: f64, adaptive_type: AdaptiveThresholdType, type_: ThresholdType, block_size: i32, C: f64) void {
    _ = c.AdaptiveThreshold(src.ptr, dst.*.ptr, max_value, @enumToInt(adaptive_type), @enumToInt(type_), block_size, C);
}
// ArrowedLine draws a arrow segment pointing from the first point
// to the second one.
//
// For further details, please see:
// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga0a165a3ca093fd488ac709fdf10c05b2
//
pub fn arrowedLine(img: *Mat, pt1: Point, pt2: Point, color: Color, thickness: i32) void {
    _ = c.ArrowedLine(img.*.ptr, pt1.toC(), pt2.toC(), color.toScalar().toC(), thickness);
}

pub fn circle(img: *Mat, center: Point, radius: i32, color: Color, thickness: i32) void {
    _ = c.Circle(img.*.ptr, center.toC(), radius, color.toScalar().toC(), thickness);
}

pub fn circleWithParams(img: *Mat, center: Point, radius: i32, color: Color, thickness: i32, line_type: LineType, shift: i32) void {
    _ = c.CircleWithParams(img.*.ptr, center.toC(), radius, color.toScalar().toC(), thickness, @enumToInt(line_type), shift);
}

// Ellipse draws a simple or thick elliptic arc or fills an ellipse sector.
//
// For further details, please see:
// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
//
pub fn ellipse(img: *Mat, center: Point, axes: Point, angle: f64, start_angle: f64, end_angle: f64, color: Color, thickness: i32) void {
    _ = c.Ellipse(img.*.ptr, center.toC(), axes.toC(), angle, start_angle, end_angle, color.toScalar().toC(), thickness);
}

pub fn ellipseWithParams(img: *Mat, center: Point, axes: Point, angle: f64, start_angle: f64, end_angle: f64, color: Color, thickness: i32, line_type: LineType, shift: i32) void {
    _ = c.EllipseWithParams(img.*.ptr, center.toC(), axes.toC(), angle, start_angle, end_angle, color.toScalar().toC(), thickness, @enumToInt(line_type), shift);
}

pub fn line(img: *Mat, pt1: Point, pt2: Point, color: Color, thickness: i32) void {
    _ = c.Line(img.ptr, pt1.toC(), pt2.toC(), color.toScalar().toC(), thickness);
}

pub fn rectangle(img: *Mat, rect: Rect, color: Color, thickness: i32) void {
    _ = c.Rectangle(img.*.ptr, rect.toC(), color.toScalar().toC(), thickness);
}

pub fn rectangleWithParams(img: *Mat, rect: Rect, color: Color, thickness: i32, line_type: LineType, shift: i32) void {
    _ = c.RectangleWithParams(img.*.ptr, rect.toC(), color.toScalar().toC(), thickness, @enumToInt(line_type), shift);
}

// pub fn fillPoly(img: *Mat, points: PointsVector, color: Color) void;
// pub fn fillPolyWithParams(img: *Mat, points: PointsVector, color: Color, line_type: LineType, shift: c_int, offset: Point) void;
// pub fn polylines(img: Mat, points: PointsVector, isClosed: bool, color: Scalar, thickness: c_int) void;

/// GetTextSize calculates the width and height of a text string.
/// It returns an image.Point with the size required to draw text using
/// a specific font face, scale, and thickness.
///
/// For further details, please see:
/// http://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga3d2abfcb995fd2db908c8288199dba82
///
pub fn getTextSize(text: []const u8, font_face: HersheyFont, font_scale: f64, thickness: i32) Size {
    return Size.initFromC(c.GetTextSize(utils.castZigU8ToC(text), @enumToInt(font_face), font_scale, thickness));
}

/// GetTextSizeWithBaseline calculates the width and height of a text string including the basline of the text.
/// It returns an image.Point with the size required to draw text using
/// a specific font face, scale, and thickness as well as its baseline.
///
/// For further details, please see:
/// http://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga3d2abfcb995fd2db908c8288199dba82
///
pub fn getTextSizeWithBaseline(text: []const u8, font_face: HersheyFont, font_scale: f64, thickness: i32) struct { size: Size, baseline: i32 } {
    var baseline: i32 = 0;
    const size = Size.initFromC(c.GetTextSizeWithBaseline(utils.castZigU8ToC(text), @enumToInt(font_face), font_scale, thickness, &baseline));
    return .{
        .size = size,
        .baseline = baseline,
    };
}

// PutText draws a text string.
// It renders the specified text string into the img Mat at the location
// passed in the "org" param, using the desired font face, font scale,
// color, and line thinkness.
//
// For further details, please see:
// http://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
//
pub fn putText(img: *Mat, text: []const u8, org: Point, font_face: HersheyFont, font_scale: f64, color: Color, thickness: i32) void {
    _ = c.PutText(img.*.ptr, utils.castZigU8ToC(text), org.toC(), @enumToInt(font_face), font_scale, color.toScalar().toC(), thickness);
}

// PutTextWithParams draws a text string.
// It renders the specified text string into the img Mat at the location
// passed in the "org" param, using the desired font face, font scale,
// color, and line thinkness.
//
// For further details, please see:
// http://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
//
pub fn putTextWithParams(img: *Mat, text: []const u8, org: Point, font_face: HersheyFont, font_scale: f64, color: Color, thickness: i32, line_type: LineType, bottom_left_origin: bool) void {
    _ = c.PutTextWithParams(img.*.ptr, utils.castZigU8ToC(text), org.toC(), @enumToInt(font_face), font_scale, color.toScalar().toC(), thickness, @enumToInt(line_type), bottom_left_origin);
}

/// Resize resizes an image.
/// It resizes the image src down to or up to the specified size, storing the
/// result in dst. Note that src and dst may be the same image. If you wish to
/// scale by factor, an empty sz may be passed and non-zero fx and fy. Likewise,
/// if you wish to scale to an explicit size, a non-empty sz may be passed with
/// zero for both fx and fy.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
pub fn resize(src: Mat, dst: *Mat, sz: Size, fx: f64, fy: f64, interp: InterpolationFlag) void {
    _ = c.Resize(src.ptr, dst.*.ptr, sz.toC(), fx, fy, @enumToInt(interp));
}

pub fn getRectSubPix(src: Mat, patch_size: Size, center: Point, dst: *Mat) void {
    _ = c.GetRectSubPix(src.ptr, patch_size.toC(), center.toC(), dst.*.ptr);
}

pub fn getRotationMatrix2D(center: Point, angle: f64, scale: f64) !Mat {
    return try Mat.initFromC(c.GetRotationMatrix2D(center.toC(), angle, scale));
}

pub fn warpAffine(src: Mat, dst: *Mat, rot_mat: Mat, dsize: Size) void {
    _ = c.WarpAffine(src.ptr, dst.*.ptr, rot_mat.ptr, dsize.toC());
}

pub fn warpAffineWithParams(src: Mat, dst: *Mat, rot_mat: Mat, dsize: Size, flags: InterpolationFlag, border_mode: BorderType, border_value: Color) void {
    _ = c.WarpAffineWithParams(src.ptr, dst.*.ptr, rot_mat.ptr, dsize.toC(), @enumToInt(flags), @enumToInt(border_mode), border_value.toScalar().toC());
}

pub fn warpPerspective(src: Mat, dst: *Mat, m: Mat, dsize: Size) void {
    _ = c.WarpPerspective(src.ptr, dst.*.ptr, m.ptr, dsize.toC());
}

pub fn warpPerspectiveWithParams(src: Mat, dst: *Mat, rot_mat: Mat, dsize: Size, flags: i32, border_mode: BorderType, border_value: Color) void {
    _ = c.WarpPerspectiveWithParams(src.ptr, dst.*.ptr, rot_mat.ptr, dsize.toC(), flags, @enumToInt(border_mode), border_value.toScalar().toC());
}

pub fn watershed(image: Mat, markers: *Mat) void {
    _ = c.Watershed(image.ptr, markers.*.ptr);
}

// ApplyColorMap applies a GNU Octave/MATLAB equivalent colormap on a given image.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html#gadf478a5e5ff49d8aa24e726ea6f65d15
pub fn applyColorMap(src: Mat, dst: *Mat, colormap: ColormapType) void {
    _ = c.ApplyColorMap(src.ptr, dst.*.ptr, @enumToInt(colormap));
}

// ApplyCustomColorMap applies a custom defined colormap on a given image.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html#gacb22288ddccc55f9bd9e6d492b409cae
pub fn applyCustomColorMap(src: Mat, dst: *Mat, colormap: Mat) void {
    _ = c.ApplyCustomColorMap(src.ptr, dst.*.ptr, colormap.ptr);
}

// GetPerspectiveTransform returns 3x3 perspective transformation for the
// corresponding 4 point pairs as image.Point.
//
// For further details, please see:
// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga8c1ae0e3589a9d77fffc962c49b22043
pub fn getPerspectiveTransform(src: PointVector, dst: PointVector) !Mat {
    return try Mat.initFromC(c.GetPerspectiveTransform(src.toC(), dst.toC()));
}

// GetPerspectiveTransform2f returns 3x3 perspective transformation for the
// corresponding 4 point pairs as gocv.Point2f.
//
// For further details, please see:
// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga8c1ae0e3589a9d77fffc962c49b22043
pub fn getPerspectiveTransform2f(src: Point2fVector, dst: Point2fVector) !Mat {
    return try Mat.initFromC(c.GetPerspectiveTransform2f(src.toC(), dst.toC()));
}

// GetAffineTransform returns a 2x3 affine transformation matrix for the
// corresponding 3 point pairs as image.Point.
//
// For further details, please see:
// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga8f6d378f9f8eebb5cb55cd3ae295a999
pub fn getAffineTransform(src: PointVector, dst: PointVector) !Mat {
    return try Mat.initFromC(c.GetAffineTransform(src.toC(), dst.toC()));
}

// GetAffineTransform2f returns a 2x3 affine transformation matrix for the
// corresponding 3 point pairs as gocv.Point2f.
//
// For further details, please see:
// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga8f6d378f9f8eebb5cb55cd3ae295a999
pub fn getAffineTransform2f(src: Point2fVector, dst: Point2fVector) !Mat {
    return try Mat.initFromC(c.GetAffineTransform2f(src.toC(), dst.toC()));
}

// FindHomography finds an optimal homography matrix using 4 or more point pairs (as opposed to GetPerspectiveTransform, which uses exactly 4)
//
// For further details, please see:
// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
//
pub fn findHomography(src: Mat, dst: *Mat, method: HomographyMethod, ransac_reproj_threshold: f64, mask: *Mat, max_iters: i32, confidence: f64) !Mat {
    return try Mat.initFromC(c.FindHomography(src.ptr, dst.*.ptr, @enumToInt(method), ransac_reproj_threshold, mask.*.ptr, max_iters, confidence));
}

// pub fn drawContours(src: Mat, contours: PointsVector, contour_idx: c_int, color: Color, thickness: c_int) void;
// pub fn drawContoursWithParams(src: Mat, contours: PointsVector, contourIdx: c_int, color: Scalar, thickness: c_int, lineType: c_int, hierarchy: Mat, maxLevel: c_int, offset: Point) void;

// Sobel calculates the first, second, third, or mixed image derivatives using an extended Sobel operator
//
// For further details, please see:
// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
//
pub fn sobel(src: Mat, dst: *Mat, ddepth: MatType, dx: i32, dy: i32, ksize: i32, scale: f64, delta: f64, border_type: BorderType) void {
    _ = c.Sobel(src.ptr, dst.*.ptr, @enumToInt(ddepth), dx, dy, ksize, scale, delta, @enumToInt(border_type));
}

// SpatialGradient calculates the first order image derivative in both x and y using a Sobel operator.
//
// For further details, please see:
// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga405d03b20c782b65a4daf54d233239a2
//
pub fn spatialGradient(src: Mat, dx: *Mat, dy: *Mat, ksize: i32, border_type: BorderType) void {
    _ = c.SpatialGradient(src.ptr, dx.*.ptr, dy.*.ptr, ksize, @enumToInt(border_type));
}

// Remap applies a generic geometrical transformation to an image.
//
// For further details, please see:
// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
pub fn remap(src: Mat, dst: *Mat, map1: Mat, map2: Mat, interpolation: InterpolationFlag, border_mode: BorderType, border_value: Color) void {
    _ = c.Remap(src.ptr, dst.*.ptr, map1.ptr, map2.ptr, @enumToInt(interpolation), @enumToInt(border_mode), border_value.toScalar().toC());
}

// Filter2D applies an arbitrary linear filter to an image.
//
// For further details, please see:
// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
pub fn filter2D(src: Mat, dst: *Mat, ddepth: i32, kernel: Mat, anchor: Point, delta: f64, border_type: BorderType) void {
    _ = c.Filter2D(src.ptr, dst.*.ptr, ddepth, kernel.ptr, anchor.toC(), delta, @enumToInt(border_type));
}

// SepFilter2D applies a separable linear filter to the image.
//
// For further details, please see:
// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga910e29ff7d7b105057d1625a4bf6318d
pub fn sepFilter2D(src: Mat, dst: *Mat, ddepth: i32, kernel_x: Mat, kernel_y: Mat, anchor: Point, delta: f64, border_type: BorderType) void {
    _ = c.SepFilter2D(src.ptr, dst.*.ptr, ddepth, kernel_x.ptr, kernel_y.ptr, anchor.toC(), delta, @enumToInt(border_type));
}

// LogPolar remaps an image to semilog-polar coordinates space.
//
// For further details, please see:
// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaec3a0b126a85b5ca2c667b16e0ae022d
pub fn logPolar(src: Mat, dst: *Mat, center: Point, m: f64, flags: InterpolationFlag) void {
    _ = c.LogPolar(src.ptr, dst.*.ptr, center.toC(), m, @enumToInt(flags));
}

// FitLine fits a line to a 2D or 3D point set.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaf849da1fdafa67ee84b1e9a23b93f91f
pub fn fitLine(pts: PointVector, line_mat: *Mat, dist_type: DistanceType, param: f64, reps: f64, aeps: f64) void {
    _ = c.FitLine(pts.toC(), line_mat.*.ptr, @enumToInt(dist_type), param, reps, aeps);
}

// LinearPolar remaps an image to polar coordinates space.
//
// For further details, please see:
// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaa38a6884ac8b6e0b9bed47939b5362f3
//
pub fn linearPolar(src: Mat, dst: *Mat, center: Point, max_radius: f64, flags: InterpolationFlag) void {
    _ = c.LinearPolar(src.ptr, dst.*.ptr, center.toC(), max_radius, @enumToInt(flags));
}

// ClipLine clips the line against the image rectangle.
// For further details, please see:
// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gaf483cb46ad6b049bc35ec67052ef1c2c
//
pub fn clipLine(imgSize: Size, pt1: Point, pt2: Point) bool {
    return c.ClipLine(imgSize.toC(), pt1.toC(), pt2.toC());
}

pub fn InvertAffineTransform(src: Mat, dst: *Mat) void {
    _ = c.InvertAffineTransform(src.ptr, dst.*.ptr);
}

// Apply phaseCorrelate.
//
// For further details, please see:
// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga552420a2ace9ef3fb053cd630fdb4952
//
pub fn PhaseCorrelate(src1: Mat, src2: Mat, window: Mat) struct { point: Point2f, response: f64 } {
    var response: f64 = undefined;
    const p = c.PhaseCorrelate(src1.ptr, src2.ptr, window.ptr, &response);
    return .{ .point = Point2f.fromC(p), .response = response };
}

// Adds the square of a source image to the accumulator image.
//
// For further details, please see:
// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga1a567a79901513811ff3b9976923b199
//
pub fn accumulate(src: Mat, dst: *Mat) void {
    _ = c.Accumulate(src.ptr, dst.*.ptr);
}

// Adds an image to the accumulator image with mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga1a567a79901513811ff3b9976923b199
//
pub fn accumulateWithMask(src: Mat, dst: *Mat, mask: Mat) void {
    _ = c.AccumulateWithMask(src.ptr, dst.*.ptr, mask.ptr);
}

// Adds the square of a source image to the accumulator image.
//
// For further details, please see:
// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#gacb75e7ffb573227088cef9ceaf80be8c
//
pub fn accumulateSquare(src: Mat, dst: *Mat) void {
    _ = c.AccumulateSquare(src.ptr, dst.*.ptr);
}

// Adds the square of a source image to the accumulator image with mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#gacb75e7ffb573227088cef9ceaf80be8c
//
pub fn accumulateSquareWithMask(src: Mat, dst: *Mat, mask: Mat) void {
    _ = c.AccumulateSquareWithMask(src.ptr, dst.*.ptr, mask.ptr);
}

// Adds the per-element product of two input images to the accumulator image.
//
// For further details, please see:
// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga82518a940ecfda49460f66117ac82520
//
pub fn accumulateProduct(src1: Mat, src2: Mat, dst: *Mat) void {
    _ = c.AccumulateProduct(src1.ptr, src2.ptr, dst.*.ptr);
}

// Adds the per-element product of two input images to the accumulator image with mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga82518a940ecfda49460f66117ac82520
//
pub fn accumulateProductWithMask(src1: Mat, src2: Mat, dst: *Mat, mask: Mat) void {
    _ = c.AccumulateProductWithMask(src1.ptr, src2.ptr, dst.*.ptr, mask.ptr);
}

// Updates a running average.
//
// For further details, please see:
// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga4f9552b541187f61f6818e8d2d826bc7
//
pub fn accumulatedWeighted(src: Mat, dst: *Mat, alpha: f64) void {
    _ = c.AccumulatedWeighted(src.ptr, dst.*.ptr, alpha);
}

// Updates a running average with mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga4f9552b541187f61f6818e8d2d826bc7
//
pub fn accumulatedWeightedWithMask(src: Mat, dst: *Mat, alpha: f64, mask: Mat) void {
    _ = c.AccumulatedWeightedWithMask(src.ptr, dst.*.ptr, alpha, mask.ptr);
}

test "imgproc" {
    _ = @import("imgproc/test.zig");
}

//*    implementation done
//*    pub const CLAHE = ?*anyopaque;
//*    pub extern fn ArcLength(curve: PointVector, is_closed: bool) f64;
//*    pub extern fn ApproxPolyDP(curve: PointVector, epsilon: f64, closed: bool) PointVector;
//*    pub extern fn CvtColor(src: Mat, dst: Mat, code: c_int) void;
//*    pub extern fn EqualizeHist(src: Mat, dst: Mat) void;
//     pub extern fn CalcHist(mats: struct_Mats, chans: IntVector, mask: Mat, hist: Mat, sz: IntVector, rng: FloatVector, acc: bool) void;
//     pub extern fn CalcBackProject(mats: struct_Mats, chans: IntVector, hist: Mat, backProject: Mat, rng: FloatVector, uniform: bool) void;
//*    pub extern fn CompareHist(hist1: Mat, hist2: Mat, method: c_int) f64;
//*    pub extern fn ConvexHull(points: PointVector, hull: Mat, clockwise: bool, returnPoints: bool) void;
//*    pub extern fn ConvexityDefects(points: PointVector, hull: Mat, result: Mat) void;
//*    pub extern fn BilateralFilter(src: Mat, dst: Mat, d: c_int, sc: f64, ss: f64) void;
//*    pub extern fn Blur(src: Mat, dst: Mat, ps: Size) void;
//*    pub extern fn BoxFilter(src: Mat, dst: Mat, ddepth: c_int, ps: Size) void;
//*    pub extern fn SqBoxFilter(src: Mat, dst: Mat, ddepth: c_int, ps: Size) void;
//*    pub extern fn Dilate(src: Mat, dst: Mat, kernel: Mat) void;
//*    pub extern fn DilateWithParams(src: Mat, dst: Mat, kernel: Mat, anchor: Point, iterations: c_int, borderType: c_int, borderValue: Scalar) void;
//*    pub extern fn DistanceTransform(src: Mat, dst: Mat, labels: Mat, distanceType: c_int, maskSize: c_int, labelType: c_int) void;
//*    pub extern fn Erode(src: Mat, dst: Mat, kernel: Mat) void;
//*    pub extern fn ErodeWithParams(src: Mat, dst: Mat, kernel: Mat, anchor: Point, iterations: c_int, borderType: c_int) void;
//*    pub extern fn MatchTemplate(image: Mat, templ: Mat, result: Mat, method: c_int, mask: Mat) void;
//     pub extern fn Moments(src: Mat, binaryImage: bool) struct_Moment;
//*    pub extern fn PyrDown(src: Mat, dst: Mat, dstsize: Size, borderType: c_int) void;
//*    pub extern fn PyrUp(src: Mat, dst: Mat, dstsize: Size, borderType: c_int) void;
//*    pub extern fn BoundingRect(pts: PointVector) struct_Rect;
//*    pub extern fn BoxPoints(rect: RotatedRect, boxPts: Mat) void;
//*    pub extern fn ContourArea(pts: PointVector) f64;
//*    pub extern fn MinAreaRect(pts: PointVector) struct_RotatedRect;
//*    pub extern fn FitEllipse(pts: PointVector) struct_RotatedRect;
//     pub extern fn MinEnclosingCircle(pts: PointVector, center: [*c]Point2f, radius: [*c]f32) void;
//     pub extern fn FindContours(src: Mat, hierarchy: Mat, mode: c_int, method: c_int) PointsVector;
//*    pub extern fn PointPolygonTest(pts: PointVector, pt: Point, measureDist: bool) f64;
//*    pub extern fn ConnectedComponents(src: Mat, dst: Mat, connectivity: c_int, ltype: c_int, ccltype: c_int) c_int;
//*    pub extern fn ConnectedComponentsWithStats(src: Mat, labels: Mat, stats: Mat, centroids: Mat, connectivity: c_int, ltype: c_int, ccltype: c_int) c_int;
//*    pub extern fn GaussianBlur(src: Mat, dst: Mat, ps: Size, sX: f64, sY: f64, bt: c_int) void;
//*    pub extern fn GetGaussianKernel(ksize: c_int, sigma: f64, ktype: c_int) Mat;
//*    pub extern fn Laplacian(src: Mat, dst: Mat, dDepth: c_int, kSize: c_int, scale: f64, delta: f64, borderType: c_int) void;
//*    pub extern fn Scharr(src: Mat, dst: Mat, dDepth: c_int, dx: c_int, dy: c_int, scale: f64, delta: f64, borderType: c_int) void;
//*    pub extern fn GetStructuringElement(shape: c_int, ksize: Size) Mat;
//*    pub extern fn MorphologyDefaultBorderValue(...) Scalar;
//*    pub extern fn MorphologyEx(src: Mat, dst: Mat, op: c_int, kernel: Mat) void;
//*    pub extern fn MorphologyExWithParams(src: Mat, dst: Mat, op: c_int, kernel: Mat, pt: Point, iterations: c_int, borderType: c_int) void;
//*    pub extern fn MedianBlur(src: Mat, dst: Mat, ksize: c_int) void;
//*    pub extern fn Canny(src: Mat, edges: Mat, t1: f64, t2: f64) void;
//*    pub extern fn CornerSubPix(img: Mat, corners: Mat, winSize: Size, zeroZone: Size, criteria: TermCriteria) void;
//*    pub extern fn GoodFeaturesToTrack(img: Mat, corners: Mat, maxCorners: c_int, quality: f64, minDist: f64) void;
//*    pub extern fn GrabCut(img: Mat, mask: Mat, rect: Rect, bgdModel: Mat, fgdModel: Mat, iterCount: c_int, mode: c_int) void;
//*    pub extern fn HoughCircles(src: Mat, circles: Mat, method: c_int, dp: f64, minDist: f64) void;
//*    pub extern fn HoughCirclesWithParams(src: Mat, circles: Mat, method: c_int, dp: f64, minDist: f64, param1: f64, param2: f64, minRadius: c_int, maxRadius: c_int) void;
//*    pub extern fn HoughLines(src: Mat, lines: Mat, rho: f64, theta: f64, threshold: c_int) void;
//*    pub extern fn HoughLinesP(src: Mat, lines: Mat, rho: f64, theta: f64, threshold: c_int) void;
//*    pub extern fn HoughLinesPWithParams(src: Mat, lines: Mat, rho: f64, theta: f64, threshold: c_int, minLineLength: f64, maxLineGap: f64) void;
//*    pub extern fn HoughLinesPointSet(points: Mat, lines: Mat, lines_max: c_int, threshold: c_int, min_rho: f64, max_rho: f64, rho_step: f64, min_theta: f64, max_theta: f64, theta_step: f64) void;
//*    pub extern fn Integral(src: Mat, sum: Mat, sqsum: Mat, tilted: Mat) void;
//*    pub extern fn Threshold(src: Mat, dst: Mat, thresh: f64, maxvalue: f64, typ: c_int) f64;
//*    pub extern fn AdaptiveThreshold(src: Mat, dst: Mat, maxValue: f64, adaptiveTyp: c_int, typ: c_int, blockSize: c_int, c: f64) void;
//*    pub extern fn ArrowedLine(img: Mat, pt1: Point, pt2: Point, color: Scalar, thickness: c_int) void;
//*    pub extern fn Circle(img: Mat, center: Point, radius: c_int, color: Scalar, thickness: c_int) void;
//*    pub extern fn CircleWithParams(img: Mat, center: Point, radius: c_int, color: Scalar, thickness: c_int, lineType: c_int, shift: c_int) void;
//*    pub extern fn Ellipse(img: Mat, center: Point, axes: Point, angle: f64, startAngle: f64, endAngle: f64, color: Scalar, thickness: c_int) void;
//*    pub extern fn EllipseWithParams(img: Mat, center: Point, axes: Point, angle: f64, startAngle: f64, endAngle: f64, color: Scalar, thickness: c_int, lineType: c_int, shift: c_int) void;
//*    pub extern fn Line(img: Mat, pt1: Point, pt2: Point, color: Scalar, thickness: c_int) void;
//*    pub extern fn Rectangle(img: Mat, rect: Rect, color: Scalar, thickness: c_int) void;
//*    pub extern fn RectangleWithParams(img: Mat, rect: Rect, color: Scalar, thickness: c_int, lineType: c_int, shift: c_int) void;
//     pub extern fn FillPoly(img: Mat, points: PointsVector, color: Scalar) void;
//     pub extern fn FillPolyWithParams(img: Mat, points: PointsVector, color: Scalar, lineType: c_int, shift: c_int, offset: Point) void;
//     pub extern fn Polylines(img: Mat, points: PointsVector, isClosed: bool, color: Scalar, thickness: c_int) void;
//*    pub extern fn GetTextSize(text: [*c]const u8, fontFace: c_int, fontScale: f64, thickness: c_int) struct_Size;
//*    pub extern fn GetTextSizeWithBaseline(text: [*c]const u8, fontFace: c_int, fontScale: f64, thickness: c_int, baseline: [*c]c_int) struct_Size;
//*    pub extern fn PutText(img: Mat, text: [*c]const u8, org: Point, fontFace: c_int, fontScale: f64, color: Scalar, thickness: c_int) void;
//*    pub extern fn PutTextWithParams(img: Mat, text: [*c]const u8, org: Point, fontFace: c_int, fontScale: f64, color: Scalar, thickness: c_int, lineType: c_int, bottomLeftOrigin: bool) void;
//*    pub extern fn Resize(src: Mat, dst: Mat, sz: Size, fx: f64, fy: f64, interp: c_int) void;
//*    pub extern fn GetRectSubPix(src: Mat, patchSize: Size, center: Point, dst: Mat) void;
//*    pub extern fn GetRotationMatrix2D(center: Point, angle: f64, scale: f64) Mat;
//*    pub extern fn WarpAffine(src: Mat, dst: Mat, rot_mat: Mat, dsize: Size) void;
//*    pub extern fn WarpAffineWithParams(src: Mat, dst: Mat, rot_mat: Mat, dsize: Size, flags: c_int, borderMode: c_int, borderValue: Scalar) void;
//*    pub extern fn WarpPerspective(src: Mat, dst: Mat, m: Mat, dsize: Size) void;
//*    pub extern fn WarpPerspectiveWithParams(src: Mat, dst: Mat, rot_mat: Mat, dsize: Size, flags: c_int, borderMode: c_int, borderValue: Scalar) void;
//*    pub extern fn Watershed(image: Mat, markers: Mat) void;
//*    pub extern fn ApplyColorMap(src: Mat, dst: Mat, colormap: c_int) void;
//*    pub extern fn ApplyCustomColorMap(src: Mat, dst: Mat, colormap: Mat) void;
//*    pub extern fn GetPerspectiveTransform(src: PointVector, dst: PointVector) Mat;
//*    pub extern fn GetPerspectiveTransform2f(src: Point2fVector, dst: Point2fVector) Mat;
//*    pub extern fn GetAffineTransform(src: PointVector, dst: PointVector) Mat;
//*    pub extern fn GetAffineTransform2f(src: Point2fVector, dst: Point2fVector) Mat;
//*    pub extern fn FindHomography(src: Mat, dst: Mat, method: c_int, ransacReprojThreshold: f64, mask: Mat, maxIters: c_int, confidence: f64) Mat;
//*    pub extern fn DrawContours(src: Mat, contours: PointsVector, contourIdx: c_int, color: Scalar, thickness: c_int) void;
//*    pub extern fn DrawContoursWithParams(src: Mat, contours: PointsVector, contourIdx: c_int, color: Scalar, thickness: c_int, lineType: c_int, hierarchy: Mat, maxLevel: c_int, offset: Point) void;
//*    pub extern fn Sobel(src: Mat, dst: Mat, ddepth: c_int, dx: c_int, dy: c_int, ksize: c_int, scale: f64, delta: f64, borderType: c_int) void;
//*    pub extern fn SpatialGradient(src: Mat, dx: Mat, dy: Mat, ksize: c_int, borderType: c_int) void;
//*    pub extern fn Remap(src: Mat, dst: Mat, map1: Mat, map2: Mat, interpolation: c_int, borderMode: c_int, borderValue: Scalar) void;
//*    pub extern fn Filter2D(src: Mat, dst: Mat, ddepth: c_int, kernel: Mat, anchor: Point, delta: f64, borderType: c_int) void;
//*    pub extern fn SepFilter2D(src: Mat, dst: Mat, ddepth: c_int, kernelX: Mat, kernelY: Mat, anchor: Point, delta: f64, borderType: c_int) void;
//*    pub extern fn LogPolar(src: Mat, dst: Mat, center: Point, m: f64, flags: c_int) void;
//*    pub extern fn FitLine(pts: PointVector, line: Mat, distType: c_int, param: f64, reps: f64, aeps: f64) void;
//*    pub extern fn LinearPolar(src: Mat, dst: Mat, center: Point, maxRadius: f64, flags: c_int) void;
//*    pub extern fn ClipLine(imgSize: Size, pt1: Point, pt2: Point) bool;
//*    pub extern fn CLAHE_Create(...) CLAHE;
//*    pub extern fn CLAHE_CreateWithParams(clipLimit: f64, tileGridSize: Size) CLAHE;
//*    pub extern fn CLAHE_Close(c: CLAHE) void;
//*    pub extern fn CLAHE_Apply(c: CLAHE, src: Mat, dst: Mat) void;
//*    pub extern fn InvertAffineTransform(src: Mat, dst: Mat) void;
//*    pub extern fn PhaseCorrelate(src1: Mat, src2: Mat, window: Mat, response: [*c]f64) Point2f;
//*    pub extern fn Mat_Accumulate(src: Mat, dst: Mat) void;
//*    pub extern fn Mat_AccumulateWithMask(src: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_AccumulateSquare(src: Mat, dst: Mat) void;
//*    pub extern fn Mat_AccumulateSquareWithMask(src: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_AccumulateProduct(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_AccumulateProductWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_AccumulatedWeighted(src: Mat, dst: Mat, alpha: f64) void;
//*    pub extern fn Mat_AccumulatedWeightedWithMask(src: Mat, dst: Mat, alpha: f64, mask: Mat) void;
