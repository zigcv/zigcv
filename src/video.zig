const c = @import("c_api.zig");
const core = @import("core.zig");
const Mat = core.Mat;
const Rect = core.Rect;
const Size = core.Size;
const TermCriteria = core.TermCriteria;
const Tracker = c.Tracker;

// cv::OPTFLOW_USE_INITIAL_FLOW = 4,
// cv::OPTFLOW_LK_GET_MIN_EIGENVALS = 8,
// cv::OPTFLOW_FARNEBACK_GAUSSIAN = 256
// For further details, please see: https://docs.opencv.org/master/dc/d6b/group__video__track.html#gga2c6cc144c9eee043575d5b311ac8af08a9d4430ac75199af0cf6fcdefba30eafe
pub const OptflowUseInitialFlow = 4;
pub const OptflowLkGetMinEigenvals = 8;
pub const OptflowFarnebackGaussian = 256;

// cv::MOTION_TRANSLATION = 0,
// cv::MOTION_EUCLIDEAN = 1,
// cv::MOTION_AFFINE = 2,
// cv::MOTION_HOMOGRAPHY = 3
// For further details, please see: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ggaaedb1f94e6b143cef163622c531afd88a01106d6d20122b782ff25eaeffe9a5be
pub const MotionTranslation = 0;
pub const MotionEuclidean = 1;
pub const MotionAffine = 2;
pub const MotionHomography = 3;

// BackgroundSubtractorMOG2 is a wrapper around the cv::BackgroundSubtractorMOG2.
const BackgroundSubtractorMOG2 = struct {
    ptr: c.BackgroundSubtractorMOG2,

    const Self = @This();

    // NewBackgroundSubtractorMOG2 returns a new BackgroundSubtractor algorithm
    // of type MOG2. MOG2 is a Gaussian Mixture-based Background/Foreground
    // Segmentation Algorithm.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/de/de1/group__video__motion.html#ga2beb2dee7a073809ccec60f145b6b29c
    // https://docs.opencv.org/master/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
    //
    pub fn init() Self {
        return Self{
            .ptr = c.BackgroundSubtractorMOG2_Create(),
        };
    }

    // NewBackgroundSubtractorMOG2WithParams returns a new BackgroundSubtractor algorithm
    // of type MOG2 with customized parameters. MOG2 is a Gaussian Mixture-based Background/Foreground
    // Segmentation Algorithm.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/de/de1/group__video__motion.html#ga2beb2dee7a073809ccec60f145b6b29c
    // https://docs.opencv.org/master/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
    //
    pub fn initWithPatams(history: c_int, varThreshold: f64, detectShadows: bool) BackgroundSubtractorMOG2 {
        return Self{
            .ptr = c.BackgroundSubtractorMOG2_CreateWithParams(history, varThreshold, detectShadows),
        };
    }

    // Close BackgroundSubtractorMOG2.
    pub fn deinit(self: *Self) void {
        _ = c.BackgroundSubtractorMOG2_Close(self.ptr);
    }

    // Apply computes a foreground mask using the current BackgroundSubtractorMOG2.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/df6/classcv_1_1BackgroundSubtractor.html#aa735e76f7069b3fa9c3f32395f9ccd21
    //
    pub fn Apply(self: *Self, src: Mat, dst: *Mat) void {
        _ = c.BackgroundSubtractorMOG2_Apply(self.ptr, src.ptr, dst.*.ptr);
    }
};

// BackgroundSubtractorKNN is a wrapper around the cv::BackgroundSubtractorKNN.
pub const BackgroundSubtractorKNN = struct {
    ptr: c.BackgroundSubtractorKNN,

    const Self = @This();

    // NewBackgroundSubtractorKNN returns a new BackgroundSubtractor algorithm
    // of type KNN. K-Nearest Neighbors (KNN) uses a Background/Foreground
    // Segmentation Algorithm
    //
    // For further details, please see:
    // https://docs.opencv.org/master/de/de1/group__video__motion.html#gac9be925771f805b6fdb614ec2292006d
    // https://docs.opencv.org/master/db/d88/classcv_1_1BackgroundSubtractorKNN.html
    //
    pub fn init() Self {
        return BackgroundSubtractorKNN{
            .ptr = c.BackgroundSubtractorKNN_Create(),
        };
    }

    // NewBackgroundSubtractorKNNWithParams returns a new BackgroundSubtractor algorithm
    // of type KNN with customized parameters. K-Nearest Neighbors (KNN) uses a Background/Foreground
    // Segmentation Algorithm
    //
    // For further details, please see:
    // https://docs.opencv.org/master/de/de1/group__video__motion.html#gac9be925771f805b6fdb614ec2292006d
    // https://docs.opencv.org/master/db/d88/classcv_1_1BackgroundSubtractorKNN.html
    //
    pub fn initWithPatams(history: c_int, dist2Threshold: f64, detectShadows: bool) Self {
        return Self{
            .p = c.BackgroundSubtractorKNN_CreateWithParams(history, dist2Threshold, detectShadows),
        };
    }

    // Close BackgroundSubtractorKNN.
    pub fn Close(self: *Self) void {
        _ = c.BackgroundSubtractorKNN_Close(self.ptr);
    }

    // Apply computes a foreground mask using the current BackgroundSubtractorKNN.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/df6/classcv_1_1BackgroundSubtractor.html#aa735e76f7069b3fa9c3f32395f9ccd21
    //
    pub fn Apply(self: *Self, src: Mat, dst: *Mat) void {
        _ = c.BackgroundSubtractorKNN_Apply(self.ptr, src.ptr, dst.*.ptr);
    }
};

// CalcOpticalFlowFarneback computes a dense optical flow using
// Gunnar Farneback's algorithm.
//
// For further details, please see:
// https://docs.opencv.org/master/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
//
pub fn CalcOpticalFlowFarneback(
    prev_img: Mat,
    next_img: Mat,
    flow: *Mat,
    pyr_scale: f64,
    levels: c_int,
    winsize: c_int,
    iterations: c_int,
    poly_n: c_int,
    poly_sigma: f64,
    flags: c_int,
) void {
    _ = c.CalcOpticalFlowFarneback(
        prev_img.ptr,
        next_img.ptr,
        flow.*.ptr,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    );
}

// CalcOpticalFlowPyrLK calculates an optical flow for a sparse feature set using
// the iterative Lucas-Kanade method with pyramids.
//
// For further details, please see:
// https://docs.opencv.org/master/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
//
pub fn calcOpticalFlowPyrLK(
    prev_img: Mat,
    next_img: Mat,
    prev_pts: Mat,
    next_pts: Mat,
    status: *Mat,
    err: *Mat,
) void {
    _ = c.CalcOpticalFlowPyrLK(
        prev_img.ptr,
        next_img.ptr,
        prev_pts.ptr,
        next_pts.ptr,
        status.*.ptr,
        err.*.ptr,
    );
}

// CalcOpticalFlowPyrLKWithParams calculates an optical flow for a sparse feature set using
// the iterative Lucas-Kanade method with pyramids.
//
// For further details, please see:
// https://docs.opencv.org/master/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
//
pub fn calcOpticalFlowPyrLKWithParams(
    prev_img: Mat,
    next_img: Mat,
    prev_pts: Mat,
    next_pts: Mat,
    status: *Mat,
    err: *Mat,
    win_size: Size,
    max_level: c_int,
    criteria: TermCriteria,
    flags: c_int,
    min_eig_threshold: f64,
) void {
    _ = c.CalcOpticalFlowPyrLKWithParams(
        prev_img.ptr,
        next_img.ptr,
        prev_pts.ptr,
        next_pts.ptr,
        status.*.ptr,
        err.*.ptr,
        win_size.toC(),
        max_level,
        criteria.ptr,
        flags,
        min_eig_threshold,
    );
}

// FindTransformECC finds the geometric transform (warp) between two images in terms of the ECC criterion.
//
// For futther details, please see:
// https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga1aa357007eaec11e9ed03500ecbcbe47
//
pub fn findTransformECC(
    template_image: Mat,
    input_image: Mat,
    warp_matrix: *Mat,
    motion_type: c_int,
    criteria: TermCriteria,
    input_mask: Mat,
    gauss_filt_size: c_int,
) f64 {
    return @as(
        f64,
        c.FindTransformECC(
            template_image.ptr,
            input_image.ptr,
            warp_matrix.*.ptr,
            motion_type,
            criteria.ptr,
            input_mask.ptr,
            gauss_filt_size,
        ),
    );
}

//*    implementation done
//*    pub const BackgroundSubtractorMOG2 = ?*anyopaque;
//*    pub const BackgroundSubtractorKNN = ?*anyopaque;
//*    pub const Tracker = ?*anyopaque;
//*    pub const TrackerMIL = ?*anyopaque;
//*    pub const TrackerGOTURN = ?*anyopaque;
//*    pub extern fn BackgroundSubtractorMOG2_Create(...) BackgroundSubtractorMOG2;
//*    pub extern fn BackgroundSubtractorMOG2_CreateWithParams(history: c_int, varThreshold: f64, detectShadows: bool) BackgroundSubtractorMOG2;
//*    pub extern fn BackgroundSubtractorMOG2_Close(b: BackgroundSubtractorMOG2) void;
//*    pub extern fn BackgroundSubtractorMOG2_Apply(b: BackgroundSubtractorMOG2, src: Mat, dst: Mat) void;
//*    pub extern fn BackgroundSubtractorKNN_Create(...) BackgroundSubtractorKNN;
//*    pub extern fn BackgroundSubtractorKNN_CreateWithParams(history: c_int, dist2Threshold: f64, detectShadows: bool) BackgroundSubtractorKNN;
//*    pub extern fn BackgroundSubtractorKNN_Close(b: BackgroundSubtractorKNN) void;
//*    pub extern fn BackgroundSubtractorKNN_Apply(b: BackgroundSubtractorKNN, src: Mat, dst: Mat) void;
//*    pub extern fn CalcOpticalFlowPyrLK(prevImg: Mat, nextImg: Mat, prevPts: Mat, nextPts: Mat, status: Mat, err: Mat) void;
//*    pub extern fn CalcOpticalFlowPyrLKWithParams(prevImg: Mat, nextImg: Mat, prevPts: Mat, nextPts: Mat, status: Mat, err: Mat, winSize: Size, maxLevel: c_int, criteria: TermCriteria, flags: c_int, minEigThreshold: f64) void;
//*    pub extern fn CalcOpticalFlowFarneback(prevImg: Mat, nextImg: Mat, flow: Mat, pyrScale: f64, levels: c_int, winsize: c_int, iterations: c_int, polyN: c_int, polySigma: f64, flags: c_int) void;
//*    pub extern fn FindTransformECC(templateImage: Mat, inputImage: Mat, warpMatrix: Mat, motionType: c_int, criteria: TermCriteria, inputMask: Mat, gaussFiltSize: c_int) f64;
//     pub extern fn Tracker_Init(self: Tracker, image: Mat, boundingBox: Rect) bool;
//     pub extern fn Tracker_Update(self: Tracker, image: Mat, boundingBox: [*c]Rect) bool;
//     pub extern fn TrackerMIL_Create(...) TrackerMIL;
//     pub extern fn TrackerMIL_Close(self: TrackerMIL) void;
