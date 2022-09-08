const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const Mat = core.Mat;
const Size = core.Size;

// Calib is a wrapper around OpenCV's "Camera Calibration and 3D Reconstruction" of
// Fisheye Camera model
//
// For more details, please see:
// https://docs.opencv.org/trunk/db/d58/group__calib3d__fisheye.html

// CalibFlag value for calibration
pub const CalibFlag = enum(u11) {
    // CalibUseIntrinsicGuess indicates that cameraMatrix contains valid initial values
    // of fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially
    // set to the image center ( imageSize is used), and focal distances are computed
    // in a least-squares fashion.
    UseIntrinsicGuess = 1 << 0,

    // CalibRecomputeExtrinsic indicates that extrinsic will be recomputed after each
    // iteration of intrinsic optimization.
    RecomputeExtrinsic = 1 << 1,

    // CalibCheckCond indicates that the functions will check validity of condition number
    CheckCond = 1 << 2,

    // CalibFixSkew indicates that skew coefficient (alpha) is set to zero and stay zero
    FixSkew = 1 << 3,

    // CalibFixK1 indicates that selected distortion coefficients are set to zeros and stay zero
    FixK1 = 1 << 4,

    // CalibFixK2 indicates that selected distortion coefficients are set to zeros and stay zero
    FixK2 = 1 << 5,

    // CalibFixK3 indicates that selected distortion coefficients are set to zeros and stay zero
    FixK3 = 1 << 6,

    // CalibFixK4 indicates that selected distortion coefficients are set to zeros and stay zero
    FixK4 = 1 << 7,

    // CalibFixIntrinsic indicates that fix K1, K2? and D1, D2? so that only R, T matrices are estimated
    FixIntrinsic = 1 << 8,

    // CalibFixPrincipalPoint indicates that the principal point is not changed during the global optimization.
    // It stays at the center or at a different location specified when CalibUseIntrinsicGuess is set too.
    FixPrincipalPoint = 1 << 9,

    ZeroDisparity = 1 << 10,
};

pub const Fisheye = struct {
    pub fn undistortImage(image: Mat, undistorted: *Mat, k: Mat, d: Mat) void {
        _ = c.Fisheye_UndistortImage(
            image.ptr,
            undistorted.*.ptr,
            k.ptr,
            d.ptr,
        );
    }

    pub fn undistortImageWithParams(distorted: Mat, undistorted: *Mat, k: Mat, d: Mat, k_new: Mat, size: Size) void {
        _ = c.Fisheye_UndistortImageWithParams(distorted.ptr, undistorted.*.ptr, k.ptr, d.ptr, k_new.ptr, size.toC());
    }

    pub fn undistortPoints(distorted: Mat, undistorted: *Mat, k: Mat, d: Mat, r: Mat, p: Mat) void {
        _ = c.Fisheye_UndistortPoints(distorted.ptr, undistorted.*.ptr, k.ptr, d.ptr, r.ptr, p.ptr);
    }
    pub fn estimateNewCameraMatrixForUndistortRectify(
        k: Mat,
        d: Mat,
        img_size: Size,
        r: Mat,
        p: *Mat,
        balance: f64,
        new_size: Size,
        fov_scale: f64,
    ) Mat {
        var mat = Mat.init();
        _ = c.Fisheye_EstimateNewCameraMatrixForUndistortRectify(
            k.ptr,
            d.ptr,
            img_size.toC(),
            r.ptr,
            p.*.ptr,
            balance,
            new_size.toC(),
            fov_scale,
        );
        return mat;
    }
};

// InitUndistortRectifyMap computes the joint undistortion and rectification transformation and represents the result in the form of maps for remap
//
// For further details, please see:
// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
//
pub fn initUndistortRectifyMap(
    camera_matrix: Mat,
    dist_coeffs: Mat,
    r: Mat,
    new_camera_matrix: Mat,
    size: Size,
    m1type: i32,
    map1: Mat,
    map2: Mat,
) void {
    _ = c.InitUndistortRectifyMap(
        camera_matrix.ptr,
        dist_coeffs.ptr,
        r.ptr,
        new_camera_matrix.ptr,
        size.toC(),
        m1type,
        map1.ptr,
        map2.ptr,
    );
}

//*    implementation done
//*    pub extern fn Fisheye_UndistortImage(distorted: Mat, undistorted: Mat, k: Mat, d: Mat) void;
//*    pub extern fn Fisheye_UndistortImageWithParams(distorted: Mat, undistorted: Mat, k: Mat, d: Mat, knew: Mat, size: Size) void;
//*    pub extern fn Fisheye_UndistortPoints(distorted: Mat, undistorted: Mat, k: Mat, d: Mat, R: Mat, P: Mat) void;
//*    pub extern fn Fisheye_EstimateNewCameraMatrixForUndistortRectify(k: Mat, d: Mat, imgSize: Size, r: Mat, p: Mat, balance: f64, newSize: Size, fovScale: f64) void;
//*    pub extern fn InitUndistortRectifyMap(cameraMatrix: Mat, distCoeffs: Mat, r: Mat, newCameraMatrix: Mat, size: Size, m1type: c_int, map1: Mat, map2: Mat) void;
//     pub extern fn GetOptimalNewCameraMatrixWithParams(cameraMatrix: Mat, distCoeffs: Mat, size: Size, alpha: f64, newImgSize: Size, validPixROI: [*c]Rect, centerPrincipalPoint: bool) Mat;
//     pub extern fn CalibrateCamera(objectPoints: Points3fVector, imagePoints: Points2fVector, imageSize: Size, cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat, flag: c_int) f64;
//     pub extern fn Undistort(src: Mat, dst: Mat, cameraMatrix: Mat, distCoeffs: Mat, newCameraMatrix: Mat) void;
//     pub extern fn UndistortPoints(distorted: Mat, undistorted: Mat, k: Mat, d: Mat, r: Mat, p: Mat) void;
//     pub extern fn FindChessboardCorners(image: Mat, patternSize: Size, corners: Mat, flags: c_int) bool;
//     pub extern fn FindChessboardCornersSB(image: Mat, patternSize: Size, corners: Mat, flags: c_int) bool;
//     pub extern fn FindChessboardCornersSBWithMeta(image: Mat, patternSize: Size, corners: Mat, flags: c_int, meta: Mat) bool;
//     pub extern fn DrawChessboardCorners(image: Mat, patternSize: Size, corners: Mat, patternWasFound: bool) void;
//     pub extern fn EstimateAffinePartial2D(from: Point2fVector, to: Point2fVector) Mat;
//     pub extern fn EstimateAffine2D(from: Point2fVector, to: Point2fVector) Mat;
//     pub extern fn EstimateAffine2DWithParams(from: Point2fVector, to: Point2fVector, inliers: Mat, method: c_int, ransacReprojThreshold: f64, maxIters: usize, confidence: f64, refineIters: usize) Mat;
