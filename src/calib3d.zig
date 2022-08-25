//*    implementation done
//     pub extern fn Fisheye_UndistortImage(distorted: Mat, undistorted: Mat, k: Mat, d: Mat) void;
//     pub extern fn Fisheye_UndistortImageWithParams(distorted: Mat, undistorted: Mat, k: Mat, d: Mat, knew: Mat, size: Size) void;
//     pub extern fn Fisheye_UndistortPoints(distorted: Mat, undistorted: Mat, k: Mat, d: Mat, R: Mat, P: Mat) void;
//     pub extern fn Fisheye_EstimateNewCameraMatrixForUndistortRectify(k: Mat, d: Mat, imgSize: Size, r: Mat, p: Mat, balance: f64, newSize: Size, fovScale: f64) void;
//     pub extern fn InitUndistortRectifyMap(cameraMatrix: Mat, distCoeffs: Mat, r: Mat, newCameraMatrix: Mat, size: Size, m1type: c_int, map1: Mat, map2: Mat) void;
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
