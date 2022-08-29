const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const Keypoin = core.KeyPoint;

pub const SIFT = struct {
    ptr: c.SIFT,

    const Self = @This();

    // NewSIFT returns a new SIFT algorithm.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
    //
    pub fn init() Self {
        return .{ .ptr = c.SIFT_Create() };
    }

    pub fn initFromC(ptr: c.SIFT) Self {
        return .{ .ptr = ptr };
    }

    pub fn toC(self: Self) c.SIFT {
        return self.ptr;
    }

    pub fn deinit(self: *Self) void {
        _ = c.SIFT_Close(self.ptr);
    }

    pub fn detect(self: Self, src: Mat) KeyPoints {
        return KeyPoints.initFromC(c.SIFT_Detect(self.ptr, src.toC()));
    }

    pub fn detectAndCompute(self: Self, src: Mat, mask: Mat, desc: *Mat, allocator) !KeyPoints {
        const return_keypoints: c.KeyPoints = c.SIFT_DetectAndCompute(self.ptr, src.ptr, mask.ptr, desc.*.ptr);
        defer c.KeyPoints_Close(return_keypoints);
        return Keypoint.arrayFromC(return_keypoints, allocator);
    }
};

//*    implementation done
//     pub const AKAZE = ?*anyopaque;
//     pub const AgastFeatureDetector = ?*anyopaque;
//     pub const BRISK = ?*anyopaque;
//     pub const FastFeatureDetector = ?*anyopaque;
//     pub const GFTTDetector = ?*anyopaque;
//     pub const KAZE = ?*anyopaque;
//     pub const MSER = ?*anyopaque;
//     pub const ORB = ?*anyopaque;
//     pub const SimpleBlobDetector = ?*anyopaque;
//     pub const BFMatcher = ?*anyopaque;
//     pub const FlannBasedMatcher = ?*anyopaque;
//*    pub const SIFT = ?*anyopaque;
//     pub extern fn AKAZE_Create(...) AKAZE;
//     pub extern fn AKAZE_Close(a: AKAZE) void;
//     pub extern fn AKAZE_Detect(a: AKAZE, src: Mat) struct_KeyPoints;
//     pub extern fn AKAZE_DetectAndCompute(a: AKAZE, src: Mat, mask: Mat, desc: Mat) struct_KeyPoints;
//     pub extern fn AgastFeatureDetector_Create(...) AgastFeatureDetector;
//     pub extern fn AgastFeatureDetector_Close(a: AgastFeatureDetector) void;
//     pub extern fn AgastFeatureDetector_Detect(a: AgastFeatureDetector, src: Mat) struct_KeyPoints;
//     pub extern fn BRISK_Create(...) BRISK;
//     pub extern fn BRISK_Close(b: BRISK) void;
//     pub extern fn BRISK_Detect(b: BRISK, src: Mat) struct_KeyPoints;
//     pub extern fn BRISK_DetectAndCompute(b: BRISK, src: Mat, mask: Mat, desc: Mat) struct_KeyPoints;
//     pub extern fn FastFeatureDetector_Create(...) FastFeatureDetector;
//     pub extern fn FastFeatureDetector_CreateWithParams(threshold: c_int, nonmaxSuppression: bool, @"type": c_int) FastFeatureDetector;
//     pub extern fn FastFeatureDetector_Close(f: FastFeatureDetector) void;
//     pub extern fn FastFeatureDetector_Detect(f: FastFeatureDetector, src: Mat) struct_KeyPoints;
//     pub extern fn GFTTDetector_Create(...) GFTTDetector;
//     pub extern fn GFTTDetector_Close(a: GFTTDetector) void;
//     pub extern fn GFTTDetector_Detect(a: GFTTDetector, src: Mat) struct_KeyPoints;
//     pub extern fn KAZE_Create(...) KAZE;
//     pub extern fn KAZE_Close(a: KAZE) void;
//     pub extern fn KAZE_Detect(a: KAZE, src: Mat) struct_KeyPoints;
//     pub extern fn KAZE_DetectAndCompute(a: KAZE, src: Mat, mask: Mat, desc: Mat) struct_KeyPoints;
//     pub extern fn MSER_Create(...) MSER;
//     pub extern fn MSER_Close(a: MSER) void;
//     pub extern fn MSER_Detect(a: MSER, src: Mat) struct_KeyPoints;
//     pub extern fn ORB_Create(...) ORB;
//     pub extern fn ORB_CreateWithParams(nfeatures: c_int, scaleFactor: f32, nlevels: c_int, edgeThreshold: c_int, firstLevel: c_int, WTA_K: c_int, scoreType: c_int, patchSize: c_int, fastThreshold: c_int) ORB;
//     pub extern fn ORB_Close(o: ORB) void;
//     pub extern fn ORB_Detect(o: ORB, src: Mat) struct_KeyPoints;
//     pub extern fn ORB_DetectAndCompute(o: ORB, src: Mat, mask: Mat, desc: Mat) struct_KeyPoints;
//     pub extern fn SimpleBlobDetector_Create(...) SimpleBlobDetector;
//     pub extern fn SimpleBlobDetector_Create_WithParams(params: SimpleBlobDetectorParams) SimpleBlobDetector;
//     pub extern fn SimpleBlobDetector_Close(b: SimpleBlobDetector) void;
//     pub extern fn SimpleBlobDetector_Detect(b: SimpleBlobDetector, src: Mat) struct_KeyPoints;
//     pub extern fn SimpleBlobDetectorParams_Create(...) SimpleBlobDetectorParams;
//     pub extern fn BFMatcher_Create(...) BFMatcher;
//     pub extern fn BFMatcher_CreateWithParams(normType: c_int, crossCheck: bool) BFMatcher;
//     pub extern fn BFMatcher_Close(b: BFMatcher) void;
//     pub extern fn BFMatcher_KnnMatch(b: BFMatcher, query: Mat, train: Mat, k: c_int) struct_MultiDMatches;
//     pub extern fn FlannBasedMatcher_Create(...) FlannBasedMatcher;
//     pub extern fn FlannBasedMatcher_Close(f: FlannBasedMatcher) void;
//     pub extern fn FlannBasedMatcher_KnnMatch(f: FlannBasedMatcher, query: Mat, train: Mat, k: c_int) struct_MultiDMatches;
//     pub extern fn DrawKeyPoints(src: Mat, kp: struct_KeyPoints, dst: Mat, s: Scalar, flags: c_int) void;
//*    pub extern fn SIFT_Create(...) SIFT;
//*    pub extern fn SIFT_Close(f: SIFT) void;
//*    pub extern fn SIFT_Detect(f: SIFT, src: Mat) struct_KeyPoints;
//*    pub extern fn SIFT_DetectAndCompute(f: SIFT, src: Mat, mask: Mat, desc: Mat) struct_KeyPoints;
//     pub extern fn DrawMatches(img1: Mat, kp1: struct_KeyPoints, img2: Mat, kp2: struct_KeyPoints, matches1to2: struct_DMatches, outImg: Mat, matchesColor: Scalar, pointColor: Scalar, matchesMask: struct_ByteArray, flags: c_int) void;
