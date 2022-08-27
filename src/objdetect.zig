const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const Mat = core.Mat;
const Rect = core.Rect;
const Size = core.Size;

// CascadeClassifier is a cascade classifier class for object detection.
//
// For further details, please see:
// http://docs.opencv.org/master/d1/de5/classcv_1_1CascadeClassifier.html
//
pub const CascadeClassifier = struct {
    ptr: c.CascadeClassifier,

    const Self = @This();

    pub fn init() Self {
        return .{ .ptr = c.CascadeClassifier_New() };
    }

    pub fn deinit(self: *Self) void {
        _ = c.CascadeClassifier_Close(self.ptr);
    }

    // Load cascade classifier from a file.
    //
    // For further details, please see:
    // http://docs.opencv.org/master/d1/de5/classcv_1_1CascadeClassifier.html#a1a5884c8cc749422f9eb77c2471958bc
    pub fn load(self: *Self, name: []const u8) !void {
        const result = c.CascadeClassifier_Load(self.ptr, utils.castZigU8ToC(name));
        if (result == 0) {
            return error.CascadeClassifierLoadFailed;
        }
    }

    // DetectMultiScale detects objects of different sizes in the input Mat image.
    // The detected objects are returned as a slice of image.Rectangle structs.
    //
    // For further details, please see:
    // http://docs.opencv.org/master/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    //
    pub fn detectMultiScale(self: Self, img: Mat, allocator: std.mem.Allocator) !std.ArrayList(Rect) {
        const rec: c.struct_Rects = c.CascadeClassifier_DetectMultiScale(self.ptr, img.ptr);

        var return_rects = std.ArrayList(Rect).init(allocator);
        {
            var i: usize = 0;
            while (i < rec.length) : (i += 1) {
                try return_rects.append(Rect.initFromC(rec.rects[i]));
            }
        }
        return return_rects;
    }

    // DetectMultiScaleWithParams calls DetectMultiScale but allows setting parameters
    // to values other than just the defaults.
    //
    // For further details, please see:
    // http://docs.opencv.org/master/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    //
    pub fn detectMultiScaleWithParams(
        self: Self,
        img: Mat,
        scale: f64,
        min_neighbors: c_int,
        flags: c_int,
        min_size: Size,
        max_size: Size,
        allocator: std.mem.Allocator,
    ) !std.ArrayList(Rect) {
        const rec: c.struct_Rects = c.CascadeClassifier_DetectMultiScaleWithParams(
            self.ptr,
            img.ptr,
            scale,
            min_neighbors,
            flags,
            min_size.toC(),
            max_size.toC(),
        );

        var return_rects = std.ArrayList(Rect).init(allocator);
        {
            var i: usize = 0;
            while (i < rec.length) : (i += 1) {
                try return_rects.append(Rect.initFromC(rec.rects[i]));
            }
        }
        return return_rects;
    }
};

// HOGDescriptor is a Histogram Of Gradiants (HOG) for object detection.
//
// For further details, please see:
// https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a723b95b709cfd3f95cf9e616de988fc8
//
pub const HOGDescriptor = struct {
    ptr: c.HOGDescriptor,

    const Self = @This();

    pub fn init() Self {
        return .{ .ptr = c.HOGDescriptor_New() };
    }

    pub fn deinit(self: *Self) void {
        _ = c.HOGDescriptor_Close(self.ptr);
    }

    pub fn load(self: *Self, name: []const u8) !void {
        const result = c.HOGDescriptor_Load(self.ptr, utils.castZigU8ToC(name));
        if (result == 0) {
            return error.HOGDescriptorLoadFailed;
        }
    }

    // DetectMultiScale detects objects in the input Mat image.
    // The detected objects are returned as a slice of image.Rectangle structs.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a660e5cd036fd5ddf0f5767b352acd948
    //
    pub fn detectMultiScale(self: Self, img: Mat, allocator: std.mem.Allocator) !std.ArrayList(Rect) {
        const rec: c.struct_Rects = c.HOGDescriptor_DetectMultiScale(self.ptr, img.ptr);

        var return_rects = std.ArrayList(Rect).init(allocator);
        {
            var i: usize = 0;
            while (i < rec.length) : (i += 1) {
                try return_rects.append(Rect.initFromC(rec.rects[i]));
            }
        }
        return return_rects;
    }

    // DetectMultiScaleWithParams calls DetectMultiScale but allows setting parameters
    // to values other than just the defaults.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a660e5cd036fd5ddf0f5767b352acd948
    //
    pub fn detectMultiScaleWithParams(
        self: Self,
        img: Mat,
        hit_thresh: f64,
        win_stride: Size,
        padding: Size,
        scale: f64,
        final_threshold: f64,
        use_meanshift_grouping: bool,
        allocator: std.mem.Allocator,
    ) !std.ArrayList(Rect) {
        const rec: c.struct_Rects = c.HOGDescriptor_DetectMultiScaleWithParams(
            self.ptr,
            img.ptr,
            hit_thresh,
            win_stride.toC(),
            padding.toC(),
            scale,
            final_threshold,
            use_meanshift_grouping,
        );

        var return_rects = std.ArrayList(Rect).init(allocator);
        {
            var i: usize = 0;
            while (i < rec.length) : (i += 1) {
                try return_rects.append(Rect.initFromC(rec.rects[i]));
            }
        }
        return return_rects;
    }

    // HOGDefaultPeopleDetector returns a new Mat with the HOG DefaultPeopleDetector.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a660e5cd036fd5ddf0f5767b352acd948
    //
    pub fn getDefaultPeopleDetector() !Mat {
        return Mat.initFromC(c.HOG_GetDefaultPeopleDetector());
    }

    // SetSVMDetector sets the data for the HOGDescriptor.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a09e354ad701f56f9c550dc0385dc36f1
    //
    pub fn descriptorSetSVMDetector(self: Self, det: *Mat) void {
        _ = c.HOGDescriptor_SetSVMDetector(self.ptr, det.*.ptr);
    }
};

// QRCodeDetector groups the object candidate rectangles.
//
// For further details, please see:
// https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html
//
pub const QRCodeDetector = struct {
    ptr: c.QRCodeDetector,

    const Self = @This();

    pub fn init() Self {
        return .{ .ptr = c.QRCodeDetector_New() };
    }

    pub fn deinit(self: *Self) void {
        _ = c.QRCodeDetector_Close(self.ptr);
    }

    // DetectAndDecode Both detects and decodes QR code.
    //
    // Returns true as long as some QR code was detected even in case where the decoding failed
    // For further details, please see:
    // https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html#a7290bd6a5d59b14a37979c3a14fbf394
    //
    pub fn detectAndDecode(self: Self, input: Mat, points: *Mat, straight_qrcode: *Mat) []const u8 {
        const result = c.QRCoeDetector_DetectAndDecode(self.ptr, input.ptr, points.*.ptr, straight_qrcode.*.ptr);
        return utils.castZigU8ToC(result);
    }

    // Detect detects QR code in image and returns the quadrangle containing the code.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html#a64373f7d877d27473f64fe04bb57d22b
    //
    pub fn detect(self: Self, input: Mat, points: *Mat) bool {
        return c.QRCodeDetector_Detect(self.ptr, input.ptr, points.*.ptr);
    }

    // Decode decodes QR code in image once it's found by the detect() method. Returns UTF8-encoded output string or empty string if the code cannot be decoded.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html#a4172c2eb4825c844fb1b0ae67202d329
    //
    pub fn decode(self: Self, input: Mat, points: Mat, straight_qrcode: *Mat) []const u8 {
        const result = c.QRCodeDetector_Decode(self.ptr, input.ptr, points.ptr, straight_qrcode.*.ptr);
        return utils.castZigU8ToC(result);
    }

    // Detects QR codes in image and finds of the quadrangles containing the codes.
    //
    // Each quadrangle would be returned as a row in the `points` Mat and each point is a Vecf.
    // Returns true if QR code was detected
    // For usage please see TestQRCodeDetector
    // For further details, please see:
    // https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html#aaf2b6b2115b8e8fbc9acf3a8f68872b6
    pub fn DetectMulti(self: Self, input: Mat, points: *Mat) bool {
        return c.QRCodeDetector_DetectMulti(self.ptr, input.ptr, points.*.ptr);
    }

    // Decode decodes QR code in image once it's found by the detect() method. Returns UTF8-encoded output string or empty string if the code cannot be decoded.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html#a4172c2eb4825c844fb1b0ae67202d329
    //
    const return_detectAndDecodeMulti = struct {
        is_detected: bool,
        decoded: std.ArrayList(std.ArrayList(u8)),
        qr_codes: std.ArrayList(Mat),
        points: Mat,
    };
    pub fn detectAndDecodeMulti(
        self: Self,
        input: Mat,
        allocator: std.mem.Allocator,
    ) !return_detectAndDecodeMulti {
        var c_decoded = c.CStrings{};
        defer c.CStrings_Close(c_decoded);
        var c_qr_codes = c.Mats{};
        defer c.Mats_Close(c_qr_codes);

        var points = Mat.init();

        const result = c.QRCodeDetector_DetectAndDecodeMulti(
            self.ptr,
            input.ptr,
            &c_decoded,
            points.ptr,
            &c_qr_codes,
        );
        var decoded = std.ArrayList(std.ArrayList(u8));
        {
            var i: usize = 0;
            while (i < c_decoded.length) : (i += 1) {
                const s = std.ArrayList(u8).init(allocator);
                {
                    var j: usize = 0;
                    while (true) {
                        const char = c_decoded.strs[i][j];
                        if (char == 0) break;
                        try s.append(char);
                    }
                }
                try decoded.append(s);
            }
        }

        var qr_codes: std.ArrayList(Mat) = undefined;
        {
            var i: usize = 0;
            while (i < c_qr_codes.length) {
                const cmm: c.Mat = c_qr_codes.mats[i];
                const mm: Mat = try Mat.initFromC(cmm);
                try qr_codes.append(Mat.clone(mm));
            }
        }

        return .{
            .is_detected = result,
            .points = points,
            .decoded = decoded,
            .points = points,
            .qr_codes = qr_codes,
        };
    }
};

//*    implementation done
//*    pub const CascadeClassifier = ?*anyopaque;
//*    pub const HOGDescriptor = ?*anyopaque;
//*    pub const QRCodeDetector = ?*anyopaque;
//*    pub extern fn CascadeClassifier_New(...) CascadeClassifier;
//*    pub extern fn CascadeClassifier_Close(cs: CascadeClassifier) void;
//*    pub extern fn CascadeClassifier_Load(cs: CascadeClassifier, name: [*c]const u8) c_int;
//*    pub extern fn CascadeClassifier_DetectMultiScale(cs: CascadeClassifier, img: Mat) struct_Rects;
//*    pub extern fn CascadeClassifier_DetectMultiScaleWithParams(cs: CascadeClassifier, img: Mat, scale: f64, minNeighbors: c_int, flags: c_int, minSize: Size, maxSize: Size) struct_Rects;
//*    pub extern fn HOGDescriptor_New(...) HOGDescriptor;
//*    pub extern fn HOGDescriptor_Close(hog: HOGDescriptor) void;
//*    pub extern fn HOGDescriptor_Load(hog: HOGDescriptor, name: [*c]const u8) c_int;
//*    pub extern fn HOGDescriptor_DetectMultiScale(hog: HOGDescriptor, img: Mat) struct_Rects;
//*    pub extern fn HOGDescriptor_DetectMultiScaleWithParams(hog: HOGDescriptor, img: Mat, hitThresh: f64, winStride: Size, padding: Size, scale: f64, finalThreshold: f64, useMeanshiftGrouping: bool) struct_Rects;
//*    pub extern fn HOG_GetDefaultPeopleDetector(...) Mat;
//*    pub extern fn HOGDescriptor_SetSVMDetector(hog: HOGDescriptor, det: Mat) void;
//*    pub extern fn GroupRectangles(rects: struct_Rects, groupThreshold: c_int, eps: f64) struct_Rects;
//*    pub extern fn QRCodeDetector_New(...) QRCodeDetector;
//*    pub extern fn QRCodeDetector_DetectAndDecode(qr: QRCodeDetector, input: Mat, points: Mat, straight_qrcode: Mat) [*c]const u8;
//*    pub extern fn QRCodeDetector_Detect(qr: QRCodeDetector, input: Mat, points: Mat) bool;
//*    pub extern fn QRCodeDetector_Decode(qr: QRCodeDetector, input: Mat, inputPoints: Mat, straight_qrcode: Mat) [*c]const u8;
//*    pub extern fn QRCodeDetector_Close(qr: QRCodeDetector) void;
//*    pub extern fn QRCodeDetector_DetectMulti(qr: QRCodeDetector, input: Mat, points: Mat) bool;
//*    pub extern fn QRCodeDetector_DetectAndDecodeMulti(qr: QRCodeDetector, input: Mat, decoded: [*c]CStrings, points: Mat, mats: [*c]struct_Mats) bool;
