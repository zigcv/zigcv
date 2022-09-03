const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const Mat = core.Mat;
const Size = core.Size;
const Scalar = core.Scalar;
const AsyncArray = @import("asyncarray.zig").AsyncArray;

pub const NetBackendType = enum(i32) {
    // NetBackendDefault is the default backend.
    NetBackendDefault = 0,

    // NetBackendHalide is the Halide backend.
    NetBackendHalide = 1,

    // NetBackendOpenVINO is the OpenVINO backend.
    NetBackendOpenVINO = 2,

    // NetBackendOpenCV is the OpenCV backend.
    NetBackendOpenCV = 3,

    // NetBackendVKCOM is the Vulkan backend.
    NetBackendVKCOM = 4,

    // NetBackendCUDA is the Cuda backend.
    NetBackendCUDA = 5,

    // ParseNetBackend returns a valid NetBackendType given a string. Valid values are:
    // - halide
    // - openvino
    // - opencv
    // - vulkan
    // - cuda
    // - default
    pub fn parse(backend: []const u8) @This() {
        return switch (backend) {
            "halide" => .NetBackendHalide,
            "openvino" => .NetBackendOpenVINO,
            "opencv" => .NetBackendOpenCV,
            "vulkan" => .NetBackendVKCOM,
            "cuda" => .NetBackendCUDA,
            else => .NetBackendDefault,
        };
    }
};

pub const NetTargetType = enum(i32) {
    // NetTargetCPU is the default CPU device target.
    NetTargetCPU = 0,

    // NetTargetFP32 is the 32-bit OpenCL target.
    NetTargetFP32 = 1,

    // NetTargetFP16 is the 16-bit OpenCL target.
    NetTargetFP16 = 2,

    // NetTargetVPU is the Movidius VPU target.
    NetTargetVPU = 3,

    // NetTargetVulkan is the NVIDIA Vulkan target.
    NetTargetVulkan = 4,

    // NetTargetFPGA is the FPGA target.
    NetTargetFPGA = 5,

    // NetTargetCUDA is the CUDA target.
    NetTargetCUDA = 6,

    // NetTargetCUDAFP16 is the CUDA target.
    NetTargetCUDAFP16 = 7,

    pub fn parse(target: []const u8) @This() {
        return switch (target) {
            "fp32" => .NetTargetFP32,
            "fp16" => .NetTargetFP16,
            "vpu" => .NetTargetVPU,
            "vulkan" => .NetTargetVulkan,
            "fpga" => .NetTargetFPGA,
            "cuda" => .NetTargetCUDA,
            "cuda_fp16" => .NetTargetCUDAFP16,
            "cpu" => .NetTargetCPU,
            else => .NetTargetCPU,
        };
    }
};

pub const Net = struct {
    ptr: c.Net,

    const Self = @This();

    pub fn init() Self {
        return Self{ .ptr = null };
    }

    pub fn fromC(ptr: c.Net) !Self {
        if (ptr == null) {
            return error.RuntimeError;
        }
        return Self{ .ptr = ptr };
    }

    pub fn deinit(self: *Self) void {
        if (self.ptr != null) _ = c.Net_Close(self.ptr);
    }

    pub fn readNet(model: []const u8, config: []const u8) !Self {
        return try Self.fromC(c.Net_ReadNet(utils.castZigU8ToC(model), utils.castZigU8ToC(config)));
    }

    pub fn readNetFromCaffe(prototxt: []const u8, caffe_model: []const u8) !Self {
        return try Self.fromC(c.Net_ReadNetFromCaffe(utils.castZigU8ToC(prototxt), utils.castZigU8ToC(caffe_model)));
    }

    pub fn readNetFromTensorflow(model: []const u8) !Self {
        return try Self.fromC(c.Net_ReadNetFromTensorflow(utils.castZigU8ToC(model)));
    }

    pub fn readNetFromTorch(model: []const u8) !Self {
        return try Self.fromC(c.Net_ReadNetFromTorch(utils.castZigU8ToC(model)));
    }

    pub fn readNetFromONNX(model: []const u8) !Self {
        return try Self.fromC(c.Net_ReadNetFromONNX(utils.castZigU8ToC(model)));
    }

    pub fn empty(self: Self) bool {
        if (self.ptr == null) return false;
        return c.Net_Empty(self.ptr);
    }

    pub fn setInput(self: *Self, blob: Mat, name: []const u8) !void {
        if (self.ptr == null) return error.nullPointerError;
        _ = c.Net_SetInput(self.ptr, blob.ptr, utils.castZigU8ToC(name));
    }

    pub fn forward(self: *Self, output_name: []const u8) !Mat {
        if (self.ptr == null) return error.nullPointerError;
        return Mat.fromC(c.Net_Forward(self.ptr, utils.castZigU8ToC(output_name)));
    }

    pub fn forwardAsync(self: *Self, output_name: []const u8) !AsyncArray {
        if (self.ptr == null) return error.nullPointerError;
        return AsyncArray.fromC(c.Net_ForwardAsync(self.*.ptr, utils.castZigU8ToC(output_name)));
    }

    // SetPreferableBackend ask network to use specific computation backend.
    //
    // For further details, please see:
    // https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#a7f767df11386d39374db49cd8df8f59e
    //
    pub fn setPreferableBackend(self: *Self, backend: NetBackendType) !void {
        if (self.ptr == null) return error.nullPointerError;
        _ = c.Net_SetPreferableBackend(self.ptr, @enumToInt(backend));
    }

    // SetPreferableTarget ask network to make computations on specific target device.
    //
    // For further details, please see:
    // https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#a9dddbefbc7f3defbe3eeb5dc3d3483f4
    pub fn setPreferableTarget(self: *Self, target: NetTargetType) !void {
        if (self.ptr == null) return error.nullPointerError;
        _ = c.Net_SetPreferableTarget(self.ptr, @enumToInt(target));
    }

    // GetPerfProfile returns overall time for inference and timings (in ticks) for layers
    //
    // For further details, please see:
    // https://docs.opencv.org/master/db/d30/classcv_1_1dnn_1_1Net.html#a06ce946f675f75d1c020c5ddbc78aedc
    //
    pub fn getPerfProfile(self: Self) !f64 {
        if (self.ptr == null) return error.nullPointerError;
        return c.Net_GetPerfProfile(self.ptr);
    }

    pub fn getUnconnectedOutLayers(self: Self, allocator: std.mem.Allocator) !std.ArrayList(i32) {
        if (self.ptr == null) return error.nullPointerError;
        var res = c.IntVector{};
        defer c.IntVector_Close(res);
        _ = c.Net_GetUnconnectedOutLayers(self.ptr, &res);
        var return_res = std.ArrayList(i32).init(allocator);
        {
            var i: usize = 0;
            while (i < res.length) : (i += 1) {
                try return_res.append(res.val[i]);
            }
        }
        return return_res;
    }

    pub fn getLayerNames(self: *Self) []const u8 {
        if (self.ptr == null) return error.nullPointerError;
        const c_strs = c.CStrings{};
        defer c.CStrings_Close(c_strs);
        _ = c.Net_GetLayerNames(self.ptr, &c_strs);
        return std.mem.span(c_strs);
    }

    pub fn getLayer(self: Self, layerid: c_int) !Layer {
        if (self.ptr == null) return error.nullPointerError;
        return Layer.fromC(c.Net_GetLayer(self.ptr, layerid));
    }

    pub fn getBlobChannel(self: Self, imgidx: c_int, chnidx: c_int) !Mat {
        if (self.ptr == null) return error.nullPointerError;
        return try Mat.fromC(c.Net_GetBlobChannel(self.ptr, imgidx, chnidx));
    }

    pub fn getBlobSize(self: Self, blob: Mat) !Size {
        if (self.ptr == null) return error.nullPointerError;
        return Size.fromC(c.Net_GetBlobSize(self.ptr, blob.ptr));
    }
};

// BlobFromImage creates 4-dimensional blob from image. Optionally resizes and crops
// image from center, subtract mean values, scales values by scalefactor,
// swap Blue and Red channels.
//
// For further details, please see:
// https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga152367f253c81b53fe6862b299f5c5cd
//
pub fn blobFromImage(image: Mat, scalefactor: f64, size: Size, mean: Scalar, swap_rb: bool, crop: bool) !Mat {
    return try Mat.fromC(c.Net_BlobFromImage(image.ptr, scalefactor, size.toC(), mean.toC(), swap_rb, crop));
}

// BlobFromImages Creates 4-dimensional blob from series of images.
// Optionally resizes and crops images from center, subtract mean values,
// scales values by scalefactor, swap Blue and Red channels.
//
// For further details, please see:
// https://docs.opencv.org/master/d6/d0f/group__dnn.html#ga2b89ed84432e4395f5a1412c2926293c
//
pub fn blobFromImages(images: []const Mat, blob: *Mat, scalefactor: f64, size: Size, mean: Scalar, swap_r_b: bool, crop: bool, ddepth: c_int, allocator: std.mem.Allocator) !void {
    var c_mats = try core.matsToCMats(images, allocator);
    c.Net_BlobFromImages(c_mats, blob.*.ptr, scalefactor, size.toC(), mean.toC(), swap_r_b, crop, ddepth);
}

pub const Layer = struct {
    ptr: ?c.Layer,

    const Self = @This();

    pub fn fromC(ptr: c.Layer) Self {
        return Self{ .ptr = ptr };
    }

    pub fn deinit(self: *Self) void {
        if (self.ptr != null) _ = c.Layer_Close(self.ptr);
    }

    pub fn toC(self: Self) c.Layer {
        return self.ptr;
    }

    pub fn inputNameToIndex(self: *Self, name: []const u8) !c_int {
        if (self.ptr == null) return error.nullPointerError;
        return c.Layer_InputNameToIndex(self.ptr, utils.castZigU8ToC(name));
    }

    pub fn outputNameToIndex(self: *Self, name: []const u8) !c_int {
        if (self.ptr == null) return error.nullPointerError;
        return c.Layer_OutputNameToIndex(self.ptr, utils.castZigU8ToC(name));
    }

    pub fn getName(self: Self) ![]const u8 {
        if (self.ptr == null) return error.nullPointerError;
        return utils.castZigU8ToC(c.Layer_GetName(self.ptr));
    }

    pub fn getType(self: Self) ![]const u8 {
        if (self.ptr == null) return error.nullPointerError;
        return utils.castZigU8ToC(c.Layer_GetType(self.ptr));
    }
};

//*    implementation done
//*    pub const Net = ?*anyopaque;
//*    pub const Layer = ?*anyopaque;
//*    pub extern fn Net_ReadNet(model: [*c]const u8, config: [*c]const u8) Net;
//     pub extern fn Net_ReadNetBytes(framework: [*c]const u8, model: struct_ByteArray, config: struct_ByteArray) Net;
//*    pub extern fn Net_ReadNetFromCaffe(prototxt: [*c]const u8, caffeModel: [*c]const u8) Net;
//     pub extern fn Net_ReadNetFromCaffeBytes(prototxt: struct_ByteArray, caffeModel: struct_ByteArray) Net;
//*    pub extern fn Net_ReadNetFromTensorflow(model: [*c]const u8) Net;
//     pub extern fn Net_ReadNetFromTensorflowBytes(model: struct_ByteArray) Net;
//*    pub extern fn Net_ReadNetFromTorch(model: [*c]const u8) Net;
//*    pub extern fn Net_ReadNetFromONNX(model: [*c]const u8) Net;
//     pub extern fn Net_ReadNetFromONNXBytes(model: struct_ByteArray) Net;
//*    pub extern fn Net_BlobFromImage(image: Mat, scalefactor: f64, size: Size, mean: Scalar, swapRB: bool, crop: bool) Mat;
//*    pub extern fn Net_BlobFromImages(images: struct_Mats, blob: Mat, scalefactor: f64, size: Size, mean: Scalar, swapRB: bool, crop: bool, ddepth: c_int) void;
//     pub extern fn Net_ImagesFromBlob(blob_: Mat, images_: [*c]struct_Mats) void;
//*    pub extern fn Net_Close(net: Net) void;
//*    pub extern fn Net_Empty(net: Net) bool;
//*    pub extern fn Net_SetInput(net: Net, blob: Mat, name: [*c]const u8) void;
//*    pub extern fn Net_Forward(net: Net, outputName: [*c]const u8) Mat;
//     pub extern fn Net_ForwardLayers(net: Net, outputBlobs: [*c]struct_Mats, outBlobNames: struct_CStrings) void;
//*    pub extern fn Net_SetPreferableBackend(net: Net, backend: c_int) void;
//*    pub extern fn Net_SetPreferableTarget(net: Net, target: c_int) void;
//*    pub extern fn Net_GetPerfProfile(net: Net) i64;
//*    pub extern fn Net_GetUnconnectedOutLayers(net: Net, res: [*c]IntVector) void;
//*    pub extern fn Net_GetLayerNames(net: Net, names: [*c]CStrings) void;
//*    pub extern fn Net_GetBlobChannel(blob: Mat, imgidx: c_int, chnidx: c_int) Mat;
//*    pub extern fn Net_GetBlobSize(blob: Mat) Scalar;
//*    pub extern fn Net_GetLayer(net: Net, layerid: c_int) Layer;
//*    pub extern fn Layer_Close(layer: Layer) void;
//*    pub extern fn Layer_InputNameToIndex(layer: Layer, name: [*c]const u8) c_int;
//*    pub extern fn Layer_OutputNameToIndex(layer: Layer, name: [*c]const u8) c_int;
//*    pub extern fn Layer_GetName(layer: Layer) [*c]const u8;
//*    pub extern fn Layer_GetType(layer: Layer) [*c]const u8;
//     pub extern fn NMSBoxes(bboxes: struct_Rects, scores: FloatVector, score_threshold: f32, nms_threshold: f32, indices: [*c]IntVector) void;
//     pub extern fn NMSBoxesWithParams(bboxes: struct_Rects, scores: FloatVector, score_threshold: f32, nms_threshold: f32, indices: [*c]IntVector, eta: f32, top_k: c_int) void;
