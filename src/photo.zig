const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const Mat = core.Mat;
const Mats = core.Mats;
const Point = core.Point;

pub const SeamlessCloneFlag = enum(u2) {
    // NormalClone The power of the method is fully expressed when inserting objects with complex outlines into a new background.
    NormalClone = 0,

    // MixedClone The classic method, color-based selection and alpha masking might be time consuming and often leaves an undesirable halo. Seamless cloning, even averaged with the original image, is not effective. Mixed seamless cloning based on a loose selection proves effective.
    MixedClone,

    // MonochromeTransfer Monochrome transfer allows the user to easily replace certain features of one object by alternative features.
    MonochromeTransfer,
};

pub const EdgeFilter = enum(u1) {
    // RecursFilter Recursive Filtering.
    RecursFilter = 1,

    // NormconvFilter Normalized Convolution Filtering.
    NormconvFilter = 2,
};

// ColorChange mix two differently colored versions of an image seamlessly.
//
// For further details, please see:
// https://docs.opencv.org/master/df/da0/group__photo__clone.html#ga6684f35dc669ff6196a7c340dc73b98e
//
pub fn colorChange(src: Mat, mask: Mat, dst: *Mat, red_mul: f32, green_mul: f32, blue_mul: f32) void {
    _ = c.ColorChange(src.ptr, mask.ptr, dst.*.ptr, red_mul, green_mul, blue_mul);
}

// SeamlessClone blend two image by Poisson Blending.
//
// For further details, please see:
// https://docs.opencv.org/master/df/da0/group__photo__clone.html#ga2bf426e4c93a6b1f21705513dfeca49d
//
pub fn seamlessClone(src: Mat, dst: *Mat, mask: Mat, p: Point, blend: Mat, flags: SeamlessCloneFlag) void {
    _ = c.SeamlessClone(src.ptr, dst.*.ptr, mask.ptr, p.toC(), blend.ptr, @enumToInt(flags));
}

// IlluminationChange modifies locally the apparent illumination of an image.
//
// For further details, please see:
// https://docs.opencv.org/master/df/da0/group__photo__clone.html#gac5025767cf2febd8029d474278e886c7
//
pub fn IlluminationChange(src: Mat, mask: Mat, dst: *Mat, alpha: f32, beta: f32) void {
    _ = c.IlluminationChange(src.ptr, mask.ptr, dst.*.ptr, alpha, beta);
}

// TextureFlattening washes out the texture of the selected region, giving its contents a flat aspect.
//
// For further details, please see:
// https://docs.opencv.org/master/df/da0/group__photo__clone.html#gad55df6aa53797365fa7cc23959a54004
//
pub fn TextureFlattening(src: Mat, mask: Mat, dst: *Mat, low_threshold: f32, high_threshold: f32, kernel_size: c_int) void {
    _ = c.TextureFlattening(src.ptr, mask.ptr, dst.*.ptr, low_threshold, high_threshold, kernel_size);
}

//     pub extern fn FastNlMeansDenoisingColoredMulti(src: struct_Mats, dst: Mat, imgToDenoiseIndex: c_int, temporalWindowSize: c_int) void;
// FastNlMeansDenoisingColoredMulti denoises the selected images.
//
// For further details, please see:
// https://docs.opencv.org/master/d1/d79/group__photo__denoise.html#gaa501e71f52fb2dc17ff8ca5e7d2d3619
//
pub fn fastNlMeansDenoisingColoredMulti(
    src: []Mat,
    dst: *Mat,
    img_to_denoise_index: c_int,
    temporal_window_size: c_int,
    allocator: std.mem.Allocator,
) !void {
    var c_mats = try Mat.toCStructs(src, allocator);
    defer Mat.deinitCStructs(c_mats, allocator);
    _ = c.FastNlMeansDenoisingColoredMulti(c_mats, dst.*.ptr, img_to_denoise_index, temporal_window_size);
}

// FastNlMeansDenoisingColoredMulti denoises the selected images.
//
// For further details, please see:
// https://docs.opencv.org/master/d1/d79/group__photo__denoise.html#gaa501e71f52fb2dc17ff8ca5e7d2d3619
//
pub fn FastNlMeansDenoisingColoredWithParams(
    src: []Mat,
    dst: *Mat,
    img_to_denoise_index: c_int,
    temporal_window_size: c_int,
    h: f32,
    h_color: f32,
    template_window_size: c_int,
    search_window_size: c_int,
    allocator: std.mem.Allocator,
) !void {
    var c_mats = try Mat.toCStructs(src, allocator);
    defer Mat.deinitCStructs(c_mats, allocator);
    _ = c.FastNlMeansDenoisingColoredMultiWithParams(
        c_mats,
        dst.*.ptr,
        img_to_denoise_index,
        temporal_window_size,
        h,
        h_color,
        template_window_size,
        search_window_size,
    );
}

// FastNlMeansDenoising performs image denoising using Non-local Means Denoising algorithm
// http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/
//
// For further details, please see:
// https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93
//
pub fn fastNlMeansDenoising(src: Mat, dst: *Mat) void {
    _ = c.FastNlMeansDenoising(src.ptr, dst.*.ptr);
}

// FastNlMeansDenoisingWithParams performs image denoising using Non-local Means Denoising algorithm
// http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/
//
// For further details, please see:
// https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93
//
pub fn fastNlMeansDenoisingWithParams(
    src: Mat,
    dst: *Mat,
    h: f32,
    template_window_size: c_int,
    search_window_size: c_int,
) void {
    _ = c.FastNlMeansDenoisingWithParams(
        src.ptr,
        dst.*.ptr,
        h,
        template_window_size,
        search_window_size,
    );
}

// FastNlMeansDenoisingColored is a modification of fastNlMeansDenoising function for colored images.
//
// For further details, please see:
// https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html#ga21abc1c8b0e15f78cd3eff672cb6c476
//
pub fn fastNlMeansDenoisingColored(src: Mat, dst: *Mat) void {
    _ = c.FastNlMeansDenoisingColored(src.ptr, dst.*.ptr);
}

// FastNlMeansDenoisingColoredWithParams is a modification of fastNlMeansDenoising function for colored images.
//
// For further details, please see:
// https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html#ga21abc1c8b0e15f78cd3eff672cb6c476
//
pub fn fastNlMeansDenoisingColoredWithParams(
    src: Mat,
    dst: *Mat,
    h: f32,
    h_color: f32,
    template_window_size: c_int,
    search_window_size: c_int,
) void {
    _ = c.FastNlMeansDenoisingColoredWithParams(
        src.ptr,
        dst.*.ptr,
        h,
        h_color,
        template_window_size,
        search_window_size,
    );
}

pub const MergeMertens = struct {
    ptr: c.MergeMertens,

    const Self = @This();

    pub fn init() Self {
        return .{ .ptr = c.MergeMertens_Create() };
    }

    pub fn initWithParams(contrast_weight: f32, saturation_weight: f32, exposure_weight: f32) Self {
        return .{ .ptr = c.MergeMertens_CreateWithParams(contrast_weight, saturation_weight, exposure_weight) };
    }

    pub fn deinit(self: *Self) void {
        _ = c.MergeMertens_Destroy(self.ptr);
    }

    pub fn process(self: *Self, src: []const Mat, dst: *Mat, allocator: std.mem.Allocator) !void {
        var c_mats: c.struct_Mats = try Mat.toCStructs(src, allocator);
        defer Mat.deinitCStructs(c_mats, allocator);
        _ = c.MergeMertens_Process(self.ptr, c_mats, dst.*.ptr);
    }
};

pub const AlignMTB = struct {
    ptr: c.AlignMTB,

    const Self = @This();

    pub fn init() Self {
        return .{ .ptr = c.AlignMTB_Create() };
    }

    pub fn initWithParams(max_bits: c_int, exclude_range: c_int, cut: bool) Self {
        return .{ .ptr = c.AlignMTB_CreateWithParams(max_bits, exclude_range, cut) };
    }

    pub fn deinit(self: *Self) void {
        _ = c.AlignMTB_Destroy(self.ptr);
    }

    pub fn process(self: Self, src: []const Mat, allocator: std.mem.Allocator) !Mats {
        var c_mats: c.struct_Mats = try Mat.toCStructs(src, allocator);
        defer Mat.deinitCStructs(c_mats, allocator);
        var c_dst_mats = c.struct_Mats{};
        _ = c.AlignMTB_Process(self.ptr, c_mats, &c_dst_mats);
        return Mats.toArrayList(c_dst_mats, allocator);
    }
};

// DetailEnhance filter enhances the details of a particular image
//
// For further details, please see:
// https://docs.opencv.org/4.x/df/dac/group__photo__render.html#gae5930dd822c713b36f8529b21ddebd0c
//
pub fn detailEnhance(src: Mat, dst: *Mat, sigma_s: f32, sigma_r: f32) void {
    _ = c.DetailEnhance(src.ptr, dst.*.ptr, sigma_s, sigma_r);
}

// EdgePreservingFilter filtering is the fundamental operation in image and video processing.
// Edge-preserving smoothing filters are used in many different applications.
//
// For further details, please see:
// https://docs.opencv.org/4.x/df/dac/group__photo__render.html#gafaee2977597029bc8e35da6e67bd31f7
//
pub fn edgePreservingFilter(src: Mat, dst: *Mat, flags: EdgeFilter, sigma_s: f32, sigma_r: f32) void {
    _ = c.EdgePreservingFilter(src.ptr, dst.*.ptr, @enumToInt(flags), sigma_s, sigma_r);
}

// EdgePreservingFilter filtering is the fundamental operation in image and video processing.
// Edge-preserving smoothing filters are used in many different applications.
//
// For further details, please see:
// https://docs.opencv.org/4.x/df/dac/group__photo__render.html#gafaee2977597029bc8e35da6e67bd31f7
//
pub fn edgePreservingFilterWithKernel(src: Mat, dst: *Mat, kernel: Mat) void {
    _ = c.EdgePreservingFilterWithKernel(src.ptr, dst.*.ptr, kernel.ptr);
}

// PencilSketch pencil-like non-photorealistic line drawing.
//
// For further details, please see:
// https://docs.opencv.org/4.x/df/dac/group__photo__render.html#gae5930dd822c713b36f8529b21ddebd0c
//
pub fn pencilSketch(src: Mat, dst1: *Mat, dst2: *Mat, sigma_s: f32, sigma_r: f32, shade_factor: f32) void {
    _ = c.PencilSketch(src.ptr, dst1.*.ptr, dst2.*.ptr, sigma_s, sigma_r, shade_factor);
}

//*    implementation done
//*    pub const MergeMertens = ?*anyopaque;
//*    pub const AlignMTB = ?*anyopaque;
//*    pub extern fn ColorChange(src: Mat, mask: Mat, dst: Mat, red_mul: f32, green_mul: f32, blue_mul: f32) void;
//*    pub extern fn SeamlessClone(src: Mat, dst: Mat, mask: Mat, p: Point, blend: Mat, flags: c_int) void;
//*    pub extern fn IlluminationChange(src: Mat, mask: Mat, dst: Mat, alpha: f32, beta: f32) void;
//*    pub extern fn TextureFlattening(src: Mat, mask: Mat, dst: Mat, low_threshold: f32, high_threshold: f32, kernel_size: c_int) void;
//*    pub extern fn FastNlMeansDenoisingColoredMulti(src: struct_Mats, dst: Mat, imgToDenoiseIndex: c_int, temporalWindowSize: c_int) void;
//*    pub extern fn FastNlMeansDenoisingColoredMultiWithParams(src: struct_Mats, dst: Mat, imgToDenoiseIndex: c_int, temporalWindowSize: c_int, h: f32, hColor: f32, templateWindowSize: c_int, searchWindowSize: c_int) void;
//*    pub extern fn FastNlMeansDenoising(src: Mat, dst: Mat) void;
//*    pub extern fn FastNlMeansDenoisingWithParams(src: Mat, dst: Mat, h: f32, templateWindowSize: c_int, searchWindowSize: c_int) void;
//*    pub extern fn FastNlMeansDenoisingColored(src: Mat, dst: Mat) void;
//*    pub extern fn FastNlMeansDenoisingColoredWithParams(src: Mat, dst: Mat, h: f32, hColor: f32, templateWindowSize: c_int, searchWindowSize: c_int) void;
//*    pub extern fn MergeMertens_Create(...) MergeMertens;
//*    pub extern fn MergeMertens_CreateWithParams(contrast_weight: f32, saturation_weight: f32, exposure_weight: f32) MergeMertens;
//*    pub extern fn MergeMertens_Process(b: MergeMertens, src: struct_Mats, dst: Mat) void;
//*    pub extern fn MergeMertens_Close(b: MergeMertens) void;
//*    pub extern fn AlignMTB_Create(...) AlignMTB;
//*    pub extern fn AlignMTB_CreateWithParams(max_bits: c_int, exclude_range: c_int, cut: bool) AlignMTB;
//*    pub extern fn AlignMTB_Process(b: AlignMTB, src: struct_Mats, dst: [*c]struct_Mats) void;
//*    pub extern fn AlignMTB_Close(b: AlignMTB) void;
//*    pub extern fn DetailEnhance(src: Mat, dst: Mat, sigma_s: f32, sigma_r: f32) void;
//*    pub extern fn EdgePreservingFilter(src: Mat, dst: Mat, filter: c_int, sigma_s: f32, sigma_r: f32) void;
//*    pub extern fn PencilSketch(src: Mat, dst1: Mat, dst2: Mat, sigma_s: f32, sigma_r: f32, shade_factor: f32) void;
//*    pub extern fn Stylization(src: Mat, dst: Mat, sigma_s: f32, sigma_r: f32) void;
