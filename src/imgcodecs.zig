const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const castToC = utils.castZigU8ToC;
const Mat = core.Mat;

pub const IMReadFlag = enum(i9) {
    // https://docs.opencv.org/4.6.0/d6/d87/imgcodecs_8hpp.html
    // IMReadUnchanged return the loaded image as is (with alpha channel,
    // otherwise it gets cropped).
    unchanged = -1,

    // IMReadGrayScale always convert image to the single channel
    // grayscale image.
    gray_scale = 0,

    // IMReadColor always converts image to the 3 channel BGR color image.
    color = 1,

    // IMReadAnyDepth returns 16-bit/32-bit image when the input has the corresponding
    // depth, otherwise convert it to 8-bit.
    any_depth = 2,

    // IMReadAnyColor the image is read in any possible color format.
    any_color = 4,

    // IMReadLoadGDAL uses the gdal driver for loading the image.
    load_GDAL = 8,

    // IMReadReducedGrayscale2 always converts image to the single channel grayscale image
    // and the image size reduced 1/2.
    reduced_grayscale2 = 16,

    // IMReadReducedColor2 always converts image to the 3 channel BGR color image and the
    // image size reduced 1/2.
    reduced_color2 = 17,

    // IMReadReducedGrayscale4 always converts image to the single channel grayscale image and
    // the image size reduced 1/4.
    reduced_grayscale4 = 32,

    // IMReadReducedColor4 always converts image to the 3 channel BGR color image and
    // the image size reduced 1/4.
    reduced_color4 = 33,

    // IMReadReducedGrayscale8 always convert image to the single channel grayscale image and
    // the image size reduced 1/8.
    reduced_grayscale8 = 64,

    // IMReadReducedColor8 always convert image to the 3 channel BGR color image and the
    // image size reduced 1/8.
    reduced_color8 = 65,

    // IMReadIgnoreOrientation do not rotate the image according to EXIF's orientation flag.
    ignore_orientation = 128,
};

pub const IMWriteFlag = enum(u32) {
    //IMWriteJpegQuality is the quality from 0 to 100 for JPEG (the higher is the better). Default value is 95.
    jpeg_quality = 1,

    // IMWriteJpegProgressive enables JPEG progressive feature, 0 or 1, default is False.
    jpeg_progressive = 2,

    // IMWriteJpegOptimize enables JPEG optimization, 0 or 1, default is False.
    jpeg_optimize = 3,

    // IMWriteJpegRstInterval is the JPEG restart interval, 0 - 65535, default is 0 - no restart.
    jpeg_rst_interval = 4,

    // IMWriteJpegLumaQuality separates luma quality level, 0 - 100, default is 0 - don't use.
    jpeg_luma_quality = 5,

    // IMWriteJpegChromaQuality separates chroma quality level, 0 - 100, default is 0 - don't use.
    jpeg_chroma_quality = 6,

    // IMWritePngCompression is the compression level from 0 to 9 for PNG. A
    // higher value means a smaller size and longer compression time.
    // If specified, strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT (Z_DEFAULT_STRATEGY).
    // Default value is 1 (best speed setting).
    png_compression = 16,

    // IMWritePngStrategy is one of cv::IMWritePNGFlags, default is IMWRITE_PNG_STRATEGY_RLE.
    png_strategy = 17,

    // IMWritePngBilevel is the binary level PNG, 0 or 1, default is 0.
    png_bilevel = 18,

    // IMWritePxmBinary for PPM, PGM, or PBM can be a binary format flag, 0 or 1. Default value is 1.
    pxm_binary = 32,

    // IMWriteWebpQuality is the quality from 1 to 100 for WEBP (the higher is
    // the better). By default (without any parameter) and for quality above
    // 100 the lossless compression is used.
    webp_quality = 64,

    // IMWritePamTupletype sets the TUPLETYPE field to the corresponding string
    // value that is defined for the format.
    pam_tupletype = 128,

    tiff_resunit = 256,

    tiff_xres = 257,

    tiff_ydpi = 258,

    tiff_compression = 259,

    jpeg2000_compression_x1000 = 272,
};

pub const FileExt = enum(u4) {
    png,
    jpg,
    gif,

    pub fn toString(fe: @This()) []const u8 {
        return switch (fe) {
            .png => "png",
            .jpg => "jpg",
            .gif => "gif",
        };
    }
};

// IMRead reads an image from a file into a Mat.
// The flags param is one of the IMReadFlag flags.
// If the image cannot be read (because of missing file, improper permissions,
// unsupported or invalid format), the function returns an empty Mat.
//
// For further details, please see:
// http://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
//
pub fn imRead(filename: []const u8, flags: IMReadFlag) !Mat {
    var cMat: c.Mat = c.Image_IMRead(castToC(filename), @enumToInt(flags));
    return try Mat.fromC(cMat);
}

// IMWrite writes a Mat to an image file.
//
// For further details, please see:
// http://docs.opencv.org/master/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
//
pub fn imWrite(filename: []const u8, img: Mat) !void {
    const result = c.Image_IMWrite(castToC(filename), img.ptr);
    if (!result) {
        return error.IMWriteFailed;
    }
}

// IMWriteWithParams writes a Mat to an image file. With that func you can
// pass compression parameters.
//
// For further details, please see:
// http://docs.opencv.org/master/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
// https://docs.opencv.org/4.6.0/d8/d6a/group__imgcodecs__flags.html
//
pub fn imWriteWithParams(filename: []const u8, img: Mat, comptime params: []const struct { f: IMWriteFlag, v: i32 }) !void {
    comptime var int_params: [params.len * 2]c_int = undefined;
    for (params) |p, i| {
        int_params[2 * i] = @enumToInt(p.f);
        int_params[2 * i + 1] = @enumToInt(p.v);
    }
    comptime var c_params = c.IntVector{
        .val = @ptrCast([*]c_int, int_params),
        .length = params.len,
    };
    const result = c.Image_IMWrite_WithParams(castToC(filename), img.ptr, c_params);
    if (!result) {
        return error.IMWriteFailed;
    }
}

// IMDecode reads an image from a buffer in memory.
// The function IMDecode reads an image from the specified buffer in memory.
// If the buffer is too short or contains invalid data, the function
// returns an empty matrix.
//
// For further details, please see:
// https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga26a67788faa58ade337f8d28ba0eb19e
//
pub fn imDecode(buf: []u8, flags: IMReadFlag) !Mat {
    var data = @ptrCast([*]u8, buf);
    return try Mat.fromC(c.Image_IMDecode(data, @enumToInt(flags)));
}

// IMEncode encodes an image Mat into a memory buffer.
// This function compresses the image and stores it in the returned memory buffer,
// using the image format passed in in the form of a file extension string.
//
// For further details, please see:
// http://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga461f9ac09887e47797a54567df3b8b63
//
pub fn imEncode(file_ext: FileExt, img: Mat) ![]c_int {
    var c_vector = c.IntVector;
    c.Image_IMEncode(file_ext.toString(), img.ptr, &c_vector);
    return std.mem.span(c_vector.val)[0..c_vector.length];
}

// IMEncodeWithParams encodes an image Mat into a memory buffer.
// This function compresses the image and stores it in the returned memory buffer,
// using the image format passed in in the form of a file extension string.
//
// Usage example:
//  buffer, err := gocv.IMEncodeWithParams(gocv.JPEGFileExt, img, []int{gocv.IMWriteJpegQuality, quality})
//
// For further details, please see:
// http://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga461f9ac09887e47797a54567df3b8b63
//
pub fn imEncodeWithParams(file_ext: FileExt, img: Mat, comptime params: []const struct { f: IMWriteFlag, v: i32 }) []c_int {
    comptime var int_params: [params.len * 2]c_int = undefined;
    for (params) |p, i| {
        int_params[2 * i] = @enumToInt(p.f);
        int_params[2 * i + 1] = @enumToInt(p.v);
    }
    comptime var c_params = c.IntVector{
        .val = @ptrCast([*]c_int, int_params),
        .length = params.len,
    };
    var c_vector = c.IntVector;
    c.Image_IMEncode_WithParams(file_ext.toString(), img.ptr, c_params, c_vector);
    return std.mem.span(c_vector.val)[0..c_vector.length];
}

//*    implementation done
//*    pub extern fn Image_IMRead(filename: [*c]const u8, flags: c_int) Mat;
//*    pub extern fn Image_IMWrite(filename: [*c]const u8, img: Mat) bool;
//*    pub extern fn Image_IMWrite_WithParams(filename: [*c]const u8, img: Mat, params: IntVector) bool;
//*    pub extern fn Image_IMEncode(fileExt: [*c]const u8, img: Mat, vector: ?*anyopaque) void;
//*    pub extern fn Image_IMEncode_WithParams(fileExt: [*c]const u8, img: Mat, params: IntVector, vector: ?*anyopaque) void;
//*    pub extern fn Image_IMDecode(buf: ByteArray, flags: c_int) Mat;
