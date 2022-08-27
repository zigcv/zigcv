const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const castToC = utils.castZigU8ToC;

const Mat = core.Mat;

pub const IMReadFlag = enum(i32) {
    // https://docs.opencv.org/4.6.0/d6/d87/imgcodecs_8hpp.html
    // IMReadUnchanged return the loaded image as is (with alpha channel,
    // otherwise it gets cropped).
    IMReadUnchanged = -1,

    // IMReadGrayScale always convert image to the single channel
    // grayscale image.
    IMReadGrayScale = 0,

    // IMReadColor always converts image to the 3 channel BGR color image.
    IMReadColor = 1,

    // IMReadAnyDepth returns 16-bit/32-bit image when the input has the corresponding
    // depth, otherwise convert it to 8-bit.
    IMReadAnyDepth = 2,

    // IMReadAnyColor the image is read in any possible color format.
    IMReadAnyColor = 4,

    // IMReadLoadGDAL uses the gdal driver for loading the image.
    IMReadLoadGDAL = 8,

    // IMReadReducedGrayscale2 always converts image to the single channel grayscale image
    // and the image size reduced 1/2.
    IMReadReducedGrayscale2 = 16,

    // IMReadReducedColor2 always converts image to the 3 channel BGR color image and the
    // image size reduced 1/2.
    IMReadReducedColor2 = 17,

    // IMReadReducedGrayscale4 always converts image to the single channel grayscale image and
    // the image size reduced 1/4.
    IMReadReducedGrayscale4 = 32,

    // IMReadReducedColor4 always converts image to the 3 channel BGR color image and
    // the image size reduced 1/4.
    IMReadReducedColor4 = 33,

    // IMReadReducedGrayscale8 always convert image to the single channel grayscale image and
    // the image size reduced 1/8.
    IMReadReducedGrayscale8 = 64,

    // IMReadReducedColor8 always convert image to the 3 channel BGR color image and the
    // image size reduced 1/8.
    IMReadReducedColor8 = 65,

    // IMReadIgnoreOrientation do not rotate the image according to EXIF's orientation flag.
    IMReadIgnoreOrientation = 128,
};

pub const IMWriteFlag = enum(i32) {
    //IMWriteJpegQuality is the quality from 0 to 100 for JPEG (the higher is the better). Default value is 95.
    IMWriteJpegQuality = 1,

    // IMWriteJpegProgressive enables JPEG progressive feature, 0 or 1, default is False.
    IMWriteJpegProgressive = 2,

    // IMWriteJpegOptimize enables JPEG optimization, 0 or 1, default is False.
    IMWriteJpegOptimize = 3,

    // IMWriteJpegRstInterval is the JPEG restart interval, 0 - 65535, default is 0 - no restart.
    IMWriteJpegRstInterval = 4,

    // IMWriteJpegLumaQuality separates luma quality level, 0 - 100, default is 0 - don't use.
    IMWriteJpegLumaQuality = 5,

    // IMWriteJpegChromaQuality separates chroma quality level, 0 - 100, default is 0 - don't use.
    IMWriteJpegChromaQuality = 6,

    // IMWritePngCompression is the compression level from 0 to 9 for PNG. A
    // higher value means a smaller size and longer compression time.
    // If specified, strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT (Z_DEFAULT_STRATEGY).
    // Default value is 1 (best speed setting).
    IMWritePngCompression = 16,

    // IMWritePngStrategy is one of cv::IMWritePNGFlags, default is IMWRITE_PNG_STRATEGY_RLE.
    IMWritePngStrategy = 17,

    // IMWritePngBilevel is the binary level PNG, 0 or 1, default is 0.
    IMWritePngBilevel = 18,

    // IMWritePxmBinary for PPM, PGM, or PBM can be a binary format flag, 0 or 1. Default value is 1.
    IMWritePxmBinary = 32,

    // IMWriteWebpQuality is the quality from 1 to 100 for WEBP (the higher is
    // the better). By default (without any parameter) and for quality above
    // 100 the lossless compression is used.
    IMWriteWebpQuality = 64,

    // IMWritePamTupletype sets the TUPLETYPE field to the corresponding string
    // value that is defined for the format.
    IMWritePamTupletype = 128,

    // IMWritePngStrategyDefault is the value to use for normal data.
    IMWritePngStrategyDefault = 0,

    // IMWritePngStrategyFiltered is the value to use for data produced by a
    // filter (or predictor). Filtered data consists mostly of small values
    // with a somewhat random distribution. In this case, the compression
    // algorithm is tuned to compress them better.
    IMWritePngStrategyFiltered = 1,

    // IMWritePngStrategyHuffmanOnly forces Huffman encoding only (no string match).
    IMWritePngStrategyHuffmanOnly = 2,

    // IMWritePngStrategyRle is the value to use to limit match distances to
    // one (run-length encoding).
    IMWritePngStrategyRle = 3,

    // IMWritePngStrategyFixed is the value to prevent the use of dynamic
    // Huffman codes, allowing for a simpler decoder for special applications.
    IMWritePngStrategyFixed = 4,
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
    return try Mat.initFromC(cMat);
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
    comptime var int_params: [params.len * 2]i32 = undefined;
    for (params) |p, i| {
        int_params[2 * i] = @enumToInt(p.f);
        int_params[2 * i + 1] = @enumToInt(p.v);
    }
    const c_params = c.IntVector{
        .val = @ptrCast([*]c_int, int_params),
        .len = params.len,
    };
    const result = c.Image_IMWrite_WithParams(castToC(filename), img.ptr, c_params);
    if (!result) {
        return error.IMWriteFailed;
    }
}

pub fn imDecode(buf: []u8, flags: IMReadFlag) !Mat {
    var data = @ptrCast([*]u8, buf);
    return try Mat.initFromC(c.Image_IMDecode(data, @enumToInt(flags)));
}

//*    implementation done
//*    pub extern fn Image_IMRead(filename: [*c]const u8, flags: c_int) Mat;
//*    pub extern fn Image_IMWrite(filename: [*c]const u8, img: Mat) bool;
//*    pub extern fn Image_IMWrite_WithParams(filename: [*c]const u8, img: Mat, params: IntVector) bool;
//     pub extern fn Image_IMEncode(fileExt: [*c]const u8, img: Mat, vector: ?*anyopaque) void;
//     pub extern fn Image_IMEncode_WithParams(fileExt: [*c]const u8, img: Mat, params: IntVector, vector: ?*anyopaque) void;
//*    pub extern fn Image_IMDecode(buf: ByteArray, flags: c_int) Mat;
