const std = @import("std");
const testing = std.testing;
const core = @import("../core.zig");
const imgcodecs = @import("../imgcodecs.zig");
const imgproc = @import("../imgproc.zig");
const Mat = core.Mat;
const Size = core.Size;

const face_filepath = "./libs/gocv/images/face-detect.jpg";

test "imgproc CLAHE" {
    var img = try imgcodecs.imRead(face_filepath, .gray_scale);
    defer img.deinit();

    var src = try Mat.init();
    defer src.deinit();

    img.convertTo(&src, .cv8uc1);

    var dst = try Mat.init();
    defer dst.deinit();

    var c = try imgproc.CLAHE.init();
    defer c.deinit();

    c.apply(src, &dst);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc CLAHE WithParams" {
    var img = try imgcodecs.imRead(face_filepath, .gray_scale);
    defer img.deinit();

    var src = try Mat.init();
    defer src.deinit();

    img.convertTo(&src, .cv8uc1);

    var dst = try Mat.init();
    defer dst.deinit();

    var c = try imgproc.CLAHE.initWithParams(2, Size.init(10, 10));
    defer c.deinit();

    c.apply(src, &dst);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc cvtColor" {
    var img = try imgcodecs.imRead(face_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.cvtColor(img, &dst, .bgra_to_gray);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}
