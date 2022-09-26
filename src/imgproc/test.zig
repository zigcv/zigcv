const std = @import("std");
const testing = std.testing;
const core = @import("../core.zig");
const imgcodecs = @import("../imgcodecs.zig");
const imgproc = @import("../imgproc.zig");
const Mat = core.Mat;
const Size = core.Size;
const Color = core.Color;
const Point = core.Point;
const Rect = core.Rect;

const img_dir = "./libs/gocv/images/";
const face_filepath = img_dir ++ "face-detect.jpg";

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

test "imgproc approxPolyDP" {
    var img = try Mat.initSize(100, 200, .cv8uc1);
    defer img.deinit();

    const white = Color.init(255, 255, 255, 255);

    // Draw a triangle
    imgproc.line(&img, Point.init(25, 25), Point.init(25, 75), white, 1);
    imgproc.line(&img, Point.init(25, 75), Point.init(25, 75), white, 1);
    imgproc.line(&img, Point.init(75, 50), Point.init(25, 75), white, 1);
    imgproc.rectangle(&img, Rect.init(125, 25, 175, 75), white, 1);
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

test "imgproc textSize" {
    const size = imgproc.getTextSize("test", .simplex, 1.2, 1);

    try testing.expectEqual(@as(i32, 72), size.width);
    try testing.expectEqual(@as(i32, 26), size.height);

    const res = imgproc.getTextSizeWithBaseline("text", .simplex, 1.2, 1);

    try testing.expectEqual(@as(i32, 72), res.size.width);
    try testing.expectEqual(@as(i32, 26), res.size.height);
    try testing.expectEqual(@as(i32, 11), res.baseline);
}

test "imgproc putText" {
    var img = try Mat.initSize(150, 150, .cv8uc1);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    imgproc.putText(&img, "Testing", Point.init(10, 10), .plain, 1.2, Color.init(0, 0, 255, 0), 2);

    try testing.expectEqual(false, img.isEmpty());
}

test "imgproc putTextWithParams" {
    var img = try Mat.initSize(150, 150, .cv8uc1);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    imgproc.putTextWithParams(&img, "Testing", Point.init(10, 10), .plain, 1.2, Color.init(0, 0, 255, 0), 2, .line_aa, false);

    try testing.expectEqual(false, img.isEmpty());
}

test "imgproc resize" {
    var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
    defer img.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.resize(img, &dst, Size.init(0, 0), 0.5, 0.5, imgproc.InterpolationFlag.default);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 172), dst.rows());
    try testing.expectEqual(@as(i32, 200), dst.cols());

    imgproc.resize(img, &dst, Size.init(440, 377), 0, 0, imgproc.InterpolationFlag.default);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 377), dst.rows());
    try testing.expectEqual(@as(i32, 440), dst.cols());
}

test "imgproc remap" {
    var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
    defer img.deinit();

    var dst = try Mat.init();
    defer dst.deinit();
    try testing.expectEqual(true, dst.isEmpty());

    var map1 = try Mat.initSize(256, 256, .cv32fc2);
    defer map1.deinit();
    map1.set(f32, 50, 50, 25.4);
    var map2 = try Mat.init();
    defer map2.deinit();

    imgproc.remap(img, &dst, map1, map2, imgproc.InterpolationFlag.default, .constant, Color.init(0, 0, 0, 0));
    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc goodFeaturesToTrack & CornerSubPix" {
    var img = try imgcodecs.imRead(face_filepath, .gray_scale);
    defer img.deinit();

    var corners = try Mat.init();
    defer corners.deinit();

    imgproc.goodFeaturesToTrack(img, &corners, 500, 0.01, 10);
    try testing.expectEqual(false, corners.isEmpty());
    try testing.expectEqual(@as(i32, 205), corners.rows());
    try testing.expectEqual(@as(i32, 1), corners.cols());

    var tc = try core.TermCriteria.init(.{ .count = true, .eps = true }, 20, 0.03);
    defer tc.deinit();

    imgproc.cornerSubPix(
        img,
        &corners,
        core.Size.init(10, 10),
        core.Size.init(-1, -1),
        tc,
    );
    try testing.expectEqual(false, corners.isEmpty());
    try testing.expectEqual(@as(i32, 205), corners.rows());
    try testing.expectEqual(@as(i32, 1), corners.cols());
}
