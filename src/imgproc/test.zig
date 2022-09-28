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
const face_detect_filepath = img_dir ++ "face-detect.jpg";

test "imgproc CLAHE" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
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
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
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
    imgproc.line(&img, Point.init(25, 75), Point.init(75, 50), white, 1);
    imgproc.line(&img, Point.init(75, 50), Point.init(25, 25), white, 1);
    // Draw rectangle
    imgproc.rectangle(&img, Rect.init(125, 25, 50, 50), white, 1);

    var contours = try imgproc.findContours(img, .external, .simple);
    defer contours.deinit();

    var c0 = try contours.at(0);
    var triangle_perimeter = imgproc.arcLength(c0, true);
    var trianle_contour = try imgproc.approxPolyDP(c0, 0.04 * triangle_perimeter, true);
    defer trianle_contour.deinit();

    var expected_triangle_contour = [_]Point{
        Point.init(25, 25),
        Point.init(25, 75),
        Point.init(75, 50),
    };

    var actual_triangle_contour = try trianle_contour.toPoints(testing.allocator);
    defer actual_triangle_contour.deinit();

    try testing.expectEqual(expected_triangle_contour.len, actual_triangle_contour.items.len);
    for (expected_triangle_contour) |expected_point, i| {
        try testing.expectEqual(expected_point, actual_triangle_contour.items[i]);
    }

    var c1 = try contours.at(1);
    var rectangle_perimeter = imgproc.arcLength(c1, true);
    var rectangle_contour = try imgproc.approxPolyDP(c1, 0.04 * rectangle_perimeter, true);
    defer rectangle_contour.deinit();

    var expected_rectangle_contour = [_]Point{
        Point.init(125, 24),
        Point.init(124, 75),
        Point.init(175, 76),
        Point.init(176, 25),
    };

    var actual_rectangle_contour = try rectangle_contour.toPoints(testing.allocator);
    defer actual_rectangle_contour.deinit();

    try testing.expectEqual(expected_rectangle_contour.len, actual_rectangle_contour.items.len);
    for (expected_rectangle_contour) |expected_point, i| {
        try testing.expectEqual(expected_point, actual_rectangle_contour.items[i]);
    }
}

test "imgproc convexity" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var res = try imgproc.findContours(img, .external, .simple);
    defer res.deinit();
    try testing.expect(1 <= res.size());

    var area = imgproc.contourArea(try res.at(0));
    try testing.expectEqual(@as(f64, 127280), area);

    var hull = try Mat.init();
    defer hull.deinit();

    imgproc.convexHull(try res.at(0), &hull, true, false);
    try testing.expectEqual(false, hull.isEmpty());

    var defects = try Mat.init();
    defer defects.deinit();

    imgproc.convexityDefects(try res.at(0), hull, &defects);
    try testing.expectEqual(false, defects.isEmpty());
}

test "imgproc min enclosing circle" {
    const epsilon = 0.001;
    const expected_radius = 2.0;
    const expected_x = 0.0;
    const expected_y = 0.0;
    const pts = [_]Point{
        Point.init(0, 2),
        Point.init(2, 0),
        Point.init(0, -2),
        Point.init(-2, 0),
        Point.init(1, -1),
    };

    var pv = try core.PointVector.initFromPoints(pts[0..], testing.allocator);
    defer pv.deinit();

    const res = imgproc.minEnclosingCircle(pv);
    const radius = res.radius;
    const point = res.point;

    try testing.expect(@fabs(@as(f64, radius) - expected_radius) <= epsilon);
    try testing.expect(@fabs(@as(f64, point.x) - expected_x) <= epsilon);
    try testing.expect(@fabs(@as(f64, point.y) - expected_y) <= epsilon);
}

test "imgproc cvtColor" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.cvtColor(img, &dst, .bgra_to_gray);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc blur" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.blur(img, &dst, Size.init(3, 3));

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc sobel" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.sobel(img, &dst, .cv16sc1, 0, 1, 3, 1, 0, imgproc.BorderType.default);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc spatialGradient" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dx = try Mat.init();
    defer dx.deinit();

    var dy = try Mat.init();
    defer dy.deinit();

    imgproc.spatialGradient(img, &dx, &dy, .cv16sc1, imgproc.BorderType.default);

    try testing.expectEqual(false, dx.isEmpty());
    try testing.expectEqual(img.rows(), dx.rows());
    try testing.expectEqual(img.cols(), dx.cols());
    try testing.expectEqual(false, dy.isEmpty());
    try testing.expectEqual(img.rows(), dy.rows());
    try testing.expectEqual(img.cols(), dy.cols());
}

test "imgproc boxfilter" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.boxFilter(img, &dst, -1, Size.init(3, 3));

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc sqBoxfilter" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.sqBoxFilter(img, &dst, -1, Size.init(3, 3));

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc dilate" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.dilate(img, &dst, kernel);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc dilateWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.dilateWithParams(img, &dst, kernel, Point.init(0, 0), .wrap, imgproc.BorderType.default, Color{});

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc distanceTransform" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var gray = try Mat.init();
    defer gray.deinit();
    imgproc.cvtColor(img, &gray, .bgra_to_gray);

    var thresh = try Mat.init();
    defer thresh.deinit();
    _ = imgproc.threshold(gray, &thresh, 25, 255, .{ .type = .binary });

    var dst = try Mat.init();
    defer dst.deinit();

    var labels = try Mat.init();
    defer labels.deinit();

    imgproc.distanceTransform(thresh, &dst, &labels, .l2, .mask_3, .c_comp);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc matchTemplate" {
    var img_scene = try imgcodecs.imRead(img_dir ++ "face.jpg", .gray_scale);
    defer img_scene.deinit();
    try testing.expectEqual(false, img_scene.isEmpty());

    var img_template = try imgcodecs.imRead(img_dir ++ "toy.jpg", .gray_scale);
    defer img_template.deinit();
    try testing.expectEqual(false, img_template.isEmpty());

    var result = try Mat.init();
    defer result.deinit();

    var m = try Mat.init();
    imgproc.matchTemplate(img_scene, img_template, &result, .ccoeff_normed, m);
    m.deinit();

    const res = Mat.minMaxLoc(result);
    const max_confidence = res.max_val;
    try testing.expect(max_confidence >= 0.95);
}

test "imgproc moments" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var result = imgproc.moments(img, true);
    _ = result;
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

test "imgproc pyrdown" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.pyrDown(img, &dst, Size.init(dst.cols(), dst.rows()), imgproc.BorderType.default);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expect(@fabs(@intToFloat(f64, (img.cols() - 2 * dst.cols()))) < 2.0);
    try testing.expect(@fabs(@intToFloat(f64, (img.rows() - 2 * dst.rows()))) < 2.0);
}

test "imgproc pyrup" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.pyrUp(img, &dst, Size.init(dst.cols(), dst.rows()), imgproc.BorderType.default);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expect(@fabs(@intToFloat(f64, (2 * img.cols() - dst.cols()))) < 2.0);
    try testing.expect(@fabs(@intToFloat(f64, (2 * img.rows() - dst.rows()))) < 2.0);
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
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
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
