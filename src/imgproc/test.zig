const std = @import("std");
const testing = std.testing;
const core = @import("../core.zig");
const imgcodecs = @import("../imgcodecs.zig");
const imgproc = @import("../imgproc.zig");
const Mat = core.Mat;
const Size = core.Size;
const Color = core.Color;
const Scalar = core.Scalar;
const Point = core.Point;
const Rect = core.Rect;
const PointVector = core.PointVector;

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

test "imgproc bilateral filter" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.bilateralFilter(img, &dst, 1, 2.0, 3.0);

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

    imgproc.sobel(img, &dst, .cv16sc1, 0, 1, 3, 1, 0, .{});

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

    imgproc.spatialGradient(img, &dx, &dy, .cv16sc1, .{});

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

    imgproc.dilateWithParams(img, &dst, kernel, Point.init(0, 0), .{ .type = .wrap }, .{}, Color{});

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

test "imgproc pyrdown" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.pyrDown(img, &dst, Size.init(dst.cols(), dst.rows()), .{});
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

    imgproc.pyrUp(img, &dst, Size.init(dst.cols(), dst.rows()), .{});
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expect(@fabs(@intToFloat(f64, (2 * img.cols() - dst.cols()))) < 2.0);
    try testing.expect(@fabs(@intToFloat(f64, (2 * img.rows() - dst.rows()))) < 2.0);
}

test "imgproc boxPoints" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();

    var thresh = try Mat.init();
    defer thresh.deinit();

    _ = imgproc.threshold(img, &thresh, 25, 255, .{ .type = .binary });

    var contours = try imgproc.findContours(thresh, .external, .simple);
    defer contours.deinit();

    var contour = try contours.at(0);

    var hull = try Mat.init();
    defer hull.deinit();

    imgproc.convexHull(try contours.at(0), &hull, false, false);

    var hull_points = std.ArrayList(Point).init(testing.allocator);
    defer hull_points.deinit();

    {
        var i: usize = 0;
        while (i < hull.cols()) : (i += 1) {
            var j: usize = 0;
            while (j < hull.rows()) : (j += 1) {
                const p = hull.get(i32, j, i);
                try hull_points.append(contour.at(p));
            }
        }
    }

    var pvhp = try PointVector.initFromPoints(hull_points.items, testing.allocator);
    defer pvhp.deinit();

    var rect = imgproc.minAreaRect(pvhp);
    var pts = try Mat.init();
    defer pts.deinit();

    imgproc.boxPoints(rect, &pts);

    try testing.expectEqual(false, pts.isEmpty());
    try testing.expectEqual(@as(i32, 4), pts.rows());
    try testing.expectEqual(@as(i32, 2), pts.cols());
}

test "imgproc areaRect" {
    comptime var src = [_]Point{
        Point.init(0, 2),
        Point.init(2, 0),
        Point.init(4, 2),
        Point.init(2, 4),
    };

    var pv = try PointVector.initFromPoints(src[0..], testing.allocator);
    defer pv.deinit();

    var m = imgproc.minAreaRect(pv);

    try testing.expectEqual(@as(i32, 2.0), m.center.x);
    try testing.expectEqual(@as(i32, 2.0), m.center.y);
    try testing.expectEqual(@as(f64, 45.0), m.angle);
}

test "imgproc fitEllipse" {
    comptime var src = [_]Point{
        Point.init(1, 1),
        Point.init(0, 1),
        Point.init(0, 2),
        Point.init(1, 3),
        Point.init(2, 3),
        Point.init(4, 2),
        Point.init(4, 1),
        Point.init(0, 3),
        Point.init(0, 2),
    };
    var pv = try PointVector.initFromPoints(src[0..], testing.allocator);
    defer pv.deinit();

    var rect = imgproc.fitEllipse(pv);

    try testing.expectEqual(@as(i32, 2.0), rect.center.x);
    try testing.expectEqual(@as(i32, 2.0), rect.center.y);
    try testing.expectEqual(@as(f64, 78.60807800292969), rect.angle);
}

test "imgproc findContours" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var res = try imgproc.findContours(img, .external, .simple);
    defer res.deinit();

    try testing.expect(res.size() > 0);

    var r0 = try res.at(0);
    var area = imgproc.contourArea(r0);
    try testing.expectEqual(@as(f64, 127280.0), area);

    var r = imgproc.boundingRect(r0);
    try testing.expectEqual(@as(i32, 0), r.x);
    try testing.expectEqual(@as(i32, 0), r.y);
    try testing.expectEqual(@as(i32, 400), r.width);
    try testing.expectEqual(@as(i32, 320), r.height);

    var length = imgproc.arcLength(r0, true);
    try testing.expectEqual(@as(i32, 1436), @floatToInt(i32, length));

    length = imgproc.arcLength(r0, false);
    try testing.expectEqual(@as(i32, 1037), @floatToInt(i32, length));
}

test "imgproc findContoursWithParams" {
    var img = try imgcodecs.imRead(img_dir ++ "contours.png", .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var hierarchy = try Mat.init();
    defer hierarchy.deinit();

    var res = try imgproc.findContoursWithParams(img, &hierarchy, .tree, .none);
    defer res.deinit();

    try testing.expectEqual(@as(i32, 4), res.size());
    try testing.expectEqual(hierarchy.cols(), res.size());
}

test "imgproc polygonTest" {
    comptime var tests = [_]struct {
        name: []const u8, // name of the testcase
        thickness: i32, // thickness of the polygon
        point: Point, // point to be checked
        result: f64, // expected result; either distance or -1, 0, 1 based on measure parameter
        measure: bool, // enable distance measurement, if true
    }{
        .{ .name = "Outside the polygon - measure=false", .thickness = 1, .point = Point.init(5, 15), .result = -1.0, .measure = false },
        .{ .name = "On the polygon - measure=false", .thickness = 1, .point = Point.init(10, 10), .result = 0.0, .measure = false },
        .{ .name = "Inside the polygon - measure=true", .thickness = 1, .point = Point.init(20, 30), .result = 10.0, .measure = true },
        .{ .name = "Outside the polygon - measure=true", .thickness = 1, .point = Point.init(5, 15), .result = -5.0, .measure = true },
        .{ .name = "On the polygon - measure=true", .thickness = 1, .point = Point.init(10, 10), .result = 0.0, .measure = true },
    };

    comptime var pts = [_]Point{
        Point.init(10, 10),
        Point.init(10, 80),
        Point.init(80, 80),
        Point.init(80, 10),
    };

    var ctr = try PointVector.initFromPoints(pts[0..], testing.allocator);
    defer ctr.deinit();

    for (tests) |t| {
        var result = imgproc.pointPolygonTest(ctr, t.point, t.measure);
        testing.expectEqual(t.result, result) catch |e| {
            std.debug.print("Testcase: {s}\n", .{t.name});
            return e;
        };
    }
}

test "imgproc connectedComponents" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var labels = try Mat.init();
    defer labels.deinit();

    var res = imgproc.connectedComponents(img, &labels);
    try testing.expect(res > 0);
    try testing.expectEqual(false, labels.isEmpty());
}

test "imgproc connectedComponentsWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var labels = try Mat.init();
    defer labels.deinit();

    var res = imgproc.connectedComponentsWithParams(img, &labels, 8, .cv32sc1, .default);
    try testing.expect(res > 0);
    try testing.expectEqual(false, labels.isEmpty());
}

test "imgproc connectedComponentsWithStats" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var labels = try Mat.init();
    defer labels.deinit();

    var stats = try Mat.init();
    defer stats.deinit();

    var centroids = try Mat.init();
    defer centroids.deinit();

    var res = imgproc.connectedComponentsWithStats(img, &labels, &stats, &centroids);
    try testing.expect(res > 0);
    try testing.expectEqual(false, labels.isEmpty());
    try testing.expectEqual(false, stats.isEmpty());
    try testing.expectEqual(false, centroids.isEmpty());
}

test "imgproc connectedComponentsWithStatsWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var labels = try Mat.init();
    defer labels.deinit();

    var stats = try Mat.init();
    defer stats.deinit();

    var centroids = try Mat.init();
    defer centroids.deinit();

    var res = imgproc.connectedComponentsWithStatsWithParams(img, &labels, &stats, &centroids, 8, .cv32sc1, .default);
    try testing.expect(res > 0);
    try testing.expectEqual(false, labels.isEmpty());
    try testing.expectEqual(false, stats.isEmpty());
    try testing.expectEqual(false, centroids.isEmpty());
}

test "imgproc erode" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dest = try Mat.init();
    defer dest.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.erode(img, &dest, kernel);
    try testing.expectEqual(false, dest.isEmpty());
    try testing.expectEqual(img.rows(), dest.rows());
    try testing.expectEqual(img.cols(), dest.cols());
}

test "imgproc erodeWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dest = try Mat.init();
    defer dest.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.erodeWithParams(img, &dest, kernel, Point.init(-1, -1), 3, 0);
    try testing.expectEqual(false, dest.isEmpty());
    try testing.expectEqual(img.rows(), dest.rows());
    try testing.expectEqual(img.cols(), dest.cols());
}

test "imgproc morphologyDefaultBorderValue" {
    var morphologyDefaultBorderValue = imgproc.morphologyDefaultBorderValue();
    for (morphologyDefaultBorderValue.toArray()) |s| {
        try testing.expect(s != 0);
    }
}

test "imgproc morphologyEx" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dest = try Mat.init();
    defer dest.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.morphologyEx(img, &dest, .open, kernel);
    try testing.expectEqual(false, dest.isEmpty());
    try testing.expectEqual(img.rows(), dest.rows());
    try testing.expectEqual(img.cols(), dest.cols());
}

test "imgproc morphologyExWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dest = try Mat.init();
    defer dest.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.morphologyExWithParams(img, &dest, .open, kernel, 2, .{ .type = .constant });
    try testing.expectEqual(false, dest.isEmpty());
    try testing.expectEqual(img.rows(), dest.rows());
    try testing.expectEqual(img.cols(), dest.cols());
}

test "imgproc gaussianBlur" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.gaussianBlur(img, &dst, Size.init(23, 23), 30, 50, .{});
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc getGaussianKernel" {
    var kernel = try imgproc.getGaussianKernel(1, 0.5);
    defer kernel.deinit();
    try testing.expectEqual(false, kernel.isEmpty());
}

test "imgproc getGaussianKernelWithParams" {
    var kernel = try imgproc.getGaussianKernelWithParams(1, 0.5, .cv64fc1);
    defer kernel.deinit();
    try testing.expectEqual(false, kernel.isEmpty());
}

test "imgproc laplacian" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.laplacian(img, &dst, .cv16sc1, 1, 1, 0, .{});
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc scharr" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.scharr(img, &dst, .cv16sc1, 1, 0, 0, 0, .{});
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc medianBlur" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.medianBlur(img, &dst, 3);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc canny" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.canny(img, &dst, 50, 150);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
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
        Size.init(10, 10),
        Size.init(-1, -1),
        tc,
    );
    try testing.expectEqual(false, corners.isEmpty());
    try testing.expectEqual(@as(i32, 205), corners.rows());
    try testing.expectEqual(@as(i32, 1), corners.cols());
}

test "imgproc grabcut" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var src = try Mat.init();
    defer src.deinit();
    imgproc.cvtColor(img, &img, .rgba_to_bgr);
    img.convertTo(&src, .cv8uc3);

    var mask = try Mat.initSize(img.rows(), img.cols(), .cv8uc1);
    defer mask.deinit();

    var bgd_model = try Mat.init();
    defer bgd_model.deinit();
    var fgd_model = try Mat.init();
    defer fgd_model.deinit();

    var r = Rect.init(0, 0, 50, 50);

    imgproc.grabCut(src, &mask, r, &bgd_model, &fgd_model, 1, .eval);
    try testing.expectEqual(false, bgd_model.isEmpty());
    try testing.expectEqual(false, fgd_model.isEmpty());
}

test "imgproc houghCircles" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var circles = try Mat.init();
    defer circles.deinit();

    imgproc.houghCircles(img, &circles, .gradient, 5.0, 5.0);
    try testing.expectEqual(false, circles.isEmpty());
    try testing.expectEqual(@as(i32, 1), circles.rows());
    try testing.expectEqual(@as(i32, 317), circles.cols());
}

test "imgproc HoughCirclesWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var circles = try Mat.init();
    defer circles.deinit();

    imgproc.houghCirclesWithParams(img, &circles, .gradient, 5.0, 5.0, 100, 100, 0, 0);

    try testing.expectEqual(false, circles.isEmpty());
    try testing.expectEqual(@as(i32, 1), circles.rows());
    try testing.expect(circles.cols() >= 317);
}

test "imgproc houghLines" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.houghLines(img, &dst, 1, std.math.pi / 180.0, 50);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 6465), dst.rows());
    try testing.expectEqual(@as(i32, 1), dst.cols());

    try testing.expectEqual(@as(f32, 226), dst.get(f32, 0, 0));
    try testing.expectEqual(@as(f32, 0.7853982), dst.get(f32, 0, 1));

    try testing.expectEqual(@as(f32, 228), dst.get(f32, 1, 0));
    try testing.expectEqual(@as(f32, 0.7853982), dst.get(f32, 1, 1));

    try testing.expectEqual(@as(f32, 23), dst.get(f32, 6463, 0));
    try testing.expectEqual(@as(f32, 0.75049156), dst.get(f32, 6463, 1));

    try testing.expectEqual(@as(f32, 23), dst.get(f32, 6464, 0));
    try testing.expectEqual(@as(f32, 0.82030475), dst.get(f32, 6464, 1));
}

test "imgproc houghLinesP" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.houghLinesP(img, &dst, 1, std.math.pi / 180.0, 50);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 4356), dst.rows());
    try testing.expectEqual(@as(i32, 1), dst.cols());

    try testing.expectEqual(@as(i32, 46), dst.get(i32, 0, 0));
    try testing.expectEqual(@as(i32, 0), dst.get(i32, 0, 1));
    try testing.expectEqual(@as(i32, 365), dst.get(i32, 0, 2));
    try testing.expectEqual(@as(i32, 319), dst.get(i32, 0, 3));

    try testing.expectEqual(@as(i32, 86), dst.get(i32, 1, 0));
    try testing.expectEqual(@as(i32, 319), dst.get(i32, 1, 1));
    try testing.expectEqual(@as(i32, 399), dst.get(i32, 1, 2));
    try testing.expectEqual(@as(i32, 6), dst.get(i32, 1, 3));

    try testing.expectEqual(@as(i32, 96), dst.get(i32, 433, 0));
    try testing.expectEqual(@as(i32, 319), dst.get(i32, 433, 1));
    try testing.expectEqual(@as(i32, 108), dst.get(i32, 433, 2));
    try testing.expectEqual(@as(i32, 316), dst.get(i32, 433, 3));

    try testing.expectEqual(@as(i32, 39), dst.get(i32, 434, 0));
    try testing.expectEqual(@as(i32, 280), dst.get(i32, 434, 1));
    try testing.expectEqual(@as(i32, 89), dst.get(i32, 434, 2));
    try testing.expectEqual(@as(i32, 227), dst.get(i32, 434, 3));
}

test "imgproc houghLinesPWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.houghLinesPWithParams(img, &dst, 1, std.math.pi / 180.0, 50, 1, 1);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 514), dst.rows());
    try testing.expectEqual(@as(i32, 1), dst.cols());

    try testing.expectEqual(@as(i32, 46), dst.get(i32, 0, 0));
    try testing.expectEqual(@as(i32, 0), dst.get(i32, 0, 1));
    try testing.expectEqual(@as(i32, 365), dst.get(i32, 0, 2));
    try testing.expectEqual(@as(i32, 319), dst.get(i32, 0, 3));

    try testing.expectEqual(@as(i32, 86), dst.get(i32, 1, 0));
    try testing.expectEqual(@as(i32, 319), dst.get(i32, 1, 1));
    try testing.expectEqual(@as(i32, 399), dst.get(i32, 1, 2));
    try testing.expectEqual(@as(i32, 6), dst.get(i32, 1, 3));

    try testing.expectEqual(@as(i32, 0), dst.get(i32, 433, 0));
    try testing.expectEqual(@as(i32, 126), dst.get(i32, 433, 1));
    try testing.expectEqual(@as(i32, 71), dst.get(i32, 433, 2));
    try testing.expectEqual(@as(i32, 57), dst.get(i32, 433, 3));

    try testing.expectEqual(@as(i32, 309), dst.get(i32, 434, 0));
    try testing.expectEqual(@as(i32, 319), dst.get(i32, 434, 1));
    try testing.expectEqual(@as(i32, 399), dst.get(i32, 434, 2));
    try testing.expectEqual(@as(i32, 229), dst.get(i32, 434, 3));
}

test "imgproc houghLinesPointSet" {
    comptime var points = [_][2]f32{
        .{ 0, 369 },   .{ 10, 364 },  .{ 20, 358 },  .{ 30, 352 },
        .{ 40, 346 },  .{ 50, 341 },  .{ 60, 335 },  .{ 70, 329 },
        .{ 80, 323 },  .{ 90, 318 },  .{ 100, 312 }, .{ 110, 306 },
        .{ 120, 300 }, .{ 130, 295 }, .{ 140, 289 }, .{ 150, 284 },
        .{ 160, 277 }, .{ 170, 271 }, .{ 180, 266 }, .{ 190, 260 },
    };

    var img = try Mat.initSize(points.len, 1, .cv32fc2);
    defer img.deinit();

    for (points) |p, i| {
        img.set(f32, i, 0, p[0]);
        img.set(f32, i, 1, p[1]);
    }

    var dst = try Mat.init();
    defer dst.deinit();

    const rho_min = 0.0;
    const rho_max = 360.0;
    const rho_step = 1.0;
    const theta_min = 0.0;
    const theta_max = std.math.pi / 2.0;
    const theta_step = std.math.pi / 180.0;

    imgproc.houghLinesPointSet(
        img,
        &dst,
        20,
        1,
        rho_min,
        rho_max,
        rho_step,
        theta_min,
        theta_max,
        theta_step,
    );

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 20), dst.rows());
    try testing.expectEqual(@as(i32, 1), dst.cols());
    try testing.expectEqual(@as(f64, 19), dst.get(f64, 0, 0));
    try testing.expectEqual(@as(f64, 320), dst.get(f64, 0, 1));
    try testing.expectEqual(@as(f64, 1.0471975803375244), dst.get(f64, 0, 2));

    try testing.expectEqual(@as(f64, 7), dst.get(f64, 1, 0));
    try testing.expectEqual(@as(f64, 321), dst.get(f64, 1, 1));
    try testing.expectEqual(@as(f64, 1.0646508932113647), dst.get(f64, 1, 2));

    try testing.expectEqual(@as(f64, 2), dst.get(f64, 18, 0));
    try testing.expectEqual(@as(f64, 317), dst.get(f64, 18, 1));
    try testing.expectEqual(@as(f64, 1.1344640254974365), dst.get(f64, 18, 2));

    try testing.expectEqual(@as(f64, 2), dst.get(f64, 19, 0));
    try testing.expectEqual(@as(f64, 330), dst.get(f64, 19, 1));
    try testing.expectEqual(@as(f64, 1.1344640254974365), dst.get(f64, 19, 2));
}

test "imgproc integral" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var sum = try Mat.init();
    defer sum.deinit();
    var sqsum = try Mat.init();
    defer sqsum.deinit();
    var tilted = try Mat.init();
    defer tilted.deinit();

    imgproc.integral(img, &sum, &sqsum, &tilted);
    try testing.expectEqual(false, sum.isEmpty());
    try testing.expectEqual(false, sqsum.isEmpty());
    try testing.expectEqual(false, tilted.isEmpty());
}

test "imgproc threshold" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    _ = imgproc.threshold(img, &dst, 100, 255, .{ .type = .binary });
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc adaptiveThreshold" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    _ = imgproc.adaptiveThreshold(img, &dst, 255, .gaussian, .{ .type = .binary }, 11, 2);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc putText" {
    var img = try Mat.initSize(150, 150, .cv8uc1);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    imgproc.putText(&img, "Testing", Point.init(10, 10), .{ .type = .plain }, 1.2, Color.init(0, 0, 255, 0), 2);

    try testing.expectEqual(false, img.isEmpty());
}

test "imgproc putTextWithParams" {
    var img = try Mat.initSize(150, 150, .cv8uc1);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    imgproc.putTextWithParams(&img, "Testing", Point.init(10, 10), .{ .type = .plain }, 1.2, Color.init(0, 0, 255, 0), 2, .line_aa, false);

    try testing.expectEqual(false, img.isEmpty());
}

test "imgproc resize" {
    var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
    defer img.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.resize(img, &dst, Size.init(0, 0), 0.5, 0.5, .{});
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 172), dst.rows());
    try testing.expectEqual(@as(i32, 200), dst.cols());

    imgproc.resize(img, &dst, Size.init(440, 377), 0, 0, .{});
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

    imgproc.remap(img, &dst, map1, map2, .{}, .{ .type = .constant }, Color{});
    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc textSize" {
    const size = imgproc.getTextSize("test", .{ .type = .simplex }, 1.2, 1);

    try testing.expectEqual(@as(i32, 72), size.width);
    try testing.expectEqual(@as(i32, 26), size.height);

    const res = imgproc.getTextSizeWithBaseline("text", .{ .type = .simplex }, 1.2, 1);

    try testing.expectEqual(@as(i32, 72), res.size.width);
    try testing.expectEqual(@as(i32, 26), res.size.height);
    try testing.expectEqual(@as(i32, 11), res.baseline);
}
