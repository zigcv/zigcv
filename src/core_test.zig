const std = @import("std");
const testing = std.testing;
const allocator = std.testing.allocator;
const core = @import("core.zig");
const Mat = core.Mat;

test "mat" {
    var mat = try Mat.init();
    defer mat.deinit();
    try testing.expectEqual(true, mat.isEmpty());
}

test "mat size" {
    const size: usize = 10;
    var mat = try Mat.initSize(10, 10, .cv8uc1);
    defer mat.deinit();
    const mat_size = try mat.size(allocator);
    defer mat_size.deinit();
    try testing.expectEqual(size, mat_size.items[0]);
    try testing.expectEqual(size, mat_size.items[1]);
    try testing.expectEqual(@as(usize, 2), mat_size.items.len);

    try testing.expectEqual(size, mat.rows());
    try testing.expectEqual(size, mat.cols());
}

test "mat channnel" {
    var mat = try Mat.initSize(1, 1, .cv8uc1);
    defer mat.deinit();

    try testing.expectEqual(@as(usize, 1), mat.channels());
}

test "mat type" {
    var mat = try Mat.initSize(1, 1, .cv8uc1);
    defer mat.deinit();

    try testing.expectEqual(core.Mat.MatType.cv8uc1, mat.getType());

    var mat2 = try Mat.initSize(1, 1, .cv16sc2);
    defer mat2.deinit();

    try testing.expectEqual(Mat.MatType.cv16sc2, mat2.getType());
}

test "mat eye" {
    var mat = try Mat.initEye(3, 3, .cv8sc1);
    defer mat.deinit();
    {
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            var j: usize = 0;
            while (j < 3) : (j += 1) {
                if (i == j) {
                    try testing.expectEqual(@as(u8, 1), mat.at(u8, i, j));
                } else {
                    try testing.expectEqual(@as(u8, 0), mat.at(u8, i, j));
                }
            }
        }
    }
}

test "mat zeros" {
    var mat = try Mat.initZeros(3, 3, .cv8sc1);
    defer mat.deinit();
    {
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            var j: usize = 0;
            while (j < 3) : (j += 1) {
                try testing.expectEqual(@as(u8, 0), mat.at(u8, i, j));
            }
        }
    }
}

test "mat ones" {
    var mat = try Mat.initOnes(3, 3, .cv8sc1);
    defer mat.deinit();
    {
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            var j: usize = 0;
            while (j < 3) : (j += 1) {
                try testing.expectEqual(@as(u8, 1), mat.at(u8, i, j));
            }
        }
    }
}

test "mat copyTo" {
    var mat = try Mat.initOnes(100, 102, .cv8sc1);
    defer mat.deinit();
    var mat2 = try Mat.init();
    defer mat2.deinit();
    mat.copyTo(&mat2);

    try testing.expectEqual(mat.rows(), mat2.rows());
    try testing.expectEqual(mat.cols(), mat2.cols());
    try testing.expectEqual(mat.channels(), mat2.channels());
    try testing.expectEqual(mat.getType(), mat2.getType());
    {
        var i: usize = 0;
        while (i < mat.rows()) : (i += 1) {
            var j: usize = 0;
            while (j < mat.cols()) : (j += 1) {
                try testing.expectEqual(mat.at(u8, i, j), mat2.at(u8, i, j));
            }
        }
    }
}

test "mat copyToWithMask" {
    var mat = try Mat.initSize(101, 102, .cv8uc1);
    defer mat.deinit();
    var diff = try Mat.init();
    defer diff.deinit();
    var mask = try Mat.initSize(101, 102, .cv8uc1);
    defer mask.deinit();

    mat.set(u8, 0, 0, 255);
    mat.set(u8, 0, 1, 255);

    mask.set(u8, 0, 0, 255);

    var copy = try Mat.init();
    defer copy.deinit();

    mat.copyToWithMask(&copy, mask);

    try testing.expectEqual(mat.rows(), copy.rows());
    try testing.expectEqual(mat.cols(), copy.cols());

    try testing.expectEqual(@as(u8, 255), copy.at(u8, 0, 0));
    try testing.expectEqual(@as(u8, 0), copy.at(u8, 0, 1));
}

test "mat clone" {
    var mat = try Mat.initOnes(100, 102, .cv8sc1);
    defer mat.deinit();

    mat.set(i8, 0, 0, 3);

    var clone = try mat.clone();
    defer clone.deinit();

    try testing.expectEqual(mat.rows(), clone.rows());
    try testing.expectEqual(mat.cols(), clone.cols());
    try testing.expectEqual(mat.channels(), clone.channels());
    try testing.expectEqual(mat.getType(), clone.getType());

    {
        var i: usize = 0;
        while (i < mat.rows()) : (i += 1) {
            var j: usize = 0;
            while (j < mat.cols()) : (j += 1) {
                try testing.expectEqual(mat.at(u8, i, j), clone.at(u8, i, j));
            }
        }
    }
}
