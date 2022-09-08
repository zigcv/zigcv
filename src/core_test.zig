const std = @import("std");
const testing = std.testing;
const allocator = std.testing.allocator;
const core = @import("core.zig");
const Mat = core.Mat;

test "mat" {
    var mat = Mat.init();
    defer mat.deinit();
    try testing.expectEqual(true, mat.isEmpty());
}

test "mat size" {
    const size: usize = 10;
    var mat = Mat.initSize(10, 10, .cv8uc1);
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
    var mat = Mat.initSize(1, 1, .cv8uc1);
    defer mat.deinit();

    try testing.expectEqual(@as(usize, 1), mat.channels());
}

test "mat type" {
    var mat = Mat.initSize(1, 1, .cv8uc1);
    defer mat.deinit();

    try testing.expectEqual(core.Mat.MatType.cv8uc1, mat.getType());

    var mat2 = Mat.initSize(1, 1, .cv16sc2);
    defer mat2.deinit();

    try testing.expectEqual(Mat.MatType.cv16sc2, mat2.getType());
}

test "mat eye" {
    var mat = Mat.eye(3, 3, .cv8sc1);
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
    var mat = Mat.zeros(3, 3, .cv8sc1);
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
    var mat = Mat.ones(3, 3, .cv8sc1);
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
