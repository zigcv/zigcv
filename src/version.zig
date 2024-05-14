const std = @import("std");
const c = @import("c_api.zig");

/// Return OpenCV version as a string.
pub fn openCVVersion() []const u8 {
    return std.mem.span(c.openCVVersion());
}

test "show version" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const parseFloat = std.fmt.parseFloat;

    const version = openCVVersion();

    var child_process =
        std.ChildProcess.init(
        &.{ "pkg-config", "--modversion", "opencv4" },
        allocator,
    );

    _ = try child_process.spawnAndWait();

    var stdout = std.ArrayList(u8).init(allocator);
    var stderr = std.ArrayList(u8).init(allocator);

    defer {
        stdout.deinit();
        stderr.deinit();
    }

    try child_process.collectOutput(&stdout, &stderr, 100);

    const actual_version = try stdout.toOwnedSlice();

    try testing.expectEqual(parseFloat(f32, actual_version), parseFloat(f32, version));
}

//*    implementation done
//*    pub extern fn openCVVersion(...) [*c]const u8;
