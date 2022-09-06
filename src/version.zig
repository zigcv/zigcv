const std = @import("std");
const c = @import("c_api.zig");

pub fn openCVVersion() []const u8 {
    return std.mem.span(c.openCVVersion());
}

//*    implementation done
//*    pub extern fn openCVVersion(...) [*c]const u8;

test "show version" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const parseFloat = std.fmt.parseFloat;

    const version = openCVVersion();

    const actual_version = (std.ChildProcess.exec(.{
        .allocator = allocator,
        .argv = &.{ "pkg-config", "--modversion", "opencv4" },
    }) catch {
        return;
    }).stdout;

    defer allocator.free(actual_version);

    try testing.expectEqual(parseFloat(f32, actual_version), parseFloat(f32, version));
}
