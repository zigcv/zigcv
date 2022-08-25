const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");

pub fn openCVVersion() []const u8 {
    return std.mem.span(c.openCVVersion());
}

//*    implementation done
//*    pub extern fn openCVVersion(...) [*c]const u8;

test "show version" {
    std.debug.print("version:\t{s}\n", .{openCVVersion()});
}
