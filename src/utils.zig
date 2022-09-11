const std = @import("std");
const c = @import("c_api.zig");

pub fn castZigU8ToC(str: []const u8) [*]const u8 {
    return @ptrCast([*]const u8, str);
}

pub fn fromCStructsToArrayList(from_array: anytype, from_array_length: i32, comptime to_type: type, allocator: std.mem.Allocator) !std.ArrayList(to_type) {
    const len = @intCast(usize, from_array_length);
    var arr = try std.ArrayList(to_type).initCapacity(allocator, len);
    {
        var i: usize = 0;
        while (i < len) : (i += 1) {
            try arr.append(to_type.fromC(from_array[i]));
        }
    }
    return arr;
}

pub fn ensurePtrNotNull(ptr: ?*anyopaque) !*anyopaque {
    if (ptr == null) return error.AllocationError;
    return ptr.?;
}

test "ensureNotNull" {
    var ptr: ?[*]const u8 = null;
    try std.testing.expectError(error.AllocationError, ensurePtrNotNull(ptr));

    const ptr2 = opaque {};
    try std.testing.expectEqual(*anyopaque, @TypeOf(try ensurePtrNotNull(&ptr2)));
}
