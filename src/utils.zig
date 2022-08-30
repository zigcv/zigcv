const std = @import("std");
const c = @import("c_api.zig");

pub fn castZigU8ToC(str: []const u8) [*]const u8 {
    return @ptrCast([*]const u8, str);
}

pub fn cStringsToU8Array(cstr: c.CStrings, allocator: std.mem.Allocator) !std.ArrayList([]const u8) {
    var list = std.ArrayList([]const u8).init(allocator);
    {
        var i: usize = 0;
        while (i < cstr.length) : (i += 1) {
            try list.append(cstr.strs[i]);
        }
    }
    return list;
}
