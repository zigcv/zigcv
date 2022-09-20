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
            try arr.append(try wrap(to_type, to_type.initFromC(from_array[i])));
        }
    }
    return arr;
}

fn wrap(comptime T: type, value: anytype) !T {
    return value;
}

pub fn ensurePtrNotNull(ptr: ?*anyopaque) !*anyopaque {
    if (ptr == null) return error.AllocationError;
    return ptr.?;
}

pub fn downloadFile(url: []const u8, dir: []const u8, allocator: std.mem.Allocator) !void {
    if (dir[dir.len - 1] != '/') unreachable;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var split_url = std.mem.split(u8, url, "/");
    var filename: []const u8 = undefined;
    while (split_url.next()) |s| {
        filename = s;
    }
    const dir_filename = try std.fmt.allocPrint(
        arena_allocator,
        "{s}{s}",
        .{ dir, filename },
    );
    _ = std.fs.cwd().statFile(dir_filename) catch |err| {
        std.debug.print("filename: {s},\terror: {any}\n", .{ dir_filename, err });
        if (err != error.FileNotFound) unreachable;
        var child = std.ChildProcess.init(
            &.{
                "curl",
                "--create-dirs",
                "-O",
                "--output-dir",
                dir,
                url,
            },
            arena_allocator,
        );
        child.stderr = std.io.getStdErr();
        child.stdout = std.io.getStdOut();
        _ = child.spawnAndWait() catch unreachable;
    };
}

test "ensureNotNull" {
    var ptr: ?[*]const u8 = null;
    try std.testing.expectError(error.AllocationError, ensurePtrNotNull(ptr));

    const ptr2 = opaque {};
    try std.testing.expectEqual(*anyopaque, @TypeOf(try ensurePtrNotNull(&ptr2)));
}
