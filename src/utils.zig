const std = @import("std");
const c = @import("c_api.zig");

pub fn castZigU8ToC(str: []const u8) [*]const u8 {
    return @ptrCast([*]const u8, str);
}

pub fn fromCStructsToArrayList(from_array: anytype, from_array_length: i32, comptime ToType: type, allocator: std.mem.Allocator) !std.ArrayList(ToType) {
    const len = @intCast(usize, from_array_length);
    var arr = try std.ArrayList(ToType).initCapacity(allocator, len);
    {
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const elem = ToType.initFromC(from_array[i]);
            if (comptime (@typeInfo(@TypeOf(elem)) == .ErrorUnion)) {
                try arr.append(try elem);
            } else {
                try arr.append(elem);
            }
        }
    }
    return arr;
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
    var child = std.ChildProcess.init(
        &.{
            "curl",
            "--create-dirs",
            "-LO",
            "--output-dir",
            dir,
            url,
        },
        arena_allocator,
    );
    child.stderr = std.io.getStdErr();
    child.stdout = std.io.getStdOut();

    const stat = std.fs.cwd().statFile(dir_filename) catch |err| {
        std.debug.print("filename: {s},\terror: {any}\n", .{ dir_filename, err });
        if (err != error.FileNotFound) unreachable;
        _ = try child.spawnAndWait();
        return;
    };
    if(stat.size == 0) {
        _ = try child.spawnAndWait();
        return;
    }
}

test "ensureNotNull" {
    var ptr: ?[*]const u8 = null;
    try std.testing.expectError(error.AllocationError, ensurePtrNotNull(ptr));

    const ptr2 = opaque {};
    try std.testing.expectEqual(*anyopaque, @TypeOf(try ensurePtrNotNull(&ptr2)));
}
