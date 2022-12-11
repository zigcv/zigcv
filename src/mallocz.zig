// original code from zig-gamedev https://github.com/michal-z/zig-gamedev/blob/befbf1adb4/libs/zstbi/src/zstbi.zig

const std = @import("std");
const assert = std.debug.assert;

var mem_allocator: ?std.mem.Allocator = null;
var mem_allocations: ?std.AutoHashMap(usize, usize) = null;
var mem_mutex: std.Thread.Mutex = .{};
const mem_alignment = 16;
var check_allocation_count: bool = false;

pub const Config = struct {
    allocator: std.mem.Allocator,
    check_allocation_count: bool = false,
};

pub fn init(config: Config) void {
    assert(mem_allocator == null);
    mem_allocator = config.allocator;
    mem_allocations = std.AutoHashMap(usize, usize).init(config.allocator);
}

pub fn deinit() void {
    assert(mem_allocator != null);
    std.debug.assert(mem_allocations != null);
    if (check_allocation_count) std.debug.assert(mem_allocations.?.count() == 0);
    mem_allocations.?.deinit();
    mem_allocations = null;
    mem_allocator = null;
}

pub inline fn getAllocator() std.mem.Allocator {
    assert(mem_allocator != null);
    return mem_allocator.?;
}

pub export fn malloc(size: usize) callconv(.C) ?*anyopaque {
    mem_mutex.lock();
    defer mem_mutex.unlock();

    const mem = mem_allocator.?.allocBytes(
        mem_alignment,
        size,
        0,
        @returnAddress(),
    ) catch @panic("zstbi: out of memory");

    mem_allocations.?.put(@ptrToInt(mem.ptr), size) catch @panic("mallocz:\t" ++ "out of memory");

    std.debug.print("malloc size: {any}\tptr: 0x{x}\n", .{ size, @ptrToInt(mem.ptr) });

    return mem.ptr;
}

pub export fn realloc(ptr: ?*anyopaque, size: usize) callconv(.C) ?*anyopaque {
    mem_mutex.lock();
    defer mem_mutex.unlock();

    const old_size = if (ptr != null) mem_allocations.?.get(@ptrToInt(ptr.?)).? else 0;
    const old_mem = if (old_size > 0)
        @ptrCast([*]u8, ptr)[0..old_size]
    else
        @as([*]u8, undefined)[0..0];

    const new_mem = mem_allocator.?.reallocBytes(
        old_mem,
        mem_alignment,
        size,
        mem_alignment,
        0,
        @returnAddress(),
    ) catch @panic("zstbi: out of memory");

    if (ptr != null) {
        const removed = mem_allocations.?.remove(@ptrToInt(ptr.?));
        std.debug.assert(removed);
    }

    mem_allocations.?.put(@ptrToInt(new_mem.ptr), size) catch @panic("mallocz:\t" ++ "out of memory");

    std.debug.print("ralloc size: {any}\told: 0x{x}\tptr: 0x{x}\n", .{ size, @ptrToInt(old_mem.ptr), @ptrToInt(new_mem.ptr) });

    return new_mem.ptr;
}

pub export fn free(maybe_ptr: ?*anyopaque) callconv(.C) void {
    if (maybe_ptr) |ptr| {
        mem_mutex.lock();
        defer mem_mutex.unlock();

        std.debug.print("try to free  ptr: 0x{any}\n", .{@ptrToInt(ptr)});
        if (mem_allocations.?.contains(@ptrToInt(ptr))) {
            const size = mem_allocations.?.fetchRemove(@ptrToInt(ptr)).?.value;
            const mem = @ptrCast(
                [*]align(mem_alignment) u8,
                @alignCast(mem_alignment, ptr),
            )[0..size];
            mem_allocator.?.free(mem);
            std.debug.print("free size: {any}\tptr: 0x{x}\n", .{ size, @ptrToInt(ptr) });
        } else std.debug.print("failed to free  ptr: 0x{any}\n", .{@ptrToInt(ptr)});
    }
}

test "test" {}
