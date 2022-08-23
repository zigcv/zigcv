const std = @import("std");
const fmt = std.fmt;
const cv = @import("zigcv");

pub fn main() anyerror!void {
    var args = try std.process.argsWithAllocator(std.heap.page_allocator);
    defer args.deinit();
    const prog = args.next();
    const img_PATH = args.next() orelse {
        std.log.err("usage: {s} [image_PATH]", .{prog.?});
        std.os.exit(1);
    };

    // open display window
    const window_name = "Show Image";
    _ = cv.Window_New(window_name, 0);
    defer cv.Window_Close(window_name);
    var img = cv.Image_IMRead(@ptrCast([*]u8, img_PATH), -1);
    while (true) {
        _ = cv.Window_IMShow(window_name, img);
        if (cv.Window_WaitKey(1) >= 0) {
            break;
        }
    }
}
