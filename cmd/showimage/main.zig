const std = @import("std");
const cv = @import("zigcv");
const cv_c_api = cv.c_api;

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
    _ = cv_c_api.Window_New(window_name, 0);
    defer cv_c_api.Window_Close(window_name);
    var img = cv_c_api.Image_IMRead(@ptrCast([*]const u8, img_PATH), -1);
    while (true) {
        _ = cv_c_api.Window_IMShow(window_name, img);
        if (cv_c_api.Window_WaitKey(1) >= 0) {
            break;
        }
    }
}
