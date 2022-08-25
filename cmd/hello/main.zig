const std = @import("std");
const cv = @import("zigcv");
const cv_c_api = cv.c_api;

pub fn main() anyerror!void {
    // open webcam
    var webcam = cv.VideoCapture.init();
    try webcam.openDevice(0);
    defer webcam.deinit();

    // open display window
    const window_name = "Hello";
    _ = cv_c_api.Window_New(window_name, 0);
    defer cv_c_api.Window_Close(window_name);

    var img = cv.Mat.init();
    defer img.deinit();

    while (true) {
        webcam.read(&img) catch {
            std.debug.print("capture failed", .{});
            std.os.exit(1);
        };

        _ = cv_c_api.Window_IMShow(window_name, img.ptr);
        if (cv_c_api.Window_WaitKey(1) >= 0) {
            break;
        }
    }
}
