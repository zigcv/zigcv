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
    var window = cv.Window.init(window_name, .WindowNormal);
    defer window.deinit();

    var img = cv.Mat.init();
    defer img.deinit();

    while (true) {
        webcam.read(&img) catch {
            std.debug.print("capture failed", .{});
            std.os.exit(1);
        };

        window.imShow(img);
        if (window.waitKey(1) >= 0) {
            break;
        }
    }
}
