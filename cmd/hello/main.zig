const std = @import("std");
const fmt = std.fmt;
const cv = @import("zigcv");

pub fn main() anyerror!void {
    // open webcam
    var webcam = cv.VideoCapture_New();
    _ = cv.VideoCapture_OpenDevice(webcam, 0);
    defer cv.VideoCapture_Close(webcam);

    // open display window
    const window_name = "Hello";
    _ = cv.Window_New(window_name, 0);
    defer cv.Window_Close(window_name);

    var img = cv.Mat_New();
    defer cv.Mat_Close(img);

    while (true) {
        if (cv.VideoCapture_Read(webcam, img) != 1) {
            std.debug.print("capture failed", .{});
            std.os.exit(1);
        }

        _ = cv.Window_IMShow(window_name, img);
        if (cv.Window_WaitKey(1) >= 0) {
            break;
        }
    }
}
