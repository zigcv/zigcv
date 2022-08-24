const std = @import("std");
const cv = @import("zigcv");
const cv_c_api = cv.c_api;

pub fn main() anyerror!void {
    var args = try std.process.argsWithAllocator(std.heap.page_allocator);
    defer args.deinit();
    const prog = args.next();
    const devicIDChar = args.next() orelse {
        std.log.err("usage: {s} [cameraID]", .{prog.?});
        std.os.exit(1);
    };
    const deviceID = try std.fmt.parseUnsigned(c_int, devicIDChar, 10);

    // open webcam
    var webcam = cv_c_api.VideoCapture_New();
    _ = cv_c_api.VideoCapture_OpenDevice(webcam, deviceID);
    defer cv_c_api.VideoCapture_Close(webcam);

    // open display window
    const window_name = "Face Detect";
    _ = cv_c_api.Window_New(window_name, 0);
    defer cv_c_api.Window_Close(window_name);

    // prepare image matrix
    var img = cv.Mat.init();
    defer img.deinit();

    // load classifier to recognize faces
    var classifier = cv_c_api.CascadeClassifier_New();
    defer cv_c_api.CascadeClassifier_Close(classifier);

    if (cv_c_api.CascadeClassifier_Load(classifier, "./libs/gocv/data/haarcascade_frontalface_default.xml") != 1) {
        std.debug.print("no xml", .{});
        std.os.exit(1);
    }

    while (true) {
        if (cv_c_api.VideoCapture_Read(webcam, img.ptr) != 1) {
            std.debug.print("capture failed", .{});
            std.os.exit(1);
        }
        if (cv_c_api.Mat_Empty(img.ptr) == 1) {
            continue;
        }
        const rects = cv_c_api.CascadeClassifier_DetectMultiScale(classifier, img.ptr);
        std.debug.print("found {d} faces\n", .{rects.length});
        {
            var i: c_int = 0;
            while (i < rects.length) : (i += 1) {
                const r = rects.rects[0];
                std.debug.print("x:\t{}, y:\t{}, w\t{}, h\t{}\n", .{ r.x, r.y, r.width, r.height });
                var size = cv_c_api.Size{
                    .width = 75,
                    .height = 75,
                };
                cv_c_api.GaussianBlur(img.ptr, img.ptr, size, 0, 0, 4);
            }
        }

        _ = cv_c_api.Window_IMShow(window_name, img.ptr);
        if (cv_c_api.Window_WaitKey(1) >= 0) {
            break;
        }
    }
}
