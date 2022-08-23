const std = @import("std");
const fmt = std.fmt;
const cv = @import("zigcv");

pub fn main() anyerror!void {
    var args = try std.process.argsWithAllocator(std.heap.page_allocator);
    defer args.deinit();
    const prog = args.next();
    const devicIDChar = args.next() orelse {
        std.log.err("usage: {s} [cameraID]", .{prog.?});
        std.os.exit(1);
    };
    const deviceID = try fmt.parseUnsigned(c_int, devicIDChar, 10);

    // open webcam
    var webcam = cv.VideoCapture_New();
    _ = cv.VideoCapture_OpenDevice(webcam, deviceID);
    defer cv.VideoCapture_Close(webcam);

    // open display window
    const window_name = "Face Detect";
    _ = cv.Window_New(window_name, 0);
    defer cv.Window_Close(window_name);

    // prepare image matrix
    var img = cv.Mat_New();
    defer cv.Mat_Close(img);

    // color for the rect when faces detected
    const blue = cv.Scalar{
        .val1 = @as(f64, 0),
        .val2 = @as(f64, 255),
        .val3 = @as(f64, 0),
        .val4 = @as(f64, 0),
    };

    // load classifier to recognize faces
    var classifier = cv.CascadeClassifier_New();
    defer cv.CascadeClassifier_Close(classifier);

    if (cv.CascadeClassifier_Load(classifier, "./libs/gocv/data/haarcascade_frontalface_default.xml") != 1) {
        std.debug.print("no xml", .{});
        std.os.exit(1);
    }

    while (true) {
        if (cv.VideoCapture_Read(webcam, img) != 1) {
            std.debug.print("capture failed", .{});
            std.os.exit(1);
        }
        if (cv.Mat_Empty(img) == 1) {
            continue;
        }
        const rects = cv.CascadeClassifier_DetectMultiScale(classifier, img);
        std.debug.print("found {d} faces\n", .{rects.length});
        {
            var i: c_int = 0;
            while (i < rects.length) : (i += 1) {
                const r = rects.rects[0];
                std.debug.print("x:\t{}, y:\t{}, w\t{}, h\t{}\n", .{ r.x, r.y, r.width, r.height });
                cv.Rectangle(img, r, blue, 3);
            }
        }

        _ = cv.Window_IMShow(window_name, img);
        if (cv.Window_WaitKey(1) >= 0) {
            break;
        }
    }
}
