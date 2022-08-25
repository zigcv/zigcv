const std = @import("std");
const cv = @import("zigcv");
const cv_c_api = cv.c_api;

pub fn main() anyerror!void {
    var args = try std.process.argsWithAllocator(std.heap.page_allocator);
    defer args.deinit();
    const prog = args.next();
    const devic_id_char = args.next() orelse {
        std.log.err("usage: {s} [cameraID]", .{prog.?});
        std.os.exit(1);
    };
    const device_id = try std.fmt.parseUnsigned(c_int, devic_id_char, 10);

    // open webcam
    var webcam = cv.VideoCapture.init();
    try webcam.openDevice(device_id);
    defer webcam.deinit();

    // open display window
    const window_name = "Face Detect";
    var window = cv.Window.init(window_name, .WindowNormal);
    defer window.deinit();

    // prepare image matrix
    var img = cv.Mat.init();
    defer img.deinit();

    // color for the rect when faces detected
    const blue = cv_c_api.Scalar{
        .val1 = @as(f64, 0),
        .val2 = @as(f64, 255),
        .val3 = @as(f64, 0),
        .val4 = @as(f64, 0),
    };

    // load classifier to recognize faces
    var classifier = cv_c_api.CascadeClassifier_New();
    defer cv_c_api.CascadeClassifier_Close(classifier);

    if (cv_c_api.CascadeClassifier_Load(classifier, "./libs/gocv/data/haarcascade_frontalface_default.xml") != 1) {
        std.debug.print("no xml", .{});
        std.os.exit(1);
    }

    while (true) {
        webcam.read(&img) catch {
            std.debug.print("capture failed", .{});
            std.os.exit(1);
        };
        if (img.isEmpty()) {
            continue;
        }
        const rects = cv_c_api.CascadeClassifier_DetectMultiScale(classifier, img.ptr);
        std.debug.print("found {d} faces\n", .{rects.length});
        {
            var i: c_int = 0;
            while (i < rects.length) : (i += 1) {
                const r = rects.rects[0];
                std.debug.print("x:\t{}, y:\t{}, w\t{}, h\t{}\n", .{ r.x, r.y, r.width, r.height });
                cv_c_api.Rectangle(img.ptr, r, blue, 3);
            }
        }

        window.imShow(img);
        if (window.waitKey(1) >= 0) {
            break;
        }
    }
}
