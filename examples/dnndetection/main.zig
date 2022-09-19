// This example uses a deep neural network to perform object detection.
// It can be used with either the Caffe face tracking or Tensorflow object detection models that are
// included with OpenCV 3.4
//
// To perform face tracking with the Caffe model:
//
// Download the model file from:
// https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
//
// You will also need the prototxt config file:
// https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
//

const std = @import("std");
const cv = @import("zigcv");
const Mat = cv.Mat;

const model_path = thisDir() ++ "/res10_300x300_ssd_iter_140000.caffemodel";
const config_path = thisDir() ++ "/deploy.ptototxt";

pub fn main() anyerror!void {
    var allocator = std.heap.page_allocator;
    var args = try std.process.argsWithAllocator(allocator);
    const prog = args.next();
    const device_id_char = args.next() orelse {
        std.log.err("usage: {s} [cameraID]", .{prog.?});
        std.os.exit(1);
    };
    args.deinit();

    const device_id = try std.fmt.parseUnsigned(c_int, device_id_char, 10);

    // open webcam
    var webcam = try cv.VideoCapture.init();
    try webcam.openDevice(device_id);
    defer webcam.deinit();

    // open display window
    const window_name = "DNN Detection";
    var window = try cv.Window.init(window_name);
    defer window.deinit();

    // prepare image matrix
    var img = try cv.Mat.init();
    defer img.deinit();

    // open DNN object tracking model
    var net = cv.Net.readNet(model_path, config_path) catch |err| {
        std.debug.print("Error: {}\n", .{err});
        std.os.exit(1);
    };
    defer net.deinit();

    if (net.empty()) {
        std.debug.print("Error: could not load model\n", .{});
        std.os.exit(1);
    }

    net.setPreferableBackend(.default);
    net.setPreferableTarget(.cpu);

    const ratio: f64 = 1;
    const mean = cv.Scalar.init(104, 177, 123, 0);
    const swap_rgb = false;

    while (true) {
        webcam.read(&img) catch {
            std.debug.print("capture failed", .{});
            std.os.exit(1);
        };
        if (img.isEmpty()) {
            continue;
        }

        var blob = try cv.blobFromImage(img, ratio, cv.Size{ .width = 300, .height = 300 }, mean, swap_rgb, false);
        defer blob.deinit();

        net.setInput(blob, "");

        var prob = try net.forward("");
        defer prob.deinit();

        performDetection(&img, prob);

        window.imShow(img);
        if (window.waitKey(1) >= 0) {
            break;
        }
    }
}

fn performDetection(frame: *Mat, results: Mat) void {
    const green = cv.Color{ .g = 255 };
    var i: usize = 0;
    while (i < results.total()) {
        var confidence = results.get(f32, 0, i + 2);
        const cols = @intToFloat(f32, frame.*.cols());
        const rows = @intToFloat(f32, frame.*.rows());
        if (confidence > 0.5) {
            var left = @floatToInt(i32, results.get(f32, 0, i + 3) * cols);
            var top = @floatToInt(i32, results.get(f32, 0, i + 4) * rows);
            var right = @floatToInt(i32, results.get(f32, 0, i + 5) * cols);
            var bottom = @floatToInt(i32, results.get(f32, 0, i + 6) * rows);
            cv.rectangle(frame, cv.Rect{ .x = left, .y = top, .width = right, .height = bottom }, green, 2);
        }
        i += 7;
    }
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}
