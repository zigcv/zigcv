const std = @import("std");
const zigcv = @import("libs.zig");
pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const examples = [_]Program{
        .{
            .name = "hello",
            .path = "cmd/hello/main.zig",
            .desc = "Show Webcam",
        },
        .{
            .name = "version",
            .path = "cmd/version/main.zig",
            .desc = "Print OpenCV Version",
        },
        .{
            .name = "show_image",
            .path = "cmd/showimage/main.zig",
            .desc = "Show Image Demo",
        },
        .{
            .name = "face_detection",
            .path = "cmd/facedetect/main.zig",
            .desc = "Face Detection Demo",
        },
        .{
            .name = "face_blur",
            .path = "cmd/faceblur/main.zig",
            .desc = "Face Detection and Blur Demo",
        },
        .{
            .name = "dnn_detection",
            .path = "cmd/dnndetection/main.zig",
            .desc = "DNN Detection Demo",
        },
    };
    ensureSubmodules(b.allocator) catch |err| @panic(@errorName(err));

    const examples_step = b.step("examples", "Builds all the examples");

    for (examples) |ex| {
        const exe = b.addExecutable(ex.name, ex.path);

        exe.setBuildMode(mode);
        exe.setTarget(target);
        exe.install();

        zigcv.link(exe);
        zigcv.addAsPackage(exe);

        const run_cmd = exe.run();
        const run_step = b.step(ex.name, ex.desc);
        const artifact_step = &b.addInstallArtifact(exe).step;
        if (b.args) |args| {
            run_cmd.addArgs(args);
        }
        run_step.dependOn(artifact_step);
        run_step.dependOn(&run_cmd.step);
        examples_step.dependOn(&exe.step);
        examples_step.dependOn(artifact_step);
    }

    const exe_tests = b.addTest("src/main.zig");
    exe_tests.setTarget(target);
    exe_tests.setBuildMode(mode);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&exe_tests.step);
}

fn ensureSubmodules(allocator: std.mem.Allocator) !void {
    if (std.process.getEnvVarOwned(allocator, "NO_ENSURE_SUBMODULES")) |no_ensure_submodules| {
        if (std.mem.eql(u8, no_ensure_submodules, "true")) return;
    } else |_| {}
    var child = std.ChildProcess.init(&.{ "git", "submodule", "update", "--init", "--recursive" }, allocator);
    child.cwd = thisDir();
    child.stderr = std.io.getStdErr();
    child.stdout = std.io.getStdOut();
    _ = try child.spawnAndWait();
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}

const Program = struct {
    name: []const u8,
    path: []const u8,
    desc: []const u8,
};
