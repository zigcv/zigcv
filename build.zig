const std = @import("std");
pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const cv = b.addStaticLibrary("opencv", null);
    cv.setTarget(target);
    cv.setBuildMode(mode);
    cv.force_pic = true;
    addPkg(cv);
    cv.addCSourceFiles(&.{
        srcdir ++ "/core.cpp",
        srcdir ++ "/videoio.cpp",
        srcdir ++ "/highgui.cpp",
        srcdir ++ "/imgcodecs.cpp",
        srcdir ++ "/objdetect.cpp",
        srcdir ++ "/imgproc.cpp",
    }, &.{
        "--std=c++11",
    });

    const cvPkg = std.build.Pkg{
        .name = "zigcv",
        .source = std.build.FileSource{ .path = "src/main.zig" },
    };

    const examples = [_]Program{
        .{
            .name = "face_detection",
            .path = "cmd/facedetect/main.zig",
            .desc = "Face Detection Demo",
        },
    };

    const examples_step = b.step("examples", "Builds all the examples");

    for (examples) |ex| {
        const exe = b.addExecutable(ex.name, ex.path);

        exe.setBuildMode(mode);
        exe.setTarget(target);

        addPkg(exe);
        exe.linkLibrary(cv);
        exe.addPackage(cvPkg);
        exe.install();

        const run_cmd = exe.run();
        const run_step = b.step(ex.name, ex.desc);
        run_step.dependOn(&run_cmd.step);
        examples_step.dependOn(&exe.step);
    }
}

fn addPkg(exe: *std.build.LibExeObjStep) void {
    // https://github.com/hybridgroup/gocv/blob/4597f3ddbb/cgo.go
    // https://github.com/hybridgroup/gocv/blob/4597f3ddbb/cgo_static.go
    const target_os = exe.target.toTarget().os.tag;
    switch (target_os) {
        .windows => {
            exe.addIncludePath("c:/msys64/mingw64/include");
            exe.addIncludePath("c:/msys64/mingw64/include/c++/12.2.0");
            exe.addIncludePath("c:/msys64/mingw64/include/c++/12.2.0/x86_64-w64-mingw32");
            exe.addLibraryPath("c:/msys64/mingw64/lib");
            exe.addIncludePath("c:/opencv/build/install/include");
            exe.addLibraryPath("c:/opencv/build/install/x64/mingw/staticlib");
            exe.addIncludePath(srcdir);

            exe.linkSystemLibrary("opencv4");
            exe.linkSystemLibrary("stdc++.dll");
            exe.linkSystemLibrary("unwind");
            exe.linkSystemLibrary("m");
            exe.linkSystemLibrary("c");
        },
        else => {
            exe.addIncludePath("/usr/local/include");
            exe.addIncludePath("/usr/local/include/opencv4");
            exe.addIncludePath("/opt/homebrew/include");
            exe.addIncludePath("/opt/homebrew/include/opencv4");

            exe.addLibraryPath("/usr/local/lib");
            exe.addLibraryPath("/usr/local/lib/opencv4/3rdparty");
            exe.addLibraryPath("/opt/homebrew/lib");
            exe.addLibraryPath("/opt/homebrew/lib/opencv4/3rdparty");
            exe.addIncludePath(srcdir);

            exe.linkLibCpp();
            exe.linkSystemLibrary("opencv4");
            exe.linkSystemLibrary("unwind");
            exe.linkSystemLibrary("m");
            exe.linkSystemLibrary("c");
        },
    }
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

const srcdir = thisDir() ++ "/libs/gocv";
