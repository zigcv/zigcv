const std = @import("std");

pub fn addAsPackage(exe: *std.build.LibExeObjStep) void {
    addAsPackageWithCutsomName(exe, "zigcv");
}

pub fn addAsPackageWithCutsomName(exe: *std.build.LibExeObjStep, name: []const u8) void {
    exe.addPackagePath(name, getThisDir() ++ "/src/main.zig");
}

pub fn link(exe: *std.build.LibExeObjStep) void {
    ensureSubmodules(exe);

    const target = exe.target;
    const mode = exe.build_mode;
    const builder = exe.builder;

    const cv = builder.addStaticLibrary("opencv", null);
    cv.setTarget(target);
    cv.setBuildMode(mode);
    cv.force_pic = true;
    cv.addCSourceFiles(&.{
        go_srcdir ++ "asyncarray.cpp",
        go_srcdir ++ "calib3d.cpp",
        go_srcdir ++ "core.cpp",
        go_srcdir ++ "dnn.cpp",
        go_srcdir ++ "features2d.cpp",
        go_srcdir ++ "highgui.cpp",
        go_srcdir ++ "imgcodecs.cpp",
        go_srcdir ++ "imgproc.cpp",
        go_srcdir ++ "objdetect.cpp",
        go_srcdir ++ "photo.cpp",
        go_srcdir ++ "svd.cpp",
        go_srcdir ++ "version.cpp",
        go_srcdir ++ "video.cpp",
        go_srcdir ++ "videoio.cpp",
    }, c_build_options);

    linkToOpenCV(cv);

    exe.linkLibrary(cv);
    linkToOpenCV(exe);
}

fn linkToOpenCV(exe: *std.build.LibExeObjStep) void {
    const target_os = exe.target.toTarget().os.tag;

    exe.addIncludePath(go_srcdir);
    exe.addIncludePath(zig_src_dir);
    switch (target_os) {
        .windows => {
            exe.addIncludePath("c:/msys64/mingw64/include");
            exe.addIncludePath("c:/msys64/mingw64/include/c++/12.2.0");
            exe.addIncludePath("c:/msys64/mingw64/include/c++/12.2.0/x86_64-w64-mingw32");
            exe.addLibraryPath("c:/msys64/mingw64/lib");
            exe.addIncludePath("c:/opencv/build/install/include");
            exe.addLibraryPath("c:/opencv/build/install/x64/mingw/staticlib");

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

            exe.linkLibCpp();
            exe.linkSystemLibrary("opencv4");
            exe.linkSystemLibrary("unwind");
            exe.linkSystemLibrary("m");
            exe.linkSystemLibrary("c");
        },
    }
}

pub const contrib = struct {
    pub fn addAsPackage(exe: *std.build.LibExeObjStep) void {
        @This().addAsPackageWithCutsomName(exe, "zigcv_contrib");
    }

    pub fn addAsPackageWithCutsomName(exe: *std.build.LibExeObjStep, name: []const u8) void {
        exe.addPackagePath(name, getThisDir() ++ "/src/contrib/main.zig");
    }

    pub fn link(exe: *std.build.LibExeObjStep) void {
        ensureSubmodules(exe);

        const target = exe.target;
        const mode = exe.build_mode;
        const builder = exe.builder;

        const contrib_dir = go_srcdir ++ "contrib/";

        const cv_contrib = builder.addStaticLibrary("opencv_contrib", null);
        cv_contrib.setTarget(target);
        cv_contrib.setBuildMode(mode);
        cv_contrib.force_pic = true;
        cv_contrib.addCSourceFiles(&.{
            contrib_dir ++ "aruco.cpp",
            contrib_dir ++ "bgsegm.cpp",
            contrib_dir ++ "face.cpp",
            contrib_dir ++ "img_hash.cpp",
            contrib_dir ++ "tracking.cpp",
            contrib_dir ++ "wechat_qrcode.cpp",
            contrib_dir ++ "xfeatures2d.cpp",
            contrib_dir ++ "ximgproc.cpp",
            contrib_dir ++ "xphoto.cpp",
        }, c_build_options);
        cv_contrib.addIncludePath(contrib_dir);
        linkToOpenCV(cv_contrib);

        exe.linkLibrary(cv_contrib);
        linkToOpenCV(exe);
    }
};

pub const cuda = struct {
    pub fn addAsPackage(exe: *std.build.LibExeObjStep) void {
        @This().addAsPackageWithCutsomName(exe, "zigcv_cuda");
    }

    pub fn addAsPackageWithCutsomName(exe: *std.build.LibExeObjStep, name: []const u8) void {
        exe.addPackagePath(name, getThisDir() ++ "/src/cuda/main.zig");
    }

    pub fn link(exe: *std.build.LibExeObjStep) void {
        ensureSubmodules(exe);

        const target = exe.target;
        const mode = exe.build_mode;
        const builder = exe.builder;

        const cuda_dir = go_srcdir ++ "cuda/";

        const cv_cuda = builder.addStaticLibrary("opencv_cuda", null);
        cv_cuda.setTarget(target);
        cv_cuda.setBuildMode(mode);
        cv_cuda.force_pic = true;
        cv_cuda.addCSourceFiles(&.{
            cuda_dir ++ "arithm.cpp",
            cuda_dir ++ "bgsegm.cpp",
            cuda_dir ++ "core.cpp",
            cuda_dir ++ "cuda.cpp",
            cuda_dir ++ "filters.cpp",
            cuda_dir ++ "imgproc.cpp",
            cuda_dir ++ "objdetect.cpp",
            cuda_dir ++ "optflow.cpp",
            cuda_dir ++ "warping.cpp",
        }, c_build_options);
        cv_cuda.addIncludePath(go_srcdir);
        linkToOpenCV(cv_cuda);

        exe.linkLibrary(cv_cuda);
        linkToOpenCV(exe);
    }
};

inline fn getThisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}

var ensure_submodule: bool = false;
fn ensureSubmodules(exe: *std.build.LibExeObjStep) void {
    const b = exe.builder;
    if (!ensure_submodule) {
        exe.step.dependOn(&b.addSystemCommand(&.{ "git", "submodule", "update", "--init", "--recursive" }).step);
        ensure_submodule = true;
    }
}

const go_srcdir = getThisDir() ++ "/libs/gocv/";
const zig_src_dir = getThisDir() ++ "/src";

const Program = struct {
    name: []const u8,
    path: []const u8,
    desc: []const u8,
};

const c_build_options: []const []const u8 = &.{
    "-Wall",
    "-Wextra",
    "--std=c++11",
};
