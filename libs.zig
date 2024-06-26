const std = @import("std");
const ArrayList = std.ArrayList;
const LazyPath = std.build.LazyPath;

pub fn addAsPackage(exe: *std.Build.CompileStep) void {
    addAsPackageWithCustomName(exe, "zigcv");
}

pub fn addAsPackageWithCustomName(exe: *std.Build.CompileStep, name: []const u8) void {
    const owner = exe.step.owner;
    var module = std.build.createModule(owner, .{
        .source_file = std.Build.FileSource.relative("src/main.zig"),
        .dependencies = &.{},
    });
    exe.addModule(name, module);
}

pub fn link(b: *std.build.Builder, exe: *std.Build.CompileStep) void {
    ensureSubmodules(exe);

    const target = exe.target;
    const mode = exe.optimize;
    const builder = exe.step.owner;

    const go_src_files = .{
        "asyncarray.cpp",
        "calib3d.cpp",
        "core.cpp",
        "dnn.cpp",
        "features2d.cpp",
        "highgui.cpp",
        "imgcodecs.cpp",
        "imgproc.cpp",
        "objdetect.cpp",
        "photo.cpp",
        "svd.cpp",
        "version.cpp",
        "video.cpp",
        "videoio.cpp",
    };

    const cv = builder.addStaticLibrary(std.Build.StaticLibraryOptions{
        .name = "opencv",
        .target = target,
        .optimize = mode,
    });

    inline for (go_src_files) |file| {
        const go_src_dir_path = go_src_dir.getPath(b);
        const c_file_path = b.pathJoin(&.{ go_src_dir_path, file });
        cv.addCSourceFile(.{
            .file = .{ .path = c_file_path },
            .flags = c_build_options,
        });
    }

    linkToOpenCV(cv);

    exe.linkLibrary(cv);
    linkToOpenCV(exe);
}

fn linkToOpenCV(exe: *std.build.CompileStep) void {
    const target_os = exe.target.toTarget().os.tag;

    exe.addIncludePath(go_src_dir);
    exe.addIncludePath(zig_src_dir);
    switch (target_os) {
        .windows => {
            exe.addIncludePath(.{ .path = "c:/msys64/mingw64/include" });
            exe.addIncludePath(.{ .path = "c:/msys64/mingw64/include/c++/12.2.0" });
            exe.addIncludePath(.{ .path = "c:/msys64/mingw64/include/c++/12.2.0/x86_64-w64-mingw32" });
            exe.addLibraryPath(.{ .path = "c:/msys64/mingw64/lib" });
            exe.addIncludePath(.{ .path = "c:/opencv/build/install/include" });
            exe.addLibraryPath(.{ .path = "c:/opencv/build/install/x64/mingw/staticlib" });

            exe.linkSystemLibrary("opencv4");
            exe.linkSystemLibrary("stdc++.dll");
            exe.linkSystemLibrary("unwind");
            exe.linkSystemLibrary("m");
            exe.linkSystemLibrary("c");
        },
        else => {
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
        exe.addPackagePath(name, .{ .path = "src/contrib/main.zig" });
    }

    pub fn link(b: *std.build.Builder, exe: *std.build.LibExeObjStep) void {
        ensureSubmodules(exe);

        const target = exe.target;
        const optimize = exe.optimize;
        const builder = exe.step.owner;

        const contrib_dir = b.pathJoin(.{ go_src_dir.getPath(b), "contrib/" });

        const contrib_files = .{
            "aruco.cpp",
            "bgsegm.cpp",
            "face.cpp",
            "img_hash.cpp",
            "tracking.cpp",
            "wechat_qrcode.cpp",
            "xfeatures2d.cpp",
            "ximgproc.cpp",
            "xphoto.cpp",
        };

        const cv_contrib = builder.addStaticLibrary(.{
            .name = "opencv_contrib",
            .target = target,
            .optimize = optimize,
        });
        cv_contrib.force_pic = true;
        for (contrib_files) |file| {
            const c_path = b.pathJoin(&.{ contrib_dir, file });
            cv_contrib.addCSourceFile(.{
                .file = .{ .path = c_path },
                .flags = c_build_options,
            });
        }
        cv_contrib.addIncludePath(.{ .path = contrib_dir });
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
        exe.addPackagePath(name, .{ .path = "src/cuda/main.zig" });
    }

    pub fn link(b: *std.build.Builder, exe: *std.build.LibExeObjStep) void {
        ensureSubmodules(exe);

        const target = exe.target;
        const mode = exe.build_mode;
        const builder = exe.step.owner;

        const cuda_dir = b.pathJoin(&.{ go_src_dir.getPath(builder), "cuda/" });

        const cuda_files = .{
            "arithm.cpp",
            "bgsegm.cpp",
            "core.cpp",
            "cuda.cpp",
            "filters.cpp",
            "imgproc.cpp",
            "objdetect.cpp",
            "optflow.cpp",
            "warping.cpp",
        };

        const cv_cuda = builder.addStaticLibrary("opencv_cuda");
        cv_cuda.setTarget(target);
        cv_cuda.setBuildMode(mode);
        cv_cuda.force_pic = true;
        for (cuda_files) |file| {
            const c_path = b.pathJoin(&.{ cuda_dir, file });
            cv_cuda.addCSourceFile(.{
                .file = .{ .path = c_path },
                .flags = c_build_options,
            });
        }
        cv_cuda.addIncludePath(go_src_dir);
        linkToOpenCV(cv_cuda);

        exe.linkLibrary(cv_cuda);
        linkToOpenCV(exe);
    }
};

var ensure_submodule: bool = false;
fn ensureSubmodules(exe: *std.build.LibExeObjStep) void {
    const b = exe.step.owner;
    if (!ensure_submodule) {
        exe.step.dependOn(&b.addSystemCommand(&.{ "git", "submodule", "update", "--init", "--recursive" }).step);
        ensure_submodule = true;
    }
}

const go_src_dir = (LazyPath{ .path = "libs/gocv/" });
const zig_src_dir = (LazyPath{ .path = "src/" });

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
