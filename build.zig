const std = @import("std");
pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const cv = b.addStaticLibrary("opencv", null);
    cv.setTarget(target);
    cv.setBuildMode(mode);
    cv.linkLibCpp();
    cv.force_pic = true;
    addPkg(cv);
    cv.addCSourceFiles(&.{
        "libs/gocv/core.cpp",
        "libs/gocv/videoio.cpp",
        "libs/gocv/highgui.cpp",
        "libs/gocv/imgcodecs.cpp",
        "libs/gocv/objdetect.cpp",
        "libs/gocv/imgproc.cpp",
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
            exe.linkSystemLibrary("lopencv_stereo460");
            exe.linkSystemLibrary("lopencv_tracking460");
            exe.linkSystemLibrary("lopencv_superres460");
            exe.linkSystemLibrary("lopencv_stitching460");
            exe.linkSystemLibrary("lopencv_optflow460");
            exe.linkSystemLibrary("lopencv_gapi460");
            exe.linkSystemLibrary("lopencv_face460");
            exe.linkSystemLibrary("lopencv_dpm460");
            exe.linkSystemLibrary("lopencv_dnn_objdetect460");
            exe.linkSystemLibrary("lopencv_ccalib460");
            exe.linkSystemLibrary("lopencv_bioinspired460");
            exe.linkSystemLibrary("lopencv_bgsegm460");
            exe.linkSystemLibrary("lopencv_aruco460");
            exe.linkSystemLibrary("lopencv_xobjdetect460");
            exe.linkSystemLibrary("lopencv_ximgproc460");
            exe.linkSystemLibrary("lopencv_xfeatures2d460");
            exe.linkSystemLibrary("lopencv_videostab460");
            exe.linkSystemLibrary("lopencv_video460");
            exe.linkSystemLibrary("lopencv_structured_light460");
            exe.linkSystemLibrary("lopencv_shape460");
            exe.linkSystemLibrary("lopencv_rgbd460");
            exe.linkSystemLibrary("lopencv_rapid460");
            exe.linkSystemLibrary("lopencv_objdetect460");
            exe.linkSystemLibrary("lopencv_mcc460");
            exe.linkSystemLibrary("lopencv_highgui460");
            exe.linkSystemLibrary("lopencv_datasets460");
            exe.linkSystemLibrary("lopencv_calib3d460");
            exe.linkSystemLibrary("lopencv_videoio460");
            exe.linkSystemLibrary("lopencv_text460");
            exe.linkSystemLibrary("lopencv_line_descriptor460");
            exe.linkSystemLibrary("lopencv_imgcodecs460");
            exe.linkSystemLibrary("lopencv_img_hash460");
            exe.linkSystemLibrary("lopencv_hfs460");
            exe.linkSystemLibrary("lopencv_fuzzy460");
            exe.linkSystemLibrary("lopencv_features2d460");
            exe.linkSystemLibrary("lopencv_dnn_superres460");
            exe.linkSystemLibrary("lopencv_dnn460");
            exe.linkSystemLibrary("lopencv_xphoto460");
            exe.linkSystemLibrary("lopencv_wechat_qrcode460");
            exe.linkSystemLibrary("lopencv_surface_matching460");
            exe.linkSystemLibrary("lopencv_reg460");
            exe.linkSystemLibrary("lopencv_quality460");
            exe.linkSystemLibrary("lopencv_plot460");
            exe.linkSystemLibrary("lopencv_photo460");
            exe.linkSystemLibrary("lopencv_phase_unwrapping460");
            exe.linkSystemLibrary("lopencv_ml460");
            exe.linkSystemLibrary("lopencv_intensity_transform460");
            exe.linkSystemLibrary("lopencv_imgproc460");
            exe.linkSystemLibrary("lopencv_flann460");
            exe.linkSystemLibrary("lopencv_core460");
            exe.linkSystemLibrary("lade");
            exe.linkSystemLibrary("lquirc");
            exe.linkSystemLibrary("llibprotobuf");
            exe.linkSystemLibrary("lIlmImf");
            exe.linkSystemLibrary("llibpng");
            exe.linkSystemLibrary("llibopenjp2");
            exe.linkSystemLibrary("llibwebp");
            exe.linkSystemLibrary("llibtiff");
            exe.linkSystemLibrary("llibjpeg-turbo");
            exe.linkSystemLibrary("lzlib");
            exe.linkSystemLibrary("lkernel32");
            exe.linkSystemLibrary("lgdi32");
            exe.linkSystemLibrary("lwinspool");
            exe.linkSystemLibrary("lshell32");
            exe.linkSystemLibrary("lole32");
            exe.linkSystemLibrary("loleaut32");
            exe.linkSystemLibrary("luuid");
            exe.linkSystemLibrary("lcomdlg32");
            exe.linkSystemLibrary("ladvapi32");
            exe.linkSystemLibrary("luser32");

            exe.addIncludePath("C:/opencv/build/install/include");
            exe.addLibraryPath("C:/opencv/build/install/x64/mingw/staticlib");
            exe.addLibraryPath("C:/opencv/build/install/x64/mingw/staticlib");
        },
        else => {
            exe.linkSystemLibrary("opencv_gapi");
            exe.linkSystemLibrary("opencv_stitching");
            exe.linkSystemLibrary("opencv_aruco");
            exe.linkSystemLibrary("opencv_bgsegm");
            exe.linkSystemLibrary("opencv_bioinspired");
            exe.linkSystemLibrary("opencv_ccalib");
            exe.linkSystemLibrary("opencv_dnn_objdetect");
            exe.linkSystemLibrary("opencv_dpm");
            exe.linkSystemLibrary("opencv_face");
            exe.linkSystemLibrary("opencv_fuzzy");
            exe.linkSystemLibrary("opencv_hfs");
            exe.linkSystemLibrary("opencv_img_hash");
            exe.linkSystemLibrary("opencv_line_descriptor");
            exe.linkSystemLibrary("opencv_quality");
            exe.linkSystemLibrary("opencv_reg");
            exe.linkSystemLibrary("opencv_rgbd");
            exe.linkSystemLibrary("opencv_saliency");
            exe.linkSystemLibrary("opencv_stereo");
            exe.linkSystemLibrary("opencv_structured_light");
            exe.linkSystemLibrary("opencv_phase_unwrapping");
            exe.linkSystemLibrary("opencv_superres");
            exe.linkSystemLibrary("opencv_optflow");
            exe.linkSystemLibrary("opencv_surface_matching");
            exe.linkSystemLibrary("opencv_tracking");
            exe.linkSystemLibrary("opencv_datasets");
            exe.linkSystemLibrary("opencv_text");
            exe.linkSystemLibrary("opencv_highgui");
            exe.linkSystemLibrary("opencv_dnn");
            exe.linkSystemLibrary("opencv_plot");
            exe.linkSystemLibrary("opencv_videostab");
            exe.linkSystemLibrary("opencv_video");
            exe.linkSystemLibrary("opencv_videoio");
            exe.linkSystemLibrary("opencv_xfeatures2d");
            exe.linkSystemLibrary("opencv_shape");
            exe.linkSystemLibrary("opencv_ml");
            exe.linkSystemLibrary("opencv_ximgproc");
            exe.linkSystemLibrary("opencv_xobjdetect");
            exe.linkSystemLibrary("opencv_objdetect");
            exe.linkSystemLibrary("opencv_calib3d");
            exe.linkSystemLibrary("opencv_imgcodecs");
            exe.linkSystemLibrary("opencv_features2d");
            exe.linkSystemLibrary("opencv_flann");
            exe.linkSystemLibrary("opencv_xphoto");
            exe.linkSystemLibrary("opencv_wechat_qrcode");
            exe.linkSystemLibrary("opencv_photo");
            exe.linkSystemLibrary("opencv_imgproc");
            exe.linkSystemLibrary("opencv_core");
            exe.linkSystemLibrary("ittnotify");
            // exe.linkSystemLibrary("libprotobuf");
            // exe.linkSystemLibrary("IlmImf");
            exe.linkSystemLibrary("quirc");
            // exe.linkSystemLibrary("ippiw");
            // exe.linkSystemLibrary("ippicv");
            exe.linkSystemLibrary("ade");
            exe.linkSystemLibrary("z");
            exe.linkSystemLibrary("jpeg");
            exe.linkSystemLibrary("dl");
            exe.linkSystemLibrary("m");
            exe.linkSystemLibrary("pthread");
            exe.linkSystemLibrary("rt");
            // exe.linkSystemLibrary("quadmath");

            exe.addIncludePath("libs/gocv");

            exe.addIncludePath("/usr/local/include");
            exe.addIncludePath("/usr/local/include/opencv4");
            exe.addIncludePath("/opt/homebrew/include");
            exe.addIncludePath("/opt/homebrew/include/opencv4");

            exe.addLibraryPath("/usr/local/lib");
            exe.addLibraryPath("/usr/local/lib/opencv4/3rdparty");
            exe.addLibraryPath("/opt/homebrew/lib");
            exe.addLibraryPath("/opt/homebrew/lib/opencv4/3rdparty");
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
