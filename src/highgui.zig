const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const castToC = utils.castZigU8ToC;

pub const WindowFlag = enum {
    // WindowNormal indicates a normal window.
    WindowNormal,

    // WindowAutosize indicates a window sized based on the contents.
    WindowAutosize,

    // WindowFullscreen indicates a full-screen window.
    WindowFullscreen,

    // WindowFreeRatio indicates allow the user to resize without maintaining aspect ratio.
    WindowFreeRatio,

    // WindowKeepRatio indicates always maintain an aspect ratio that matches the contents.
    WindowKeepRatio,
};

pub const WindowPropertyFlag = enum(i32) {
    // WindowPropertyFullscreen fullscreen property
    // (can be WINDOW_NORMAL or WINDOW_FULLSCREEN).
    WindowPropertyFullscreen = 0,

    // WindowPropertyAutosize is autosize property
    // (can be WINDOW_NORMAL or WINDOW_AUTOSIZE).
    WindowPropertyAutosize = 1,

    // WindowPropertyAspectRatio window's aspect ration
    // (can be set to WINDOW_FREERATIO or WINDOW_KEEPRATIO).
    WindowPropertyAspectRatio = 2,

    // WindowPropertyOpenGL opengl support.
    WindowPropertyOpenGL = 3,

    // WindowPropertyVisible or not.
    WindowPropertyVisible = 4,

    // WindowPropertyTopMost status bar and tool bar
    WindowPropertyTopMost = 5,

    // WindowPropertyTopMost to toggle normal window being topmost or not
    WindowPropertyTopMost = 5,
};

pub const Window = struct {
    name: []const u8,
    trackbar_name: ?[]const u8,
    open: bool,

    const Self = @This();

    fn getCWindowName(self: Self) [*]const u8 {
        return castToC(self.name);
    }

    fn getCTrackbarName(self: Self) ![*]const u8 {
        if (self.trackbar_name) |tn| {
            return castToC(tn);
        } else {
            return error.WindowTrackbarNameNotFoundError;
        }
    }

    pub fn init(window_name: []const u8, flags: WindowFlag) Self {
        c.Window_New(@ptrCast([*]const u8, window_name), Self.windowFlagToNum(flags, c_int));
        return .{
            .name = window_name,
            .open = true,
            .trackbar_name = null,
        };
    }

    pub fn deinit(self: *Self) void {
        _ = c.Window_Close(self.getCWindowName());
        self.open = false;
    }

    pub fn isOpen(self: Self) bool {
        return self.open;
    }

    // SetWindowProperty changes parameters of a window dynamically.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga66e4a6db4d4e06148bcdfe0d70a5df27
    //
    pub fn setProperty(self: *Self, flag: WindowPropertyFlag, value: WindowFlag) void {
        _ = c.Window_SetProperty(
            self.getCWindowName(),
            @enumToInt(flag),
            self.windowFlagToNum(value, f64),
        );
    }

    // GetWindowProperty returns properties of a window.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/dfc/group__highgui.html#gaaf9504b8f9cf19024d9d44a14e461656
    //

    pub fn getProperty(self: Self, flag: WindowPropertyFlag) WindowFlag {
        const f: f64 = c.Window_GetProperty(
            self.getCWindowName(),
            @enumToInt(flag),
        );
        return self.numToWindowFlag(f);
    }

    pub fn setTitle(self: *Self, title: []const u8) void {
        _ = c.Window_SetTitle(self.getCWindowName(), castToC(title));
    }

    // WaitKey waits for a pressed key.
    // This function is the only method in OpenCV's HighGUI that can fetch
    // and handle events, so it needs to be called periodically
    // for normal event processing
    //
    // For further details, please see:
    // http://docs.opencv.org/master/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7
    //
    pub fn waitKey(self: Self, delay: c_int) c_int {
        _ = self;
        return c.Window_WaitKey(delay);
    }

    // IMShow displays an image Mat in the specified window.
    // This function should be followed by the WaitKey function which displays
    // the image for specified milliseconds. Otherwise, it won't display the image.
    //
    // For further details, please see:
    // http://docs.opencv.org/master/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563
    //
    pub fn imShow(self: *Self, mat: core.Mat) void {
        _ = c.Window_IMShow(self.getCWindowName(), mat.ptr);
    }

    // MoveWindow moves window to the specified position.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga8d86b207f7211250dbe6e28f76307ffb
    //
    pub fn move(self: *Self, x: c_int, y: c_int) void {
        _ = c.Window_Move(self.getCWindowName(), x, y);
    }

    // ResizeWindow resizes window to the specified size.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga9e80e080f7ef33f897e415358aee7f7e
    //
    pub fn resize(self: *Self, width: c_int, height: c_int) void {
        _ = c.Window_Resize(self.getCWindowName(), width, height);
    }

    // CreateTrackbar creates a trackbar and attaches it to the specified window.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/dfc/group__highgui.html#gaf78d2155d30b728fc413803745b67a9b
    //
    pub fn createTrackbar(self: *Self, trackbar_name: []const u8, max: i32) void {
        _ = c.Trackbar_Create(self.getCWindowName(), trackbar_name, trackbar_name, max);
        self.trackbar_name = trackbar_name;
    }

    // CreateTrackbarWithValue works like CreateTrackbar but also assigns a
    // variable value to be a position synchronized with the trackbar.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/dfc/group__highgui.html#gaf78d2155d30b728fc413803745b67a9b
    //
    pub fn createTrackbarWithValue(self: *Self, trackbar_name: []const u8, value: []i32, max: i32) void {
        var c_value = @ptrCast([*]const c_int, value);
        _ = c.Trackbar_CreateWithValue(self.getCWindowName(), trackbar_name, c_value, max);
        self.trackbar_name = trackbar_name;
    }

    // GetPos returns the trackbar position.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga122632e9e91b9ec06943472c55d9cda8
    //
    pub fn trackBarGetPos(self: Self) !c_int {
        const trackbar_name = try self.getCTrackbarName();
        return c.Trackbar_GetPos(self.getCWindowName(), trackbar_name);
    }

    // SetPos sets the trackbar position.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga67d73c4c9430f13481fd58410d01bd8d
    //
    pub fn trackBarSetPos(self: *Self, pos: c_int) !void {
        const trackbar_name = try self.getCTrackbarName();
        _ = c.Trackbar_SetPos(self.getCWindowName(), trackbar_name, pos);
    }

    // SetMin sets the trackbar minimum position.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/dfc/group__highgui.html#gabe26ffe8d2b60cc678895595a581b7aa
    //
    pub fn trackBarSetMin(self: *Self, pos: c_int) !void {
        const trackbar_name = try self.getCTrackbarName();
        _ = c.Trackbar_SetMin(self.getCWindowName(), trackbar_name, pos);
    }

    // SetMax sets the trackbar maximum position.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga7e5437ccba37f1154b65210902fc4480
    //
    pub fn trackBarSetMax(self: *Self, pos: c_int) !void {
        const trackbar_name = try self.getCTrackbarName();
        _ = c.Trackbar_SetMax(self.getCWindowName(), trackbar_name, pos);
    }

    fn windowFlagToNum(wf: WindowFlag, comptime T: type) T {
        return switch (wf) {
            .WindowNormal => 0x00000000,
            .WindowAutosize => 0x00000001,
            .WindowFullscreen => 1,
            .WindowFreeRatio => 0x00000100,
            .WindowKeepRatio => 0x00000000,
        };
    }

    fn numToWindowFlag(wf: anytype) WindowFlag {
        return switch (wf) {
            0x00000000 => .WindowNormal,
            0x00000001 => .WindowAutosize,
            1 => .WindowFullscreen,
            0x00000100 => .WindowFreeRatio,
            0x00000000 => .WindowKeepRatio,
            else => @panic("invalid number"),
        };
    }
};

//*    implementation done
//*    pub extern fn Window_New(winname: [*c]const u8, flags: c_int) void;
//*    pub extern fn Window_Close(winname: [*c]const u8) void;
//*    pub extern fn Window_IMShow(winname: [*c]const u8, mat: Mat) void;
//*    pub extern fn Window_GetProperty(winname: [*c]const u8, flag: c_int) f64;
//*    pub extern fn Window_SetProperty(winname: [*c]const u8, flag: c_int, value: f64) void;
//*    pub extern fn Window_SetTitle(winname: [*c]const u8, title: [*c]const u8) void;
//*    pub extern fn Window_WaitKey(c_int) c_int;
//*    pub extern fn Window_Move(winname: [*c]const u8, x: c_int, y: c_int) void;
//*    pub extern fn Window_Resize(winname: [*c]const u8, width: c_int, height: c_int) void;
//     pub extern fn Window_SelectROI(winname: [*c]const u8, img: Mat) struct_Rect;
//     pub extern fn Window_SelectROIs(winname: [*c]const u8, img: Mat) struct_Rects;
//*    pub extern fn Trackbar_Create(winname: [*c]const u8, trackname: [*c]const u8, max: c_int) void;
//*    pub extern fn Trackbar_CreateWithValue(winname: [*c]const u8, trackname: [*c]const u8, value: [*c]c_int, max: c_int) void;
//*    pub extern fn Trackbar_GetPos(winname: [*c]const u8, trackname: [*c]const u8) c_int;
//*    pub extern fn Trackbar_SetPos(winname: [*c]const u8, trackname: [*c]const u8, pos: c_int) void;
//*    pub extern fn Trackbar_SetMin(winname: [*c]const u8, trackname: [*c]const u8, pos: c_int) void;
//*    pub extern fn Trackbar_SetMax(winname: [*c]const u8, trackname: [*c]const u8, pos: c_int) void;
