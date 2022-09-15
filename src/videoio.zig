const c = @import("c_api.zig");
const core = @import("core.zig");

// Select preferred API for a capture object.
// Note: Backends are available only if they have been built with your OpenCV binaries
pub const VideoCaptureAPI = enum(i32) {
    // Auto detect == 0
    VideoCaptureAny = 0,

    // Video For Windows (obsolete, removed)
    VideoCaptureVFW = 200,

    // V4L/V4L2 capturing support
    VideoCaptureV4L = 200,

    // Same as VideoCaptureV4L
    VideoCaptureV4L2 = 200,

    // IEEE 1394 drivers
    VideoCaptureFirewire = 300,

    // Same value as VideoCaptureFirewire
    VideoCaptureFireware = 300,

    // Same value as VideoCaptureFirewire
    VideoCaptureIEEE1394 = 300,

    // Same value as VideoCaptureFirewire
    VideoCaptureDC1394 = 300,

    // Same value as VideoCaptureFirewire
    VideoCaptureCMU1394 = 300,

    // QuickTime (obsolete, removed)
    VideoCaptureQT = 500,

    // Unicap drivers (obsolete, removed)
    VideoCaptureUnicap = 600,

    // DirectShow (via videoInput)
    VideoCaptureDshow = 700,

    // PvAPI, Prosilica GigE SDK
    VideoCapturePvAPI = 800,

    // OpenNI (for Kinect)
    VideoCaptureOpenNI = 900,

    // OpenNI (for Asus Xtion)
    VideoCaptureOpenNIAsus = 910,

    // Android - not used
    VideoCaptureAndroid = 1000,

    // XIMEA Camera API
    VideoCaptureXiAPI = 1100,

    // AVFoundation framework for iOS (OS X Lion will have the same API)
    VideoCaptureAVFoundation = 1200,

    // Smartek Giganetix GigEVisionSDK
    VideoCaptureGiganetix = 1300,

    // Microsoft Media Foundation (via videoInput)
    VideoCaptureMSMF = 1400,

    // Microsoft Windows Runtime using Media Foundation
    VideoCaptureWinRT = 1410,

    // RealSense (former Intel Perceptual Computing SDK)
    VideoCaptureIntelPerc = 1500,

    // Synonym for VideoCaptureIntelPerc
    VideoCaptureRealsense = 1500,

    // OpenNI2 (for Kinect)
    VideoCaptureOpenNI2 = 1600,

    // OpenNI2 (for Asus Xtion and Occipital Structure sensors)
    VideoCaptureOpenNI2Asus = 1610,

    // gPhoto2 connection
    VideoCaptureGPhoto2 = 1700,

    // GStreamer
    VideoCaptureGstreamer = 1800,

    // Open and record video file or stream using the FFMPEG library
    VideoCaptureFFmpeg = 1900,

    // OpenCV Image Sequence (e.g. img_%02d.jpg)
    VideoCaptureImages = 2000,

    // Aravis SDK
    VideoCaptureAravis = 2100,

    // Built-in OpenCV MotionJPEG codec
    VideoCaptureOpencvMjpeg = 2200,

    // Intel MediaSDK
    VideoCaptureIntelMFX = 2300,

    // XINE engine (Linux)
    VideoCaptureXINE = 2400,
};

// VideoCaptureProperties are the properties used for VideoCapture operations.
pub const VideoCaptureProperties = enum(i32) {
    // VideoCapturePosMsec contains current position of the
    // video file in milliseconds.
    VideoCapturePosMsec = 0,

    // VideoCapturePosFrames 0-based index of the frame to be
    // decoded/captured next.
    VideoCapturePosFrames = 1,

    // VideoCapturePosAVIRatio relative position of the video file:
    // 0=start of the film, 1=end of the film.
    VideoCapturePosAVIRatio = 2,

    // VideoCaptureFrameWidth is width of the frames in the video stream.
    VideoCaptureFrameWidth = 3,

    // VideoCaptureFrameHeight controls height of frames in the video stream.
    VideoCaptureFrameHeight = 4,

    // VideoCaptureFPS controls capture frame rate.
    VideoCaptureFPS = 5,

    // VideoCaptureFOURCC contains the 4-character code of codec.
    // see VideoWriter::fourcc for details.
    VideoCaptureFOURCC = 6,

    // VideoCaptureFrameCount contains number of frames in the video file.
    VideoCaptureFrameCount = 7,

    // VideoCaptureFormat format of the Mat objects returned by
    // VideoCapture::retrieve().
    VideoCaptureFormat = 8,

    // VideoCaptureMode contains backend-specific value indicating
    // the current capture mode.
    VideoCaptureMode = 9,

    // VideoCaptureBrightness is brightness of the image
    // (only for those cameras that support).
    VideoCaptureBrightness = 10,

    // VideoCaptureContrast is contrast of the image
    // (only for cameras that support it).
    VideoCaptureContrast = 11,

    // VideoCaptureSaturation saturation of the image
    // (only for cameras that support).
    VideoCaptureSaturation = 12,

    // VideoCaptureHue hue of the image (only for cameras that support).
    VideoCaptureHue = 13,

    // VideoCaptureGain is the gain of the capture image.
    // (only for those cameras that support).
    VideoCaptureGain = 14,

    // VideoCaptureExposure is the exposure of the capture image.
    // (only for those cameras that support).
    VideoCaptureExposure = 15,

    // VideoCaptureConvertRGB is a boolean flags indicating whether
    // images should be converted to RGB.
    VideoCaptureConvertRGB = 16,

    // VideoCaptureWhiteBalanceBlueU is currently unsupported.
    VideoCaptureWhiteBalanceBlueU = 17,

    // VideoCaptureRectification is the rectification flag for stereo cameras.
    // Note: only supported by DC1394 v 2.x backend currently.
    VideoCaptureRectification = 18,

    // VideoCaptureMonochrome indicates whether images should be
    // converted to monochrome.
    VideoCaptureMonochrome = 19,

    // VideoCaptureSharpness controls image capture sharpness.
    VideoCaptureSharpness = 20,

    // VideoCaptureAutoExposure controls the DC1394 exposure control
    // done by camera, user can adjust reference level using this feature.
    VideoCaptureAutoExposure = 21,

    // VideoCaptureGamma controls video capture gamma.
    VideoCaptureGamma = 22,

    // VideoCaptureTemperature controls video capture temperature.
    VideoCaptureTemperature = 23,

    // VideoCaptureTrigger controls video capture trigger.
    VideoCaptureTrigger = 24,

    // VideoCaptureTriggerDelay controls video capture trigger delay.
    VideoCaptureTriggerDelay = 25,

    // VideoCaptureWhiteBalanceRedV controls video capture setting for
    // white balance.
    VideoCaptureWhiteBalanceRedV = 26,

    // VideoCaptureZoom controls video capture zoom.
    VideoCaptureZoom = 27,

    // VideoCaptureFocus controls video capture focus.
    VideoCaptureFocus = 28,

    // VideoCaptureGUID controls video capture GUID.
    VideoCaptureGUID = 29,

    // VideoCaptureISOSpeed controls video capture ISO speed.
    VideoCaptureISOSpeed = 30,

    // VideoCaptureBacklight controls video capture backlight.
    VideoCaptureBacklight = 32,

    // VideoCapturePan controls video capture pan.
    VideoCapturePan = 33,

    // VideoCaptureTilt controls video capture tilt.
    VideoCaptureTilt = 34,

    // VideoCaptureRoll controls video capture roll.
    VideoCaptureRoll = 35,

    // VideoCaptureIris controls video capture iris.
    VideoCaptureIris = 36,

    // VideoCaptureSettings is the pop up video/camera filter dialog. Note:
    // only supported by DSHOW backend currently. The property value is ignored.
    VideoCaptureSettings = 37,

    // VideoCaptureBufferSize controls video capture buffer size.
    VideoCaptureBufferSize = 38,

    // VideoCaptureAutoFocus controls video capture auto focus..
    VideoCaptureAutoFocus = 39,

    // VideoCaptureSarNumerator controls the sample aspect ratio: num/den (num)
    VideoCaptureSarNumerator = 40,

    // VideoCaptureSarDenominator controls the sample aspect ratio: num/den (den)
    VideoCaptureSarDenominator = 41,

    // VideoCaptureBackend is the current api backend (VideoCaptureAPI). Read-only property.
    VideoCaptureBackend = 42,

    // VideoCaptureChannel controls the video input or channel number (only for those cameras that support).
    VideoCaptureChannel = 43,

    // VideoCaptureAutoWB controls the auto white-balance.
    VideoCaptureAutoWB = 44,

    // VideoCaptureWBTemperature controls the white-balance color temperature
    VideoCaptureWBTemperature = 45,

    // VideoCaptureCodecPixelFormat shows the the codec's pixel format (4-character code). Read-only property.
    // Subset of AV_PIX_FMT_* or -1 if unknown.
    VideoCaptureCodecPixelFormat = 46,

    // VideoCaptureBitrate displays the video bitrate in kbits/s. Read-only property.
    VideoCaptureBitrate = 47,
};

pub const VideoCapture = struct {
    ptr: c.VideoCapture,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .ptr = c.VideoCapture_New(),
        };
    }

    pub fn deinit(self: *Self) void {
        c.VideoCapture_Close(self.ptr);
        self.*.ptr = null;
    }

    pub fn captureFile(self: *Self, uri: []const u8) !void {
        const c_uri = @ptrCast([*]const u8, uri);
        if (!c.VideoCapture_Open(self.ptr, c_uri)) {
            return error.VideoCaptureOpenFileError;
        }
    }

    pub fn captureFileWithAPI(self: *Self, uri: []const u8, apiPreference: VideoCaptureAPI) !void {
        const cURI = @ptrCast([*]const u8, uri);
        if (!c.VideoCapture_OpenWithAPI(self.ptr, cURI, @enumToInt(apiPreference))) {
            return error.VideoCaptureOpenFileError;
        }
    }

    pub fn openDevice(self: *Self, device: i32) !void {
        if (!c.VideoCapture_OpenDevice(self.ptr, device)) {
            return error.VideoCaptureError;
        }
    }

    pub fn openDeviceWithAPI(self: *Self, device: i32, apiPreference: VideoCaptureAPI) !void {
        if (!c.VideoCapture_OpenDeviceWithAPI(self.ptr, device, @enumToInt(apiPreference))) {
            return error.VideoCaptureError;
        }
    }

    pub fn get(self: Self, prop: VideoCaptureProperties) f64 {
        return c.VideoCapture_Get(self.ptr, @enumToInt(prop));
    }

    pub fn set(self: *Self, prop: VideoCaptureProperties, param: f64) void {
        return c.VideoCapture_Set(self.ptr, @enumToInt(prop), param);
    }

    pub fn grab(self: Self, skip: i32) void {
        c.VideoCapture_Grab(self.ptr, skip);
    }

    pub fn read(self: Self, buf: *core.Mat) !void {
        if (c.VideoCapture_Read(self.ptr, buf.*.ptr) == 0) {
            return error.VideCaptureError;
        }
    }

    pub fn isOpened(self: Self) bool {
        return c.VideoCapture_IsOpened(self.ptr) != 0;
    }
};

pub const VideoWriter = struct {
    ptr: c.VideoWriter,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .ptr = c.VideoWriter_New(),
        };
    }

    pub fn deinit(self: *Self) void {
        c.VideoWriter_Close(self.ptr);
        self.*.ptr = null;
    }

    pub fn open(
        self: *Self,
        name: []const u8,
        codec: []const u8,
        fps: f64,
        width: i32,
        height: i32,
        is_color: bool,
    ) void {
        const c_name = @ptrCast([*]const u8, name);
        const c_codec = @ptrCast([*]const u8, codec);
        _ = c.VideoWriter_Open(
            self.ptr,
            c_name,
            c_codec,
            fps,
            width,
            height,
            is_color,
        );
    }

    pub fn write(self: *Self, img: *core.Mat) void {
        _ = c.VideoWriter_Write(self.ptr, img.*.ptr);
    }

    pub fn isOpened(self: *Self) bool {
        return c.VideoWriter_IsOpened(self.ptr) != 0;
    }
};

//*    implementation done
//*    pub const VideoCapture = ?*anyopaque;
//*    pub const VideoWriter = ?*anyopaque;
//*    pub extern fn VideoCapture_New(...) VideoCapture;
//*    pub extern fn VideoCapture_Close(v: VideoCapture) void;
//*    pub extern fn VideoCapture_Open(v: VideoCapture, uri: [*c]const u8) bool;
//*    pub extern fn VideoCapture_OpenWithAPI(v: VideoCapture, uri: [*c]const u8, apiPreference: c_int) bool;
//*    pub extern fn VideoCapture_OpenDevice(v: VideoCapture, device: c_int) bool;
//*    pub extern fn VideoCapture_OpenDeviceWithAPI(v: VideoCapture, device: c_int, apiPreference: c_int) bool;
//*    pub extern fn VideoCapture_Set(v: VideoCapture, prop: c_int, param: f64) void;
//*    pub extern fn VideoCapture_Get(v: VideoCapture, prop: c_int) f64;
//*    pub extern fn VideoCapture_IsOpened(v: VideoCapture) c_int;
//*    pub extern fn VideoCapture_Read(v: VideoCapture, buf: Mat) c_int;
//*    pub extern fn VideoCapture_Grab(v: VideoCapture, skip: c_int) void;
//*    pub extern fn VideoWriter_New(...) VideoWriter;
//*    pub extern fn VideoWriter_Close(vw: VideoWriter) void;
//*    pub extern fn VideoWriter_Open(vw: VideoWriter, name: [*c]const u8, codec: [*c]const u8, fps: f64, width: c_int, height: c_int, isColor: bool) void;
//*    pub extern fn VideoWriter_IsOpened(vw: VideoWriter) c_int;
//*    pub extern fn VideoWriter_Write(vw: VideoWriter, img: Mat) void;
