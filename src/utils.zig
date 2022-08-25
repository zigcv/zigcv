pub fn castZigU8ToC(str: []const u8) [*]const u8 {
    return @ptrCast([*]const u8, str);
}
