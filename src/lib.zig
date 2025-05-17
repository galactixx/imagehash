const impl = @import("imagehash.zig");

pub const ImageHash = impl.ImageHash;
pub const Error = impl.Error;
pub const ParseError = impl.ParseError;
pub const fromJSON = impl.fromJSON;
pub const averageHash = impl.averageHash;
pub const differenceHash = impl.differenceHash;
pub const perceptualHash = impl.perceptualHash;
pub const waveletHash = impl.waveletHash;