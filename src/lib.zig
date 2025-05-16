const impl = @import("imagehash.zig");

pub const Error = impl.Error;
pub const averageHash = impl.averageHash;
pub const differenceHash = impl.differenceHash;
pub const perceptualHash = impl.perceptualHash;
pub const waveletHash = impl.waveletHash;