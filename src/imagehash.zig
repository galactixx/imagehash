const std = @import("std");
const c = @cImport({
    @cInclude("stb_image.h");
    @cInclude("stb_image_resize.h");
});

pub const Error = error{
    ResizeFailed,
    LoadFailed,
    OutOfMemory,
};

// run a wavelet transform given a specific transform
// object, either a RowTransform or ColumnTransform
// object
fn wavelet(comptime T: type, transform: *T, size: usize) void {
    for (0..size) |idx| {
        var pairIdx: usize = 0;
        while (pairIdx < size) : (pairIdx += 2) {
            transform.decomp(idx, pairIdx);
        }
        transform.write(idx);
    }
}

// run a decomposition on a given pair of values
// both an average and a difference is computed and then
// written to the temporary buffer
fn d(comptime T: type, t: *T, cur: usize, next: usize, y: usize) void {
    const invSqrt = 1.0 / std.math.sqrt(2.0);
    const avg = (t.t[cur] + t.t[next]) * invSqrt;
    const diff = (t.t[cur] - t.t[next]) * invSqrt;
    t.temp[y / 2] = avg;
    t.temp[t.temp.len / 2 + y / 2] = diff;
}

// contains logic housed within two methods (write, decomp)
// for the row transform which does the decomposition and
// also writes to the row buffer
const RowTransform = struct {
    size: usize,
    t: []f32,
    temp: []f32,

    fn write(self: *RowTransform, x: usize) void {
        const start = x * self.size;
        const end = start + self.temp.len;
        @memcpy(self.t[start..end], self.temp[0..self.temp.len]);
    }

    fn decomp(self: *RowTransform, x: usize, pair: usize) void {
        const cur = x * self.size + pair;
        d(RowTransform, self, cur, cur + 1, pair);
    }
};

// contains logic housed within two methods (write, decomp)
// for the column transform which does the decomposition and
// also writes to the row buffer
const ColumnTransform = struct {
    size: usize,
    t: []f32,
    temp: []f32,

    fn write(self: *ColumnTransform, x: usize) void {
        for (0..self.temp.len) |i| {
            self.t[x + i * self.size] = self.temp[i];
        }
    }

    fn decomp(self: *ColumnTransform, x: usize, pair: usize) void {
        const cur = x + pair * self.size;
        d(ColumnTransform, self, cur, cur + self.size, pair);
    }
};

fn int64ToHex(hex: []u8, hash: u64) ![]u8 {
    return try std.fmt.bufPrint(hex, "{x}", .{hash});
}

// calculates an image size from two c_int variables and
// converts the result to a usize
fn imageSizeFromC(width: c_int, height: c_int) usize {
    const imgWidth: usize = @intCast(width);
    const imgHeight: usize = @intCast(height);
    return imgWidth * imgHeight;
}

// 
pub const ImageHash = struct {
    hashType: []const u8,
    hash: u64,
    bits: u8 = 64,

    pub fn toJSON(self: ImageHash, alloc: std.mem.Allocator) ![]u8 {    
        return try std.json.stringifyAlloc(alloc, self, .{});
    }

    pub fn distance(self: ImageHash, hash: ImageHash) u64 {
        const diff = self.hash ^ hash.hash;
        const hammingDistance = @popCount(diff);
        return hammingDistance;
    }

    pub fn hexDigest(self: ImageHash, buf: []u8) ![]u8 {
        return try int64ToHex(buf, self.hash);
    }
};

// a simple struct to store the image and metadata after
// loading an image
const ImageLoad = struct {
    image: []u8,
    width: c_int,
    height: c_int,

    fn imageSize(self: ImageLoad) usize {
        return imageSizeFromC(self.width, self.height);
    }

    fn freeMemory(self: *ImageLoad) void {
        c.stbi_image_free(self.image.ptr);
    }
};

// converts an image with RGB channels into a grayscale image
// calculated using the luminosity formula
// L = (R * 299 + G * 587 + B * 114) / 1000
fn rgb2Gray(imageBuf: []u8, grayBuf: []u8, channels: usize) void {
    var iter: usize = 0;
    while (iter < imageBuf.len) : (iter += channels) {
        const r: f32 = @floatFromInt(imageBuf[iter + 0]);
        const g: f32 = @floatFromInt(imageBuf[iter + 1]);
        const b: f32 = @floatFromInt(imageBuf[iter + 2]);
        const y: f32 = std.math.round((r * 299 + g * 587 + b * 114) / 1000);
        grayBuf[iter / channels] = @intFromFloat(y);
    }
}

// calculates the median for an array of numbers
// uses a naive approach rather than a quick select algorithm
fn calcMedian(nums: []f32, comptime size: usize) f32 {
    var sorted: [size]f32 = undefined;
    @memcpy(sorted[0..size], nums[0..size]);
    std.mem.sort(f32, &sorted, {}, std.sort.asc(f32));

    const midIdx = sorted.len / 2;
    if (sorted.len % 2 == 0) {
        return (sorted[midIdx - 1] + sorted[midIdx]) / 2;
    } else {
        return sorted[midIdx];
    }
}

// calculates a hash by applying median thresholding to
// each pixel in the array
fn hashFromMedian(buf: []f32, median: f32) u64 {
    var hash: u64 = 0;
    for (0..buf.len) |i| {
        if (buf[i] >= median) {
            const typedI: u6 = @intCast(i);
            hash |= @as(u64, 1) << typedI;
        }
    }
    return hash;
}

// extracts the top S x S block from the input array
fn extractLL(in: []f32, out: []f32, llSize: usize, size: usize) void {
    var medianIdx: usize = 0;
    for (0..llSize) |x| {
        for (0..llSize) |y| {
            out[medianIdx] = in[x * size + y];
            medianIdx += 1;
        }
    }
}

// given a filename will load an image into an 1D slice
// and return an ImageLoad object with the slice and
// accompanying metadata
fn loadImage(filename: []const u8, channels: u8) Error!ImageLoad {
    const allocator = std.heap.page_allocator;

    var outX: c_int = 0;
    var outY: c_int = 0;
    var outChannels: c_int = 0;

    // build a null‑terminated copy of filename
    const c_filename = allocator.alloc(u8, filename.len + 1) catch {
        return Error.OutOfMemory;
    };
    defer allocator.free(c_filename);
    @memcpy(c_filename[0..filename.len], filename);
    c_filename[filename.len] = 0;

    const data = c.stbi_load(
        c_filename.ptr,
        &outX,
        &outY,
        &outChannels,
        channels
    );

    if (data == null) {
        return Error.LoadFailed;
    }
    const imgSize = imageSizeFromC(outX, outY) * channels;
    const raw: [*]u8 = @ptrCast(data);
    const slice: []u8 = raw[0..imgSize];
    return ImageLoad{ .image = slice, .width = outX, .height = outY };
}

// implementation of the average hash algorithm
pub fn averageHash(filename: []const u8) Error!ImageHash {
    const allocator = std.heap.page_allocator;

    const resizeWidth = 8;
    const resizeHeight = 8;
    const resize = resizeHeight * resizeWidth;
    const channels = 3;

    var image = try loadImage(filename, channels);

    // allocate a buffer for the grayscaled image
    const grayBuf = try allocator.alloc(u8, image.imageSize());
    rgb2Gray(image.image, grayBuf, channels);

    // define an array and resize the grayscaled image
    var resizeBuf: [resize]u8 = undefined;
    const ok = c.stbir_resize_uint8_generic(
        grayBuf.ptr,
        image.width,
        image.height, 0,
        resizeBuf[0..].ptr,
        @intCast(resizeWidth),
        @intCast(resizeHeight),
        0,
        1,
        -1,
        0,
        c.STBIR_EDGE_CLAMP,
        c.STBIR_FILTER_TRIANGLE,
        c.STBIR_COLORSPACE_LINEAR,
        null
    );
    if (ok == 0) {
        return Error.ResizeFailed;
    }
    defer image.freeMemory();
    defer allocator.free(grayBuf);

    var pixelSum: usize = 0;
    for (0..resizeBuf.len) |i| {
        pixelSum += resizeBuf[i];
    }

    const pixelAvg: u64 = pixelSum / resizeBuf.len;

    // apply threholding to each pixel value after resizing
    // and accumulate a hash based on the results
    var hash: u64 = 0;
    for (0..resizeBuf.len) |i| {
        const typedI: u6 = @intCast(i);
        if (resizeBuf[i] >= pixelAvg) {
            hash |= @as(u64, 1) << typedI;
        }
    }
    return ImageHash{.hashType = "average", .hash = hash};
}

// implementation of the difference hash algorithm
pub fn differenceHash(filename: []const u8) Error!ImageHash {
    const allocator = std.heap.page_allocator;

    const resizeWidth = 9;
    const resizeHeight = 8;
    const resize = resizeHeight * resizeWidth;
    const channels = 3;

    var image = try loadImage(filename, channels);

    // allocate a buffer for the grayscaled image
    const grayBuf = try allocator.alloc(u8, image.imageSize());
    rgb2Gray(image.image, grayBuf, channels);

    // define an array and resize the grayscaled image
    var resizeBuf: [resize]u8 = undefined;
    const ok = c.stbir_resize_uint8_generic(
        grayBuf.ptr, image.width,
        image.height,
        0,
        resizeBuf[0..].ptr,
        @intCast(resizeWidth),
        @intCast(resizeHeight),
        0,
        1,
        -1,
        0,
        c.STBIR_EDGE_CLAMP,
        c.STBIR_FILTER_TRIANGLE,
        c.STBIR_COLORSPACE_LINEAR,
        null
    );
    if (ok == 0) {
        return Error.ResizeFailed;
    }
    defer image.freeMemory();
    defer allocator.free(grayBuf);

    // apply threholding to each pixel value after resizing
    // and accumulate a hash based on the results
    var hash: u64 = 0;
    for (0..resizeHeight) |x| {
        for (0..resizeHeight) |y| {
            const curIdx = x * resizeWidth + y;

            // from index get the current value and the next
            // value in that row
            // because iteration is only through the height,
            // no element from the 9th column will ever be a
            // current value
            const curVal: isize = resizeBuf[curIdx];
            const nextVal: isize = resizeBuf[curIdx + 1];

            // the left shit if calculated and applied only
            // if the current value is larger than the next
            const lsl: u6 = @intCast(curIdx - x);
            if (curVal - nextVal >= 0) {
                hash |= @as(u64, 1) << lsl;
            }
        }
    }
    return ImageHash{.hashType = "difference", .hash = hash};
}

// implementation of the perceptual hash algorithm
pub fn perceptualHash(filename: []const u8) Error!ImageHash {
    const allocator = std.heap.page_allocator;

    const size = 32;
    const channels = 3;
    const resize = size * size;

    var image = try loadImage(filename, channels);

    // allocate a buffer for the grayscaled image
    const grayBuf = try allocator.alloc(u8, image.imageSize());
    rgb2Gray(image.image, grayBuf, channels);

    // define an array and resize the grayscaled image
    var resizeBuf: [resize]u8 = undefined;
    const ok = c.stbir_resize_uint8_generic(
        grayBuf.ptr,
        image.width, 
        image.height,
        0,
        resizeBuf[0..].ptr,
        @intCast(size),
        @intCast(size),
        0,
        1,
        -1,
        0,
        c.STBIR_EDGE_CLAMP,
        c.STBIR_FILTER_TRIANGLE,
        c.STBIR_COLORSPACE_LINEAR,
        null
    );
    if (ok == 0) {
        return Error.ResizeFailed;
    }
    defer image.freeMemory();
    defer allocator.free(grayBuf);

    // define two arrays for intermediate and final states
    var temp: [resize]f32 = undefined;
    var dct: [resize]f32 = undefined;
    for (0..resizeBuf.len) |i| {
        dct[i] = @floatFromInt(resizeBuf[i]);
    }

    // precomputing the 1D basis matrix
    var basis: [resize]f32 = undefined;
    for (0..size) |u| {
        for (0..size) |x| {
            const uf: f32 = @floatFromInt(u);
            const xf: f32 = @floatFromInt(x);
            const sf: f32 = @floatFromInt(size);
            const preCos: f32 = ((2 * xf + 1) * uf * std.math.pi) / (2 * sf);
            var alpha: f32 = undefined;
            if (u == 0) {
                alpha = 1.0 / std.math.sqrt(sf);
            } else {
                alpha = std.math.sqrt(2.0 / sf);
            }
            const factor: f32 = alpha * std.math.cos(preCos);
            basis[u * size + x] = factor;
        }
    }

    // applying a row-wise DCT pass
    for (0..size) |x| {
        for (0..size) |u| {
            var colSum: f32 = 0;
            for (0..size) |y| {
                colSum += dct[x * size + y] * basis[u * size + y];
            }
            temp[x * size + u] = colSum;
        }
    }

    // applying a column-wise DCT pass
    for (0..size) |u| {
        for (0..size) |v| {
            var rowSum: f32 = 0;
            for (0..size) |x| {
                rowSum += temp[x * size + u] * basis[v * size + x];
            }
            dct[v * size + u] = rowSum;
        }
    }

    // define a new array for calculating the median
    // only extract the top left 8 x 8 from the DCT
    const llSize: usize = 8;
    const newSize: usize = llSize * llSize;
    var llBuf: [newSize]f32 = undefined;
    extractLL(dct[0..], llBuf[0..], llSize, size);

    const dctMedian = calcMedian(llBuf[0..], newSize);
    const intHash = hashFromMedian(llBuf[0..], dctMedian);
    return ImageHash{.hashType = "perceptual", .hash = intHash};
}

// implementation of the wavelet hash algorithm
pub fn waveletHash(filename: []const u8) Error!ImageHash {
    const allocator = std.heap.page_allocator;

    const size = 64;
    const channels = 3;
    const resize = size * size;

    var image = try loadImage(filename, channels);

    // allocate a buffer for the grayscaled image
    const grayBuf = try allocator.alloc(u8, image.imageSize());
    rgb2Gray(image.image, grayBuf, channels);

    // define an array and resize the grayscaled image
    var resizeBuf: [resize]u8 = undefined;
    const ok = c.stbir_resize_uint8_generic(
        grayBuf.ptr,
        image.width,
        image.height,
        0,
        resizeBuf[0..].ptr,
        @intCast(size),
        @intCast(size),
        0,
        1,
        -1,
        0,
        c.STBIR_EDGE_CLAMP,
        c.STBIR_FILTER_TRIANGLE,
        c.STBIR_COLORSPACE_LINEAR,
        null
    );
    if (ok == 0) {
        return Error.ResizeFailed;
    }
    defer image.freeMemory();
    defer allocator.free(grayBuf);

    // define an array for the transformations
    var tBuf: [resize]f32 = undefined;
    for (0..resizeBuf.len) |i| {
        tBuf[i] = @floatFromInt(resizeBuf[i]);
    }

    var lvlSize: usize = size;
    var distBuf: [size]f32 = undefined;
    var tempBuf: []f32 = distBuf[0..lvlSize];

    var rowTransform = RowTransform{
        .size = 0, .t = tBuf[0..], .temp = tempBuf[0..]
    };
    var colTransform = ColumnTransform{
        .size = 0, .t = tBuf[0..], .temp = tempBuf[0..]
    };
    for (1..4) |_| {
        // row wavelet transform
        wavelet(RowTransform, &rowTransform, lvlSize);

        // column wavelet transform
        wavelet(ColumnTransform, &colTransform, lvlSize);
        lvlSize /= 2;
        tempBuf = distBuf[0..lvlSize];
    }

    // define a new array for calculating the median
    // only extract the top left 8 x 8 from the wavelet
    const llSize: usize = 8;
    const newSize: usize = llSize * llSize;
    var llBuf: [newSize]f32 = undefined;
    extractLL(tBuf[0..], llBuf[0..], llSize, size);

    const waveMedian = calcMedian(llBuf[0..], newSize);
    const intHash = hashFromMedian(llBuf[0..], waveMedian);
    return ImageHash{.hashType = "wavelet", .hash = intHash};
}

test "image hash equals" {
    const tests = [_]struct {
        path: []const u8,
        hashType: []const u8,
        hash: u64,
        hexHash: []const u8,
        hasher: *const fn ([]const u8) Error!ImageHash,
    }{
        // tests for the average hash algorithm
        .{
            .path = "./testdata/checkerboard.png",
            .hashType = "average",
            .hash = 2990062961267801748,
            .hexHash = "297ed66bd66b7e94",
            .hasher = averageHash,
        },
        .{
            .path = "./testdata/gradient.png",
            .hashType = "average",
            .hash = 2197615328739459072,
            .hexHash = "1e7f7f7f7e780000",
            .hasher = averageHash,
        },
        .{
            .path = "./testdata/noise.png",
            .hashType = "average",
            .hash = 6821913422469120,
            .hexHash = "183c7e7e3c1800",
            .hasher = averageHash,
        },

        // tests for the difference hash algorithm
        .{
            .path = "./testdata/checkerboard.png",
            .hashType = "difference",
            .hash = 15826956251609881124,
            .hexHash = "dba4a4db24dada24",
            .hasher = differenceHash,
        },
        .{
            .path = "./testdata/gradient.png",
            .hashType = "difference",
            .hash = 18228586548031439040,
            .hexHash = "fcf8f2e2e0e0c0c0",
            .hasher = differenceHash,
        },
        .{
            .path = "./testdata/noise.png",
            .hashType = "difference",
            .hash = 12858323363238572534,
            .hexHash = "b271f0f8f8f0f1f6",
            .hasher = differenceHash,
        },

        // tests for the wavelet hash algorithm
        .{
            .path = "./testdata/checkerboard.png",
            .hashType = "wavelet",
            .hash = 18446673704965373696,
            .hexHash = "ffffbfffffffff00",
            .hasher = waveletHash,
        },
        .{
            .path = "./testdata/gradient.png",
            .hashType = "wavelet",
            .hash = 18374401693019275264,
            .hexHash = "fefefcfcf0c00000",
            .hasher = waveletHash,
        },
        .{
            .path = "./testdata/noise.png",
            .hashType = "wavelet",
            .hash = 15670382578127328000,
            .hexHash = "d97861edf7bfd300",
            .hasher = waveletHash,
        },

        // tests for the perceptual hash algorithm
        .{
            .path = "./testdata/checkerboard.png",
            .hashType = "perceptual",
            .hash = 7770000005721920801,
            .hexHash = "6bd495d685d69521",
            .hasher = perceptualHash,
        },
        .{
            .path = "./testdata/gradient.png",
            .hashType = "perceptual",
            .hash = 11769193724227288235,
            .hexHash = "a35493561952fcab",
            .hasher = perceptualHash,
        },
        .{
            .path = "./testdata/noise.png",
            .hashType = "perceptual",
            .hash = 16427238378350324179,
            .hexHash = "e3f94645163c29d3",
            .hasher = perceptualHash,
        },
    };

    for (tests) |hashTest| {
        const hash = try hashTest.hasher(hashTest.path);
        try std.testing.expectEqual(hashTest.hashType, hash.hashType);
        try std.testing.expectEqual(hashTest.hash, hash.hash);

        var hexBuf: [16]u8 = undefined;
        const hexDigest = try hash.hexDigest(hexBuf[0..]);
        try std.testing.expectEqualSlices(u8, hashTest.hexHash[0..], hexDigest);
    }
}
