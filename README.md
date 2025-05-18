<p align="center">
  <img src="/docs/logo.png" alt="imagehash logo" width="75%"/>
</p>

ImageHash is a Zig package for generating robust image fingerprints using four popular algorithms: **average (ahash)**, **difference (dhash)**, **perceptual (phash)**, and **wavelet (whash)** hashing. Perfect for image deduplication, similarity detection, and quick image comparisons.

## ✨ **Features**

* **Multiple Hash Algorithms:**

  * **ahash** (Average Hash)
  * **dhash** (Difference Hash)
  * **phash** (Perceptual Hash)
  * **whash** (Wavelet Hash)
* **Flexible Image Loading:** Supports common formats via `stb_image` and resizing via `stb_image_resize`.
* **Grayscale Conversion:** Uses luminosity formula for accurate luminance-based hashing.
* **JSON Interoperability:** Serialize hashes to/from JSON (`toJSON`, `fromJSON`).
* **Hamming Distance:** Compute bitwise Hamming distance between two hashes.

## 🚀 **Getting Started**

Clone the repo and add to your build.zig dependencies.

```bash
git clone https://github.com/galactixx/imagehash.git
```

Then in your build.zig, add:

```zig
const lib = b.addStaticLibrary(.{
    .name             = "mylib",
    .root_source_file = .{ .path = "src/lib.zig" },
    .target           = target,
    .optimize         = optimize,
});

lib.addPackagePath("imagehash", "../imagehash/src");
```

## 📚 **Usage**

```zig
const std = @import("std");
const imagehash = @import("imagehash");

pub fn main() !void {
    const file = "testdata/checkerboard.png";

    // Compute all four hashes
    const ahash = try imagehash.averageHash(file);
    const dhash = try imagehash.differenceHash(file);
    const phash = try imagehash.perceptualHash(file);
    const whash = try imagehash.waveletHash(file);

    // Print hex digests
    var buf: [16]u8 = undefined;
    std.debug.print("ahash: {s}\n", .{ahash.hexDigest(buf[0..])});
    std.debug.print("phash: {s}\n", .{phash.hexDigest(buf[0..])});

    // Compare two hashes
    const dist = ahash.distance(dhash);
    std.debug.print("Hamming distance: {}\n", .{dist});
}
```

## 🔍 **API**

### `pub fn averageHash(filename: []const u8) Error!ImageHash`

Computes an 8×8 average-based hash (`ahash`).

### `pub fn differenceHash(filename: []const u8) Error!ImageHash`

Computes a 9×8 horizontal difference-based hash (`dhash`).

### `pub fn perceptualHash(filename: []const u8) Error!ImageHash`

Computes a 32×32 perceptual (DCT-based) hash and reduces to an 8×8 block (`phash`).

### `pub fn waveletHash(filename: []const u8) Error!ImageHash`

Computes a 64×64 wavelet-based hash with 4-level decomposition (`whash`).

### `pub fn (self: ImageHash) distance(other: ImageHash) u64`

Returns the Hamming distance between two hashes.

### `pub fn (self: ImageHash) toJSON(alloc: *std.mem.Allocator) ![]u8`

Serialize an `ImageHash` to a JSON byte slice.

### `pub fn fromJSON(json: []const u8, alloc: std.mem.Allocator) ParseError!ImageHash`

Parse an `ImageHash` from JSON.

## 🤝 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📞 **Contact**

Have questions or need help? Open an issue on the [GitHub repository](https://github.com/yourusername/imagehash) or reach out via Twitter @yourhandle.
