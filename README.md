<p align="center">
  <img src="/docs/logo.png" alt="imagehash logo" width="75%"/>
</p>

ImageHash is a Zig package for generating robust image fingerprints using four popular algorithms: **average (ahash)**, **difference (dhash)**, **perceptual (phash)**, and **wavelet (whash)** hashing.

## ✨ **Features**

* **Multiple Hash Algorithms:**

  * **ahash** (Average Hash)
  * **dhash** (Difference Hash)
  * **phash** (Perceptual Hash)
  * **whash** (Wavelet Hash)
  * 
## 🚀 Getting Started

### Fetch via `zig fetch`

You can use the built‑in Zig fetcher to download and pin a tarball:

```bash
zig fetch --save git+https://github.com/galactixx/imagehash#v0.2.0
```

> This adds an `imagehash` entry under `.dependencies` in your `build.zig.zon`. 

Then in your build.zig:

```zig
const imagehash_mod = b.dependency("imagehash", .{
    .target = target,
    .optimize = optimize,
}).module("imagehash");

// add to library
lib_mod.addImport("imagehash", imagehash_mod);

// add to executable
exe.root_module.addImport("imagehash", imagehash_mod);
```

This lets you `const ih = @import("imagehash");` in your Zig code.

## 📚 **Usage**

```zig
const std = @import("std");
const ih = @import("imagehash");

pub fn main() !void {
    const file = "testdata/checkerboard.png";

    // Compute all four hashes
    const ahash = try ih.averageHash(file);
    const dhash = try ih.differenceHash(file);
    const phash = try ih.perceptualHash(file);
    const whash = try ih.waveletHash(file);

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

#### `pub fn averageHash(filename: []const u8) Error!ImageHash`

Computes an 8×8 average-based hash (`ahash`).

#### `pub fn differenceHash(filename: []const u8) Error!ImageHash`

Computes a 9×8 horizontal difference-based hash (`dhash`).
s
#### `pub fn perceptualHash(filename: []const u8) Error!ImageHash`

Computes a 32×32 perceptual (DCT-based) hash and reduces to an 8×8 block (`phash`).

#### `pub fn waveletHash(filename: []const u8) Error!ImageHash`

Computes a 64×64 wavelet-based hash with 4-level decomposition (`whash`).

#### `pub fn (self: ImageHash) distance(other: ImageHash) u64`

Returns the Hamming distance between two hashes.

#### `pub fn (self: ImageHash) toJSON(alloc: *std.mem.Allocator) ![]u8`

Serialize an `ImageHash` to a JSON byte slice.

#### `pub fn fromJSON(json: []const u8, alloc: std.mem.Allocator) ParseError!ImageHash`

Parse an `ImageHash` from JSON.

## 🤝 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📞 **Contact**

Have questions or need help? Open an issue on the [GitHub repository](https://github.com/galactixx/imagehash).
