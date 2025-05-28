const std = @import("std");

pub fn build(b: *std.Build) void {
    const host = b.standardTargetOptions(.{});

    const target: std.Build.ResolvedTarget = if (
        host.result.os.tag == .windows and host.result.abi == .msvc)
        b.resolveTargetQuery(.{
        .cpu_arch = host.result.cpu.arch,
        .os_tag = .windows,
        .abi = .gnu,
    }) else host;

    const optimize = b.standardOptimizeOption(.{});
    
    const lib_mod = b.addModule("imagehash", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const stb = b.addObject(.{
        .name = "stb_impl",
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    stb.addCSourceFiles(.{
        .files = &.{"c_src/stb_impl.c"},
        .flags = &.{},
    });
    stb.addIncludePath(b.path("include/"));

    const lib = b.addLibrary(.{
        .name = "imagehash",
        .linkage = .static,
        .root_module = lib_mod,
    });
    lib.addIncludePath(b.path("include/"));
    lib.addObject(stb);
    b.installArtifact(lib);

    const img_tests = b.addTest(.{
        .root_source_file = b.path("src/imagehash.zig"),
    });
    img_tests.addIncludePath(b.path("include/"));
    img_tests.addObject(stb);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(img_tests).step);
}