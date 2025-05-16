const std = @import("std");

pub fn build(b: *std.Build) void {
    // ─────────────────────────────────────────────────────────────
    // 1. Resolve the effective target once.
    //    • Default to MinGW (`windows-gnu`) when the caller has not
    //      specified an ABI *and* the host OS is Windows.
    // ─────────────────────────────────────────────────────────────
    const host = b.standardTargetOptions(.{});

    const target: std.Build.ResolvedTarget = if (
        host.result.os.tag == .windows and host.result.abi == .msvc
    ) b.resolveTargetQuery(.{
        .cpu_arch = host.result.cpu.arch,
        .os_tag   = .windows,
        .abi      = .gnu,
    }) else host;

    const optimize = b.standardOptimizeOption(.{});

    // ─────────────────────────────────────────────────────────────
    // 2. Zig source modules
    // ─────────────────────────────────────────────────────────────
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // allow main.zig to `@import("imagehash_lib")`
    exe_mod.addImport("imagehash_lib", lib_mod);

    // ─────────────────────────────────────────────────────────────
    // 3. STB implementation object (C)
    // ─────────────────────────────────────────────────────────────
    const stb = b.addObject(.{
        .name   = "stb_impl",
        .target = target,
        .optimize = optimize,
        .link_libc  = true,
    });
    stb.addCSourceFiles(.{
        .files = &.{ "c_src/stb_impl.c" },
        .flags = &.{},
    });
    stb.addIncludePath(b.path("include/"));

    // ─────────────────────────────────────────────────────────────
    // 4. Static library that exposes the Zig API
    // ─────────────────────────────────────────────────────────────
    const lib = b.addLibrary(.{
        .name        = "imagehash",
        .linkage     = .static,
        .root_module = lib_mod,
    });
    lib.addIncludePath(b.path("include/"));
    b.installArtifact(lib);

    // ─────────────────────────────────────────────────────────────
    // 5. Command-line executable for quick testing
    // ─────────────────────────────────────────────────────────────
    const exe = b.addExecutable(.{
        .name        = "imagehash",
        .root_module = exe_mod,
    });
    exe.addIncludePath(b.path("include/"));
    exe.addObject(stb);
    b.installArtifact(exe);

    // run step: `zig build run -- <args>`
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the app").dependOn(&run_cmd.step);

    // ─────────────────────────────────────────────────────────────
    // 6. Unit tests
    // ─────────────────────────────────────────────────────────────
    const img_tests = b.addTest(.{
        .root_source_file  = b.path("src/imagehash.zig"),
    });
    img_tests.addIncludePath(b.path("include/"));
    img_tests.addObject(stb);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(img_tests).step);
}