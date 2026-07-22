fn main() {
    println!("cargo:rerun-if-changed=cuda/miner.cu");
    if std::env::var_os("MIDSTATE_SKIP_CUDA").is_some() {
        return;
    }

    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .file("cuda/miner.cu")
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-Xptxas=-v")
        .flag("-std=c++17")
        .flag("-gencode=arch=compute_61,code=sm_61")
        .flag("-gencode=arch=compute_86,code=sm_86")
        .flag("-gencode=arch=compute_89,code=sm_89")
        .flag("-gencode=arch=compute_120,code=sm_120")
        .flag("-gencode=arch=compute_120,code=compute_120")
        .compile("midstate_native_cuda");
}
