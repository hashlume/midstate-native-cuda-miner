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
        // Pascal: GTX 10-series, including GTX 1070/1080-class cards.
        .flag("-gencode=arch=compute_61,code=sm_61")
        // Volta / data-center cards.
        .flag("-gencode=arch=compute_70,code=sm_70")
        // Turing: RTX 20-series, Quadro RTX, GTX 16-series, CMP 30/40/50HX.
        .flag("-gencode=arch=compute_75,code=sm_75")
        // Ampere data-center.
        .flag("-gencode=arch=compute_80,code=sm_80")
        // Ampere consumer/workstation: RTX 30-series, RTX A5000/A6000, CMP 70/90HX.
        .flag("-gencode=arch=compute_86,code=sm_86")
        // Ada Lovelace: RTX 40-series, RTX 5000/6000 Ada.
        .flag("-gencode=arch=compute_89,code=sm_89")
        // Hopper data-center.
        .flag("-gencode=arch=compute_90,code=sm_90")
        // Blackwell: RTX 50-series / RTX PRO Blackwell workstation cards.
        .flag("-gencode=arch=compute_120,code=sm_120")
        .flag("-gencode=arch=compute_120,code=compute_120")
        .compile("midstate_native_cuda");
}
