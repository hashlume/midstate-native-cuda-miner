#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static constexpr uint32_t MAX_CANDIDATES = 512;

struct CandidateBuffer {
    uint32_t count;
    uint32_t cap;
    uint64_t nonce[MAX_CANDIDATES];
    uint8_t hash[MAX_CANDIDATES][32];
};

struct MidstateCudaResult {
    uint32_t count;
    uint64_t checked;
    double elapsed_ms;
    uint64_t nonce[MAX_CANDIDATES];
    uint8_t hash[MAX_CANDIDATES][32];
};

struct MidstateCudaWorker {
    int device;
    int blocks;
    int threads;
    uint64_t batch;
    uint32_t iterations;
    uint8_t *d_midstate;
    uint8_t *d_target;
    CandidateBuffer *d_candidates;
    cudaEvent_t started;
    cudaEvent_t finished;
};

static thread_local char last_error[256] = "ok";

static int fail(cudaError_t error, const char *operation) {
    snprintf(last_error, sizeof(last_error), "%s: %s", operation, cudaGetErrorString(error));
    return -1;
}

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

__device__ __forceinline__ void gmix(
    uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t mx, uint32_t my) {
    a = a + b + mx; d = rotr32(d ^ a, 16);
    c = c + d;      b = rotr32(b ^ c, 12);
    a = a + b + my; d = rotr32(d ^ a, 8);
    c = c + d;      b = rotr32(b ^ c, 7);
}

__device__ __forceinline__ uint32_t load32(const uint8_t *p) {
    return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
        ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ __forceinline__ void store32(uint8_t *p, uint32_t x) {
    p[0] = (uint8_t)x;
    p[1] = (uint8_t)(x >> 8);
    p[2] = (uint8_t)(x >> 16);
    p[3] = (uint8_t)(x >> 24);
}

__device__ void blake3_oneblock_words(const uint32_t m[16], uint32_t block_len, uint8_t out[32]) {
    const uint32_t IV[8] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
    };
    const uint8_t S[7][16] = {
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
        {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
        {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
        {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
        {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
        {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13},
    };

    uint32_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = IV[i];
        v[i + 8] = IV[i];
    }
    v[12] = 0;
    v[13] = 0;
    v[14] = block_len;
    v[15] = 11u;

    #pragma unroll
    for (int round = 0; round < 7; ++round) {
        const uint8_t *s = S[round];
        gmix(v[0], v[4], v[8],  v[12], m[s[0]],  m[s[1]]);
        gmix(v[1], v[5], v[9],  v[13], m[s[2]],  m[s[3]]);
        gmix(v[2], v[6], v[10], v[14], m[s[4]],  m[s[5]]);
        gmix(v[3], v[7], v[11], v[15], m[s[6]],  m[s[7]]);
        gmix(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]]);
        gmix(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        gmix(v[2], v[7], v[8],  v[13], m[s[12]], m[s[13]]);
        gmix(v[3], v[4], v[9],  v[14], m[s[14]], m[s[15]]);
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) store32(out + i * 4, v[i] ^ v[i + 8]);
}

__device__ void blake3_40(const uint8_t midstate[32], uint64_t nonce, uint8_t out[32]) {
    uint32_t words[16] = {0};
    #pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = load32(midstate + i * 4);
    words[8] = (uint32_t)nonce;
    words[9] = (uint32_t)(nonce >> 32);
    blake3_oneblock_words(words, 40, out);
}

__device__ void blake3_32_inplace(uint8_t hash[32]) {
    uint32_t words[16] = {0};
    #pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = load32(hash + i * 4);
    blake3_oneblock_words(words, 32, hash);
}

__device__ __forceinline__ bool below_target(const uint8_t hash[32], const uint8_t target[32]) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return false;
}

__global__ void mine_kernel(
    const uint8_t *midstate,
    const uint8_t *target,
    uint64_t base,
    uint64_t count,
    uint32_t iterations,
    CandidateBuffer *candidates) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t offset = gid; offset < count; offset += stride) {
        uint64_t nonce = base + offset;
        uint8_t hash[32];
        blake3_40(midstate, nonce, hash);
        for (uint32_t i = 0; i < iterations; ++i) blake3_32_inplace(hash);
        if (below_target(hash, target)) {
            uint32_t index = atomicAdd(&candidates->count, 1u);
            if (index < candidates->cap && index < MAX_CANDIDATES) {
                candidates->nonce[index] = nonce;
                #pragma unroll
                for (int j = 0; j < 32; ++j) candidates->hash[index][j] = hash[j];
            }
        }
    }
}

__global__ void hash_one_kernel(
    const uint8_t *midstate,
    uint64_t nonce,
    uint32_t iterations,
    uint8_t *output) {
    uint8_t hash[32];
    blake3_40(midstate, nonce, hash);
    for (uint32_t i = 0; i < iterations; ++i) blake3_32_inplace(hash);
    for (int i = 0; i < 32; ++i) output[i] = hash[i];
}

extern "C" int midstate_cuda_device_count() {
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    return error == cudaSuccess ? count : fail(error, "cudaGetDeviceCount");
}

extern "C" int midstate_cuda_device_name(int device, char *output, uint32_t output_len) {
    cudaDeviceProp properties;
    cudaError_t error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return fail(error, "cudaGetDeviceProperties");
    if (output_len == 0) return -1;
    snprintf(output, output_len, "%s (cc %d.%d)", properties.name, properties.major, properties.minor);
    return 0;
}

extern "C" MidstateCudaWorker *midstate_cuda_worker_create(
    int device, int blocks, int threads, uint64_t batch, uint32_t iterations) {
    if (cudaSetDevice(device) != cudaSuccess) return nullptr;
    MidstateCudaWorker *worker = new MidstateCudaWorker{};
    worker->device = device;
    worker->blocks = blocks;
    worker->threads = threads;
    worker->batch = batch;
    worker->iterations = iterations;
    if (cudaMalloc(&worker->d_midstate, 32) != cudaSuccess ||
        cudaMalloc(&worker->d_target, 32) != cudaSuccess ||
        cudaMalloc(&worker->d_candidates, sizeof(CandidateBuffer)) != cudaSuccess ||
        cudaEventCreate(&worker->started) != cudaSuccess ||
        cudaEventCreate(&worker->finished) != cudaSuccess) {
        snprintf(last_error, sizeof(last_error), "CUDA worker allocation failed");
        delete worker;
        return nullptr;
    }
    return worker;
}

extern "C" int midstate_cuda_worker_set_job(
    MidstateCudaWorker *worker, const uint8_t *midstate, const uint8_t *target) {
    if (!worker) return -1;
    if (cudaSetDevice(worker->device) != cudaSuccess) return -1;
    cudaError_t error = cudaMemcpy(worker->d_midstate, midstate, 32, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return fail(error, "copy midstate");
    error = cudaMemcpy(worker->d_target, target, 32, cudaMemcpyHostToDevice);
    return error == cudaSuccess ? 0 : fail(error, "copy target");
}

extern "C" int midstate_cuda_worker_mine(
    MidstateCudaWorker *worker, uint64_t base, MidstateCudaResult *result) {
    if (!worker || !result) return -1;
    if (cudaSetDevice(worker->device) != cudaSuccess) return -1;
    CandidateBuffer empty{};
    empty.cap = MAX_CANDIDATES;
    cudaError_t error = cudaMemcpy(
        worker->d_candidates, &empty, sizeof(empty), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return fail(error, "clear candidates");

    cudaEventRecord(worker->started);
    mine_kernel<<<worker->blocks, worker->threads>>>(
        worker->d_midstate, worker->d_target, base, worker->batch,
        worker->iterations, worker->d_candidates);
    error = cudaGetLastError();
    if (error != cudaSuccess) return fail(error, "launch mine kernel");
    cudaEventRecord(worker->finished);
    error = cudaEventSynchronize(worker->finished);
    if (error != cudaSuccess) return fail(error, "wait for mine kernel");

    CandidateBuffer host{};
    error = cudaMemcpy(&host, worker->d_candidates, sizeof(host), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return fail(error, "copy candidates");
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, worker->started, worker->finished);

    result->count = host.count > MAX_CANDIDATES ? MAX_CANDIDATES : host.count;
    result->checked = worker->batch;
    result->elapsed_ms = elapsed_ms;
    for (uint32_t i = 0; i < result->count; ++i) {
        result->nonce[i] = host.nonce[i];
        memcpy(result->hash[i], host.hash[i], 32);
    }
    return 0;
}

extern "C" int midstate_cuda_hash_one(
    int device, const uint8_t *midstate, uint64_t nonce, uint32_t iterations, uint8_t *output) {
    if (cudaSetDevice(device) != cudaSuccess) return -1;
    uint8_t *d_midstate = nullptr;
    uint8_t *d_output = nullptr;
    cudaError_t error = cudaMalloc(&d_midstate, 32);
    if (error != cudaSuccess) return fail(error, "self-test midstate allocation");
    error = cudaMalloc(&d_output, 32);
    if (error == cudaSuccess) error = cudaMemcpy(d_midstate, midstate, 32, cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
        hash_one_kernel<<<1, 1>>>(d_midstate, nonce, iterations, d_output);
        error = cudaDeviceSynchronize();
    }
    if (error == cudaSuccess) error = cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
    cudaFree(d_midstate);
    cudaFree(d_output);
    return error == cudaSuccess ? 0 : fail(error, "CUDA self-test");
}

extern "C" void midstate_cuda_worker_destroy(MidstateCudaWorker *worker) {
    if (!worker) return;
    cudaSetDevice(worker->device);
    cudaFree(worker->d_midstate);
    cudaFree(worker->d_target);
    cudaFree(worker->d_candidates);
    cudaEventDestroy(worker->started);
    cudaEventDestroy(worker->finished);
    delete worker;
}

extern "C" const char *midstate_cuda_last_error() {
    return last_error;
}
