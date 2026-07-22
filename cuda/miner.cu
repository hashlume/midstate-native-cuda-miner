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
    int chains_per_thread;
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

__host__ __device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

__host__ __device__ __forceinline__ void gmix(
    uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t mx, uint32_t my) {
    a = a + b + mx; d = rotr32(d ^ a, 16);
    c = c + d;      b = rotr32(b ^ c, 12);
    a = a + b + my; d = rotr32(d ^ a, 8);
    c = c + d;      b = rotr32(b ^ c, 7);
}

__host__ __device__ __forceinline__ uint32_t load32(const uint8_t *p) {
    return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
        ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__host__ __device__ __forceinline__ void store32(uint8_t *p, uint32_t x) {
    p[0] = (uint8_t)x;
    p[1] = (uint8_t)(x >> 8);
    p[2] = (uint8_t)(x >> 16);
    p[3] = (uint8_t)(x >> 24);
}

__host__ __device__ __forceinline__ void blake3_compress_words(
    const uint32_t input[8],
    uint32_t word8,
    uint32_t word9,
    uint32_t block_len,
    uint32_t output[8]) {
    const uint32_t IV[8] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
    };
    const uint32_t m0 = input[0], m1 = input[1], m2 = input[2], m3 = input[3];
    const uint32_t m4 = input[4], m5 = input[5], m6 = input[6], m7 = input[7];
    const uint32_t m8 = word8, m9 = word9;
    const uint32_t z = 0;

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

    #define G(a,b,c,d,x,y) gmix(v[a], v[b], v[c], v[d], x, y)
    G(0,4,8,12,m0,m1);  G(1,5,9,13,m2,m3);  G(2,6,10,14,m4,m5); G(3,7,11,15,m6,m7);
    G(0,5,10,15,m8,m9); G(1,6,11,12,z,z);   G(2,7,8,13,z,z);    G(3,4,9,14,z,z);
    G(0,4,8,12,m2,m6);  G(1,5,9,13,m3,z);   G(2,6,10,14,m7,m0); G(3,7,11,15,m4,z);
    G(0,5,10,15,m1,z);  G(1,6,11,12,z,m5);  G(2,7,8,13,m9,z);   G(3,4,9,14,z,m8);
    G(0,4,8,12,m3,m4);  G(1,5,9,13,z,z);    G(2,6,10,14,z,m2); G(3,7,11,15,m7,z);
    G(0,5,10,15,m6,m5); G(1,6,11,12,m9,m0); G(2,7,8,13,z,z);   G(3,4,9,14,m8,m1);
    G(0,4,8,12,z,m7);   G(1,5,9,13,z,m9);   G(2,6,10,14,z,m3); G(3,7,11,15,z,z);
    G(0,5,10,15,m4,m0); G(1,6,11,12,z,m2);  G(2,7,8,13,m5,m8); G(3,4,9,14,m1,m6);
    G(0,4,8,12,z,z);    G(1,5,9,13,m9,z);   G(2,6,10,14,z,z);  G(3,7,11,15,z,m8);
    G(0,5,10,15,m7,m2); G(1,6,11,12,m5,m3); G(2,7,8,13,m0,m1); G(3,4,9,14,m6,m4);
    G(0,4,8,12,m9,z);   G(1,5,9,13,z,m5);   G(2,6,10,14,m8,z); G(3,7,11,15,z,m1);
    G(0,5,10,15,z,m3);  G(1,6,11,12,m0,z);  G(2,7,8,13,m2,m6); G(3,4,9,14,m4,m7);
    G(0,4,8,12,z,z);    G(1,5,9,13,m5,m0);  G(2,6,10,14,m1,m9);G(3,7,11,15,m8,m6);
    G(0,5,10,15,z,z);   G(1,6,11,12,m2,z);  G(2,7,8,13,m3,m4); G(3,4,9,14,m7,z);
    #undef G

    #pragma unroll
    for (int i = 0; i < 8; ++i) output[i] = v[i] ^ v[i + 8];
}

__host__ __device__ __forceinline__ void gmix2(
    uint32_t &a0, uint32_t &b0, uint32_t &c0, uint32_t &d0, uint32_t mx0, uint32_t my0,
    uint32_t &a1, uint32_t &b1, uint32_t &c1, uint32_t &d1, uint32_t mx1, uint32_t my1) {
    a0 = a0 + b0 + mx0; a1 = a1 + b1 + mx1;
    d0 = rotr32(d0 ^ a0, 16); d1 = rotr32(d1 ^ a1, 16);
    c0 = c0 + d0; c1 = c1 + d1;
    b0 = rotr32(b0 ^ c0, 12); b1 = rotr32(b1 ^ c1, 12);
    a0 = a0 + b0 + my0; a1 = a1 + b1 + my1;
    d0 = rotr32(d0 ^ a0, 8); d1 = rotr32(d1 ^ a1, 8);
    c0 = c0 + d0; c1 = c1 + d1;
    b0 = rotr32(b0 ^ c0, 7); b1 = rotr32(b1 ^ c1, 7);
}

__host__ __device__ __forceinline__ void blake3_compress2_words(
    const uint32_t input0[8], uint32_t word80, uint32_t word90,
    const uint32_t input1[8], uint32_t word81, uint32_t word91,
    uint32_t block_len, uint32_t output0[8], uint32_t output1[8]) {
    const uint32_t IV[8] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
    };
    const uint32_t a0=input0[0],a1=input0[1],a2=input0[2],a3=input0[3];
    const uint32_t a4=input0[4],a5=input0[5],a6=input0[6],a7=input0[7],a8=word80,a9=word90;
    const uint32_t b0=input1[0],b1=input1[1],b2=input1[2],b3=input1[3];
    const uint32_t b4=input1[4],b5=input1[5],b6=input1[6],b7=input1[7],b8=word81,b9=word91;
    const uint32_t z=0;
    uint32_t x[16], y[16];
    #pragma unroll
    for (int i=0;i<8;++i) { x[i]=IV[i];x[i+8]=IV[i];y[i]=IV[i];y[i+8]=IV[i]; }
    x[12]=y[12]=0; x[13]=y[13]=0; x[14]=y[14]=block_len; x[15]=y[15]=11u;
    #define G2(p,q,r,s,ma,mb,na,nb) gmix2(x[p],x[q],x[r],x[s],ma,mb,y[p],y[q],y[r],y[s],na,nb)
    G2(0,4,8,12,a0,a1,b0,b1);  G2(1,5,9,13,a2,a3,b2,b3);  G2(2,6,10,14,a4,a5,b4,b5); G2(3,7,11,15,a6,a7,b6,b7);
    G2(0,5,10,15,a8,a9,b8,b9); G2(1,6,11,12,z,z,z,z);     G2(2,7,8,13,z,z,z,z);       G2(3,4,9,14,z,z,z,z);
    G2(0,4,8,12,a2,a6,b2,b6);  G2(1,5,9,13,a3,z,b3,z);    G2(2,6,10,14,a7,a0,b7,b0); G2(3,7,11,15,a4,z,b4,z);
    G2(0,5,10,15,a1,z,b1,z);   G2(1,6,11,12,z,a5,z,b5);   G2(2,7,8,13,a9,z,b9,z);     G2(3,4,9,14,z,a8,z,b8);
    G2(0,4,8,12,a3,a4,b3,b4);  G2(1,5,9,13,z,z,z,z);      G2(2,6,10,14,z,a2,z,b2);   G2(3,7,11,15,a7,z,b7,z);
    G2(0,5,10,15,a6,a5,b6,b5); G2(1,6,11,12,a9,a0,b9,b0); G2(2,7,8,13,z,z,z,z);       G2(3,4,9,14,a8,a1,b8,b1);
    G2(0,4,8,12,z,a7,z,b7);    G2(1,5,9,13,z,a9,z,b9);    G2(2,6,10,14,z,a3,z,b3);   G2(3,7,11,15,z,z,z,z);
    G2(0,5,10,15,a4,a0,b4,b0); G2(1,6,11,12,z,a2,z,b2);   G2(2,7,8,13,a5,a8,b5,b8); G2(3,4,9,14,a1,a6,b1,b6);
    G2(0,4,8,12,z,z,z,z);      G2(1,5,9,13,a9,z,b9,z);    G2(2,6,10,14,z,z,z,z);     G2(3,7,11,15,z,a8,z,b8);
    G2(0,5,10,15,a7,a2,b7,b2); G2(1,6,11,12,a5,a3,b5,b3); G2(2,7,8,13,a0,a1,b0,b1); G2(3,4,9,14,a6,a4,b6,b4);
    G2(0,4,8,12,a9,z,b9,z);    G2(1,5,9,13,z,a5,z,b5);    G2(2,6,10,14,a8,z,b8,z);   G2(3,7,11,15,z,a1,z,b1);
    G2(0,5,10,15,z,a3,z,b3);   G2(1,6,11,12,a0,z,b0,z);   G2(2,7,8,13,a2,a6,b2,b6); G2(3,4,9,14,a4,a7,b4,b7);
    G2(0,4,8,12,z,z,z,z);      G2(1,5,9,13,a5,a0,b5,b0);  G2(2,6,10,14,a1,a9,b1,b9);G2(3,7,11,15,a8,a6,b8,b6);
    G2(0,5,10,15,z,z,z,z);     G2(1,6,11,12,a2,z,b2,z);   G2(2,7,8,13,a3,a4,b3,b4); G2(3,4,9,14,a7,z,b7,z);
    #undef G2
    #pragma unroll
    for(int i=0;i<8;++i){output0[i]=x[i]^x[i+8];output1[i]=y[i]^y[i+8];}
}

__host__ __device__ __forceinline__ void extension_hash_words(
    const uint8_t midstate[32],
    uint64_t nonce,
    uint32_t iterations,
    uint32_t state[8]) {
    uint32_t input[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) input[i] = load32(midstate + i * 4);
    blake3_compress_words(input, (uint32_t)nonce, (uint32_t)(nonce >> 32), 40, state);
    for (uint32_t i = 0; i < iterations; ++i) {
        blake3_compress_words(state, 0, 0, 32, state);
    }
}

__host__ __device__ __forceinline__ void extension_hash2_words(
    const uint8_t midstate[32], uint64_t nonce0, uint64_t nonce1,
    uint32_t iterations, uint32_t state0[8], uint32_t state1[8]) {
    uint32_t input[8];
    #pragma unroll
    for(int i=0;i<8;++i) input[i]=load32(midstate+i*4);
    blake3_compress2_words(input,(uint32_t)nonce0,(uint32_t)(nonce0>>32),
                           input,(uint32_t)nonce1,(uint32_t)(nonce1>>32),40,state0,state1);
    for(uint32_t i=0;i<iterations;++i)
        blake3_compress2_words(state0,0,0,state1,0,0,32,state0,state1);
}

__device__ __forceinline__ uint8_t state_byte(const uint32_t state[8], int index) {
    return (uint8_t)(state[index >> 2] >> ((index & 3) * 8));
}

__device__ __forceinline__ bool state_below_target(const uint32_t state[8], const uint8_t target[32]) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        uint8_t byte = state_byte(state, i);
        if (byte < target[i]) return true;
        if (byte > target[i]) return false;
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
        uint32_t state[8];
        extension_hash_words(midstate, nonce, iterations, state);
        if (state_below_target(state, target)) {
            uint32_t index = atomicAdd(&candidates->count, 1u);
            if (index < candidates->cap && index < MAX_CANDIDATES) {
                candidates->nonce[index] = nonce;
                #pragma unroll
                for (int j = 0; j < 32; ++j) candidates->hash[index][j] = state_byte(state, j);
            }
        }
    }
}

__global__ void mine_kernel_dual(
    const uint8_t *midstate,const uint8_t *target,uint64_t base,uint64_t count,
    uint32_t iterations,CandidateBuffer *candidates) {
    uint64_t gid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t groups=(count+1u)/2u;
    uint64_t stride=(uint64_t)gridDim.x*blockDim.x;
    for(uint64_t group=gid;group<groups;group+=stride){
        uint64_t offset=group*2u;
        uint32_t state0[8],state1[8];
        extension_hash2_words(midstate,base+offset,base+offset+1u,iterations,state0,state1);
        if(state_below_target(state0,target)){
            uint32_t index=atomicAdd(&candidates->count,1u);
            if(index<candidates->cap&&index<MAX_CANDIDATES){candidates->nonce[index]=base+offset;
                #pragma unroll
                for(int j=0;j<32;++j)candidates->hash[index][j]=state_byte(state0,j);}
        }
        if(offset+1u<count&&state_below_target(state1,target)){
            uint32_t index=atomicAdd(&candidates->count,1u);
            if(index<candidates->cap&&index<MAX_CANDIDATES){candidates->nonce[index]=base+offset+1u;
                #pragma unroll
                for(int j=0;j<32;++j)candidates->hash[index][j]=state_byte(state1,j);}
        }
    }
}

__global__ void hash_one_kernel(
    const uint8_t *midstate,
    uint64_t nonce,
    uint32_t iterations,
    uint8_t *output) {
    uint32_t state[8];
    extension_hash_words(midstate, nonce, iterations, state);
    for (int i = 0; i < 32; ++i) output[i] = state_byte(state, i);
}

__global__ void hash_pair_kernel(const uint8_t *midstate,uint64_t nonce,uint32_t iterations,uint8_t *output){
    uint32_t state0[8],state1[8];
    extension_hash2_words(midstate,nonce,nonce+1u,iterations,state0,state1);
    for(int i=0;i<32;++i){output[i]=state_byte(state0,i);output[32+i]=state_byte(state1,i);}
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
    int device, int blocks, int threads, int requested_chains,
    uint64_t batch, uint32_t iterations) {
    cudaError_t select_error = cudaSetDevice(device);
    if (select_error != cudaSuccess) {
        fail(select_error, "cudaSetDevice");
        return nullptr;
    }
    if (threads <= 0 || batch == 0) {
        snprintf(last_error, sizeof(last_error), "threads and batch must be positive");
        return nullptr;
    }
    cudaDeviceProp properties{};
    cudaError_t properties_error=cudaGetDeviceProperties(&properties,device);
    if(properties_error!=cudaSuccess){fail(properties_error,"cudaGetDeviceProperties");return nullptr;}
    if(requested_chains<0||requested_chains>2){
        snprintf(last_error,sizeof(last_error),"chains per thread must be 0, 1, or 2");
        return nullptr;
    }
    int chains_per_thread=requested_chains==0?(properties.major>=12?2:1):requested_chains;
    uint64_t useful_threads=(batch+(uint64_t)chains_per_thread-1)/(uint64_t)chains_per_thread;
    uint64_t useful_blocks=(useful_threads+(uint64_t)threads-1)/(uint64_t)threads;
    if(blocks<=0){
        int active=1;
        cudaError_t occupancy_error=chains_per_thread==2
            ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,mine_kernel_dual,threads,0)
            : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,mine_kernel,threads,0);
        if(occupancy_error!=cudaSuccess){fail(occupancy_error,"cudaOccupancyMaxActiveBlocksPerMultiprocessor");return nullptr;}
        if(active<1&&chains_per_thread==2){
            chains_per_thread=1;
            occupancy_error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,mine_kernel,threads,0);
            if(occupancy_error!=cudaSuccess){fail(occupancy_error,"scalar CUDA occupancy");return nullptr;}
        }
        blocks=properties.multiProcessorCount*active;
    }
    if((uint64_t)blocks>useful_blocks)blocks=(int)useful_blocks;
    if (blocks < 1) blocks = 1;
    MidstateCudaWorker *worker = new MidstateCudaWorker{};
    worker->device = device;
    worker->blocks = blocks;
    worker->threads = threads;
    worker->chains_per_thread = chains_per_thread;
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

extern "C" void midstate_cuda_reference_hash_pair(
    const uint8_t *midstate,uint64_t nonce,uint32_t iterations,uint8_t *output){
    uint32_t state0[8],state1[8];
    extension_hash2_words(midstate,nonce,nonce+1u,iterations,state0,state1);
    for(int i=0;i<8;++i){store32(output+i*4,state0[i]);store32(output+32+i*4,state1[i]);}
}

extern "C" int midstate_cuda_worker_config(MidstateCudaWorker *worker,int *blocks,int *threads,int *chains){
    if(!worker)return -1;
    if(blocks)*blocks=worker->blocks;if(threads)*threads=worker->threads;if(chains)*chains=worker->chains_per_thread;
    return 0;
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
    if(worker->chains_per_thread==2)
        mine_kernel_dual<<<worker->blocks,worker->threads>>>(worker->d_midstate,worker->d_target,base,worker->batch,worker->iterations,worker->d_candidates);
    else
        mine_kernel<<<worker->blocks,worker->threads>>>(worker->d_midstate,worker->d_target,base,worker->batch,worker->iterations,worker->d_candidates);
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

extern "C" int midstate_cuda_hash_pair(int device,const uint8_t *midstate,uint64_t nonce,uint32_t iterations,uint8_t *output){
    if(cudaSetDevice(device)!=cudaSuccess)return -1;
    uint8_t *d_midstate=nullptr,*d_output=nullptr;
    cudaError_t error=cudaMalloc(&d_midstate,32);
    if(error==cudaSuccess)error=cudaMalloc(&d_output,64);
    if(error==cudaSuccess)error=cudaMemcpy(d_midstate,midstate,32,cudaMemcpyHostToDevice);
    if(error==cudaSuccess){hash_pair_kernel<<<1,1>>>(d_midstate,nonce,iterations,d_output);error=cudaDeviceSynchronize();}
    if(error==cudaSuccess)error=cudaMemcpy(output,d_output,64,cudaMemcpyDeviceToHost);
    cudaFree(d_midstate);cudaFree(d_output);
    return error==cudaSuccess?0:fail(error,"CUDA pair self-test");
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
