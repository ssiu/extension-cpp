#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp {

__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  int numel = a_contig.numel();
  muladd_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c, result_ptr);
  return result;
}

__global__ void mul_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx];
}

at::Tensor mymul_cuda(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  int numel = a_contig.numel();
  mul_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
  return result;
}

__global__ void add_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx];
}

void myadd_out_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  int numel = a_contig.numel();
  add_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
}


#define A(i,j) A[(i) * N + (j)]
#define B(i,j) B[(i) * N + (j)]
#define C(i,j) C[(i) * N + (j)]
#define sA(pointer, i,j) sA[(pointer)][((i) << 7) + (j)]
#define sB(pointer, i,j) sB[(pointer)][((i) << 7) + (j)]
#define TILE_WIDTH 128
#define BLOCK_WIDTH 8
#define FLOAT_4(pointer) reinterpret_cast<float4*>(&(pointer))[0]


__global__ __launch_bounds__(256)
void mm_new_8_kernel(float* A, float* B, float* C, int N){
    int thread_id = threadIdx.x;
    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int warp_row = (warp_id >> 1) << 5;
    int warp_col = (warp_id & 1) << 6;
    int thread_row = (lane_id >> 3) << 2;
    int thread_col = (lane_id & 7) << 2;


    int sA_row = thread_id >> 1;
    int sA_col = (thread_id & 1) << 2;

    int sB_row = thread_id >> 5;
    int sB_col = (thread_id & 31) << 2;


    int C_row = warp_row + thread_row;
    int C_col = warp_col + thread_col;


    A = &A((block_idx << 7), 0);
    B = &B(0, (block_idy << 7));
    C = &C((block_idx << 7), (block_idy << 7));

    __shared__ float sA[2][BLOCK_WIDTH * TILE_WIDTH];
    __shared__ float sB[2][BLOCK_WIDTH * TILE_WIDTH];


    float rA[4];
    float rB[4];

    float fA[8] = {};
    float fB[8] = {};

    float accum[64] = {};

    int shared_pointer = 0;
    // load first block
    FLOAT_4(rA) = FLOAT_4(A(sA_row, sA_col));
    FLOAT_4(rB) = FLOAT_4(B(sB_row, sB_col));
    #pragma unroll
    for (int i=0; i<4;i++){
        sA(shared_pointer, sA_col + i, sA_row) = rA[i];
    }

    FLOAT_4(sB(shared_pointer, sB_row, sB_col)) = FLOAT_4(rB);

    __syncthreads();

    A += BLOCK_WIDTH;
    B += BLOCK_WIDTH * N;

    for (int kBlock=0; kBlock<N/BLOCK_WIDTH; kBlock++){

        // load from gmem A, B for next block
        if (kBlock < N/BLOCK_WIDTH - 1) {
            FLOAT_4(rA) = FLOAT_4(A(sA_row, sA_col));
            FLOAT_4(rB) = FLOAT_4(B(sB_row, sB_col));
        }
        #pragma unroll
        for (int kFragment=0; kFragment<BLOCK_WIDTH; kFragment++) {
            // load from smem A, B
            FLOAT_4(fA[0]) = FLOAT_4(sA(shared_pointer, kFragment, C_row));
            FLOAT_4(fA[4]) = FLOAT_4(sA(shared_pointer, kFragment, C_row + 16));
            FLOAT_4(fB[0]) = FLOAT_4(sB(shared_pointer, kFragment, C_col));
            FLOAT_4(fB[4]) = FLOAT_4(sB(shared_pointer, kFragment, C_col + 32));
            // compute outer product
            #pragma unroll
            for (int i=0; i<8;i++){
                #pragma unroll
                for (int j=0; j<8; j++) {
                    accum[i*8+j] += fA[i] * fB[j];
                }
             }

        }

        // store to smem sA, sB for next block
        if (kBlock < N/BLOCK_WIDTH - 1) {


            //FLOAT_4(sA[sA_sOffset]) = FLOAT_4(rA);
            #pragma unroll
            for (int i=0; i<4;i++){
                sA(shared_pointer^1, sA_col + i, sA_row) = rA[i];
                //sA[shared_pointer^1][sA_sOffset + i*TILE_WIDTH] = rA[i];
            }

            FLOAT_4(sB(shared_pointer^1, sB_row, sB_col)) = FLOAT_4(rB);

            __syncthreads();

            A += BLOCK_WIDTH;
            B += BLOCK_WIDTH * N;

            shared_pointer ^= 1;
        }

    }

//    storeToGmem_5(accum, C, N, C_gOffset);

    // store to gmem C
    #pragma unroll
    for (int i=0;i<4;i++) {

        FLOAT_4(C(C_row + i, C_col)) = FLOAT_4(accum[i * 8]);
        FLOAT_4(C(C_row + i, C_col + 32)) = FLOAT_4(accum[i * 8 + 4]);
        FLOAT_4(C(C_row + i + 16, C_col)) = FLOAT_4(accum[(i+4) * 8]);
        FLOAT_4(C(C_row + i + 16, C_col + 32)) = FLOAT_4(accum[(i+4) * 8 + 4]);

    }
}


at::Tensor mm_new_8(const at::Tensor& a, const at::Tensor& b) {

  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor c = torch::empty(a_contig.sizes(), a_contig.options());
  float* a_ptr = a_contig.data_ptr<float>();
  float* b_ptr = b_contig.data_ptr<float>();
  float* c_ptr = c.data_ptr<float>();
  int N = a.size(0);
  dim3 gridDim_mm_new_8(N / TILE_WIDTH,N / TILE_WIDTH);
  dim3 blockDim_mm_new_8(256);

  mm_new_8_kernel<<<gridDim_mm_new_8, blockDim_mm_new_8>>>(a_ptr, b_ptr, c_ptr, N);

  return c;
}



// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
//   m.impl("mymuladd", &mymuladd_cuda);
//   m.impl("mymul", &mymul_cuda);
//   m.impl("myadd_out", &myadd_out_cuda);
  m.impl("mm_new_8", &mm_new_8);
}

}
