#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include "atomic.cuh"
#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

/*
 * CUDA Kernel of the forward function for Node-Edge Multiplication(reduced on edge, designed for relative positional encoding).
 */
template <typename scalar_t>
__global__ void node_mul_edge_forward_kernel(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ y, const int d, const int n, const int h) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    if (i < n) {
        for (int j = indptr[i] + tx; j < indptr[i + 1]; j += blockDim.x) {
            for (int ko = 0; ko < h; ++ko) {
                scalar_t sum = 0;
                for (int ki = 0; ki < d; ++ki) {
                    sum += A[(row[i] * h + ko) * d + ki] * B[eid[j] * d + ki];
                }
                y[eid[j] * h + ko] = sum;
            }
        }
    }
}


/*
 * CUDA Kernel of the forward function for Masked Matrix Multiplication. (argument: csr format)
 */
template <typename scalar_t>
__global__ void maskedmm_csr_forward_kernel(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const int64_t* __restrict__ indices, const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ y, const int d, const int n, const int n_row, const int h) {
    int i = blockIdx.x; 
    int tx = threadIdx.x;
    if (i < n_row) {
        for (int j = indptr[i] + tx; j < indptr[i + 1]; j += blockDim.x) {
            for (int ko = 0; ko < h; ++ko) {
                scalar_t sum = 0;
                for (int ki = 0; ki < d; ++ki) {
                    sum += A[(row[i] * h + ko) * d + ki] * B[(ko * d + ki) * n + indices[j]];
                }
                y[eid[j] * h + ko] = sum;
            }
        }
    }
}


/*
 * CUDA Kernel of the backward function for Node-Edge Multiplication(reduced on edge, designed for relative positional encoding).
 */
template <typename scalar_t>
__global__ void node_mul_edge_backward_kernel_0(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const scalar_t* __restrict__ B, const scalar_t* __restrict__ dy, scalar_t* __restrict__ dA, const int d, const int n, const int h) {
    int tx = threadIdx.x;
    int i = blockIdx.x;
    if (i < n) {
        for (int j = tx; j < d * h; j += blockDim.x) {
            scalar_t sum = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k)
                sum += dy[eid[k] * h + j / d] * B[eid[k] * d + j % d];
            dgl::AtomicAdd(dA + row[i] * d * h + j, sum); 
        }
    }
}


/*
 * CUDA Kernel of the backward function for Node-Edge Multiplication(reduced on edge, designed for relative positional encoding).
 */
template <typename scalar_t>
__global__ void node_mul_edge_backward_kernel_1(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const scalar_t* __restrict__ A, const scalar_t* __restrict__ dy, scalar_t* __restrict__ dB, const int d, const int n, const int h) {
    int tx = threadIdx.x;
    int i = blockIdx.x;
    if (i < n) {
        for (int j = tx; j < d; j += blockDim.x) {
            for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
                scalar_t sum = 0;
                for (int ki = 0; ki < h; ++ki) {
                    sum += dy[eid[k] * h + ki] * A[(row[i] * h + ki) * d + j];
                }
                dB[eid[k] * d + j] = sum;
            }
        }
    }
}


/*
 * CUDA Kernel of the backward function for Masked Matrix Multiplication. (argument: csr format)
 */
template <typename scalar_t>
__global__ void maskedmm_csr_backward_kernel(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const int64_t* __restrict__ indices, const scalar_t* __restrict__ B, const scalar_t* __restrict__ dy, scalar_t* __restrict__ dA, const int d, const int n, const int h) {
    int tx = threadIdx.x;
    int i = blockIdx.x;
    if (i < n) {
        for (int j = tx; j < d * h; j += blockDim.x) {
            scalar_t sum = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k)
                sum += dy[eid[k] * h + j / d] * B[indices[k] * d * h + j];
            dgl::AtomicAdd(dA + row[i] * d * h + j, sum);
        }
    }
}

/*
 * CUDA Kernel of the forward function for Source Multiply Edge Function.
 * For `src_mul_edge` operation, the arguments are csr(column-major) representations.
 */
template <typename scalar_t>
__global__ void vector_spmm_forward_kernel(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const int64_t* __restrict__ indices, const scalar_t* __restrict__ edata, const scalar_t* __restrict__ x, scalar_t* __restrict__ y, const int d, const int n, const int h) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    if (i < n) {
        for (int j = tx; j < d * h; j += blockDim.x) {
            scalar_t sum = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k)
                sum += edata[eid[k] * h + j / d] * x[indices[k] * d * h + j];
            dgl::AtomicAdd(y + row[i] * d * h + j, sum);
        }
    }
}

/*
 * CUDA Kernel of the backward function for Source Multiply Edge Function.
 */
template <typename scalar_t>
__global__ void vector_spmm_backward_kernel_0(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const int64_t* __restrict__ indices, const scalar_t* __restrict__ dy, const scalar_t* __restrict__ xt, scalar_t* __restrict__ dedata, const int d, const int n, const int n_row, const int h) {
    int i = blockIdx.x; 
    int tx = threadIdx.x;
    if (i < n_row) {
        for (int j = indptr[i] + tx; j < indptr[i + 1]; j += blockDim.x)
            for (int ko = 0; ko < h; ++ko) {
                scalar_t sum = 0;
                for (int ki = 0; ki < d; ++ki) {
                    sum += dy[(row[i] * h + ko) * d + ki] * xt[(ko * d + ki) * n + indices[j]];
                }
                dedata[eid[j] * h + ko] = sum;
            }
    }
}

template <typename scalar_t>
__global__ void vector_spmm_backward_kernel_1(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const int64_t* __restrict__ indices, const scalar_t* __restrict__ edata, const scalar_t* __restrict__ dy, scalar_t* __restrict__ dx, const int d, const int n_row, const int h) {
    int i = blockIdx.x; 
    int tx = threadIdx.x;
    if (i < n_row) {
        for (int j = tx; j < d * h; j += blockDim.x) {
            scalar_t sum = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k)
                sum += edata[eid[k] * h + j / d] * dy[indices[k] * d * h + j];
            dgl::AtomicAdd(dx + row[i] * d * h + j, sum);
        }
    }
}

/*
 * CUDA Kernel of forward function for Sparse Softmax
 * y = softmax(x), grouped by node.
 * indptr, eid: csr format
 */
template <typename scalar_t>
__global__ void sparse_softmax_forward_kernel_max(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const scalar_t* __restrict__ x, scalar_t* __restrict__ max_val, const int n_row, const int h) {
    int i = blockIdx.x;
    int tx = threadIdx.x;        
    if (i < n_row) {
        for (int k = indptr[i]; k < indptr[i + 1]; ++k)
            dgl::AtomicMax(max_val + row[i] * h + tx, x[eid[k] * h + tx]);
    }
}

template <typename scalar_t>
__global__ void sparse_softmax_forward_kernel_minus_exp(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const scalar_t* __restrict__ x, const scalar_t* __restrict__ max_val, scalar_t* __restrict__ sum, scalar_t* __restrict__ y, const int n_row, const int h) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    if (i < n_row) {
        scalar_t max_v = max_val[row[i] * h + tx];
        for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
            scalar_t now = exp(x[eid[k] * h + tx] - max_v);
            y[eid[k] * h + tx] = now;
            dgl::AtomicAdd(sum + row[i] * h + tx, now);
        }
    }
}

template <typename scalar_t>
__global__ void sparse_softmax_forward_kernel_norm(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const scalar_t* __restrict__ sum, scalar_t* __restrict__ y, const int n_row, const int h) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    if (i < n_row) {
        for (int k = indptr[i]; k < indptr[i + 1]; ++k)
            y[eid[k] * h + tx] /= sum[row[i] * h + tx];
    }
}

/*
 * CUDA Kernel of backward function for Sparse Softmax.
 * indptr, eid: csr format
 */
template <typename scalar_t>
__global__ void sparse_softmax_backward_kernel_0(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const scalar_t* __restrict__ dy, const scalar_t* __restrict__ y, scalar_t* __restrict__ aggre, const int n_row, const int h) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    if (i < n_row) {
        scalar_t sum = 0;
        for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
            sum += dy[eid[k] * h + tx] * y[eid[k] * h + tx];
        } 
        dgl::AtomicAdd(aggre + row[i] * h + tx, sum);
    }
}

template <typename scalar_t>
__global__ void sparse_softmax_backward_kernel_1(const int64_t* __restrict__ row, const int64_t* __restrict__ indptr, const int64_t* __restrict__ eid, const scalar_t* __restrict__ dy, const scalar_t* __restrict__ y, const scalar_t* __restrict__ aggre, scalar_t* __restrict__ dx, const int n_row, const int h) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    if (i < n_row) {
        for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
            dx[eid[k] * h + tx] = dy[eid[k] * h + tx] * y[eid[k] * h + tx] - aggre[row[i] * h + tx] * y[eid[k] * h + tx] ;
        }
    } 
}

} // End of namespace


at::Tensor node_mul_edge_cuda_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& A,
    const at::Tensor& B) {
    // indptr: (n + 1); eid: (e); A: (n, d) or (n, h, d); B: (e, d);
    cudaSetDevice(indptr.get_device());

    const auto e = eid.size(0);
    const auto n = row.size(0);
    const auto d = A.size(-1);
    const auto h = (A.dim() == 2) ? 1: A.size(1);
    auto y = (h == 1) ? at::zeros({e}, A.options()): at::zeros({e, h}, A.options());

    const int threads = 32;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "node_mul_edge_cuda_forward", ([&] {
        node_mul_edge_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            y.data<scalar_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return y;
}

// __global__ void maskedmm_csr_forward_kernel(int64_t* __restrict__ indptr, int64_t* __restrict__ eid, int64_t* __restrict__ indices, scalar_t* __restrict__ A, scalar_t* __restrict__ B, scalar_t* __restrict__ y, int d, int n) {
at::Tensor maskedmm_csr_cuda_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& A,
    const at::Tensor& B) {
    // indptr: (n + 1); eid, indices: (e); A, B: (n, d) or (n, h, d); 
    cudaSetDevice(indptr.get_device());

    const auto e = eid.size(0);
    const auto n = A.size(0);
    const auto n_row = row.size(0);
    const auto d = A.size(-1);
    const auto h = (A.dim() == 2) ? 1: A.size(1);
    auto y = (h == 1) ? at::zeros({e}, A.options()): at::zeros({e, h}, A.options());

    const int threads = 32;
    const dim3 blocks(n_row);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto Bt = (B.dim() == 2) ? B.transpose(0, 1).contiguous(): B.permute({1, 2, 0}).contiguous();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "maskedmm_csr_cuda_forward", ([&] {
        maskedmm_csr_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            indices.data<int64_t>(),
            A.data<scalar_t>(),
            Bt.data<scalar_t>(),
            y.data<scalar_t>(),
            d, n, n_row, h);
    }));
    THCudaCheck(cudaGetLastError());
    return y;
}

std::vector<at::Tensor> node_mul_edge_cuda_backward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy) {
    // indptr: (n + 1); eid: (e); dy: (e) or (e, h); A: (n, d) or (n, h, d); B: (e, d)
    cudaSetDevice(indptr.get_device());

    const auto e = eid.size(0);
    const auto n = row.size(0);
    const auto d = A.size(-1);
    const auto h = (dy.dim() == 2) ? dy.size(1): 1;

    int threads = 128;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto dA = at::zeros_like(A, A.options());
    auto dB = at::zeros_like(B, B.options());

    AT_DISPATCH_FLOATING_TYPES(A.type(), "node_mul_edge_cuda_backward_0", ([&] {
        node_mul_edge_backward_kernel_0<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            B.data<scalar_t>(),
            dy.data<scalar_t>(),
            dA.data<scalar_t>(),
            d, n, h);
    }));
    threads = d;
    AT_DISPATCH_FLOATING_TYPES(A.type(), "node_mul_edge_cuda_backward_1", ([&] {
        node_mul_edge_backward_kernel_1<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            A.data<scalar_t>(),
            dy.data<scalar_t>(),
            dB.data<scalar_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return {dA, dB};
}


// __global__ void maskedmm_csr_backward_kernel(int64_t* __restrict__ indptr_r, int64_t* __restrict__ eid_r, int64_t* __restrict__ indices_r, int64_t* __restrict__ indptr_c, int64_t* __restrict__ eid_c, int64_t* __restrict__ indices_c, scalar_t* __restrict__ A, scalar_t* __restrict__ B, scalar_t* __restrict__ dy, scalar_t* __restrict__ dA, scalar_t* __restrict__ dB, int d, int n)
std::vector<at::Tensor> maskedmm_csr_cuda_backward(
    const at::Tensor& row,
    const at::Tensor& indptr_r,
    const at::Tensor& eid_r,
    const at::Tensor& indices_r,
    const at::Tensor& col,
    const at::Tensor& indptr_c,
    const at::Tensor& eid_c,
    const at::Tensor& indices_c,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy) {
    // indptr_r, indptr_c: (n + 1); eid_r, eid_c, indices_r, indices_c: (e); dy: (e) or (e, h); A, B: (n, d) or (n, h, d)
    cudaSetDevice(indptr_r.get_device());

    const auto e = eid_r.size(0);
    const auto n_row = row.size(0);
    const auto d = A.size(-1);
    const auto h = (dy.dim() == 2) ? dy.size(1): 1;

    const int threads = 128;
    const dim3 blocks_row(n_row);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto dA = at::zeros_like(A, A.options());
    auto dB = at::zeros_like(B, B.options());

    AT_DISPATCH_FLOATING_TYPES(B.type(), "maskedmm_csr_cuda_backward", ([&] {
        maskedmm_csr_backward_kernel<scalar_t><<<blocks_row, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr_r.data<int64_t>(),
            eid_r.data<int64_t>(),
            indices_r.data<int64_t>(),
            B.data<scalar_t>(),
            dy.data<scalar_t>(),
            dA.data<scalar_t>(),
            d, n_row, h);
    }));
    THCudaCheck(cudaGetLastError());

    const auto n_col = col.size(0);
    const dim3 blocks_col(n_col);
    AT_DISPATCH_FLOATING_TYPES(A.type(), "maskedmm_csr_cuda_backward", ([&] {
        maskedmm_csr_backward_kernel<scalar_t><<<blocks_col, threads, 0, stream>>>(
            col.data<int64_t>(),
            indptr_c.data<int64_t>(),
            eid_c.data<int64_t>(),
            indices_c.data<int64_t>(),
            A.data<scalar_t>(),
            dy.data<scalar_t>(),
            dB.data<scalar_t>(),
            d, n_col, h);
    }));
    return {dA, dB};
}

at::Tensor sparse_softmax_cuda_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& x) {
    cudaSetDevice(indptr.get_device());

    // indptr: (n + 1); eid: (e); x: (e) or (e, h);
    const auto n_row = row.size(0);
    const auto n = eid.size(0); // n <= e
    const auto h = (x.dim() == 2) ? x.size(1): 1;
    const dim3 threads(h);
    const dim3 blocks(n_row);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    auto sum = (h == 1) ? at::zeros({n}, x.options()): at::zeros({n, h}, x.options());
    auto max_val = (h == 1) ? at::zeros({n}, x.options()): at::zeros({n, h}, x.options());
    at::fill_(max_val, -1e9);
    const auto y = at::zeros_like(x, x.options());
    
    AT_DISPATCH_FLOATING_TYPES(x.type(), "sparse_softmax_cuda_forward_0",([&] {
        sparse_softmax_forward_kernel_max<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            x.data<scalar_t>(),
            max_val.data<scalar_t>(),
            n_row, h);
    }));
    AT_DISPATCH_FLOATING_TYPES(x.type(), "sparse_softmax_cuda_forward_1",([&] {
        sparse_softmax_forward_kernel_minus_exp<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            x.data<scalar_t>(),
            max_val.data<scalar_t>(),
            sum.data<scalar_t>(),
            y.data<scalar_t>(),
            n_row, h);
    }));
    AT_DISPATCH_FLOATING_TYPES(x.type(), "sparse_softmax_cuda_forward_2",([&] {
        sparse_softmax_forward_kernel_norm<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            sum.data<scalar_t>(),
            y.data<scalar_t>(),
            n_row, h);
    }));

    THCudaCheck(cudaGetLastError());
    return y;
}

at::Tensor sparse_softmax_cuda_backward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& y,
    const at::Tensor& dy) {
    cudaSetDevice(indptr.get_device());

    // indptr: (n + 1); eid: (e); y: (e) or (e, h); dy: (e) or (e, h);
    const auto n_row = row.size(0);
    const auto n = eid.size(0); // n <= e
    const auto h = (dy.dim() == 2) ? dy.size(1): 1;
    const dim3 threads(h);
    const dim3 blocks(n_row);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    auto aggre = (h == 1) ? at::zeros({n}, dy.options()): at::zeros({n, h}, dy.options());
    const auto dx = at::zeros_like(dy, dy.options());

    AT_DISPATCH_FLOATING_TYPES(y.type(), "sparse_softmax_cuda_backward_0", ([&] {
        sparse_softmax_backward_kernel_0<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            dy.data<scalar_t>(),
            y.data<scalar_t>(),
            aggre.data<scalar_t>(),
            n_row, h); 
    }));
    AT_DISPATCH_FLOATING_TYPES(y.type(), "sparse_softmax_cuda_backward_1", ([&] {
        sparse_softmax_backward_kernel_1<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            dy.data<scalar_t>(),
            y.data<scalar_t>(),
            aggre.data<scalar_t>(),
            dx.data<scalar_t>(),
            n_row, h); 
    }));
    THCudaCheck(cudaGetLastError());
    return dx;
}

at::Tensor vector_spmm_cuda_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& edata,
    const at::Tensor& x) {
    // indptr: (n + 1); eid, indices: (e); edata: (e) or (e, h); x: (n, d) or (n, h, d);
    cudaSetDevice(indptr.get_device());

    const auto n = row.size(0); 
    const auto h = (edata.dim() == 2) ? edata.size(1): 1;
    const auto d = x.size(-1); 
    
    const int threads = 32;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto y = at::zeros_like(x, x.options());
    
    AT_DISPATCH_FLOATING_TYPES(x.type(), "vector_spmm_forward", ([&] {
        vector_spmm_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            indices.data<int64_t>(),
            edata.data<scalar_t>(),
            x.data<scalar_t>(),
            y.data<scalar_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return y;
}

std::vector<at::Tensor> vector_spmm_cuda_backward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& col,
    const at::Tensor& indptr_t,
    const at::Tensor& eid_t,
    const at::Tensor& indices_t,
    const at::Tensor& edata,
    const at::Tensor& dy,
    const at::Tensor& x) {
    // indptr: (n + 1); eid, indices: (e); edata: (e) or (e, h); dy, x: (n, d) or (n, h, d); 
    cudaSetDevice(indptr.get_device());

    const auto n_row = row.size(0);
    const auto n_col = col.size(0);
    const auto n = x.size(0);
    const auto h = (edata.dim() == 2) ? edata.size(1): 1;
    const auto d = x.size(-1);

    int threads = 32;
    const dim3 blocks(n_row);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const auto xt = (h == 1) ? x.transpose(0, 1).contiguous(): x.permute({1, 2, 0}).contiguous();

    const auto dx = at::zeros_like(x, x.options());
    const auto dedata = at::zeros_like(edata, edata.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "vector_spmm_backward_0", ([&] {
        vector_spmm_backward_kernel_0<scalar_t><<<blocks, threads, 0, stream>>>(
            row.data<int64_t>(),
            indptr.data<int64_t>(),
            eid.data<int64_t>(),
            indices.data<int64_t>(),
            dy.data<scalar_t>(),
            xt.data<scalar_t>(),
            dedata.data<scalar_t>(),
            d, n, n_row, h);
    }));

    threads = 128;
    AT_DISPATCH_FLOATING_TYPES(x.type(), "vector_spmm_backward_1", ([&] {
        vector_spmm_backward_kernel_1<scalar_t><<<blocks, threads, 0, stream>>>(
            col.data<int64_t>(),
            indptr_t.data<int64_t>(),
            eid_t.data<int64_t>(),
            indices_t.data<int64_t>(),
            edata.data<scalar_t>(),
            dy.data<scalar_t>(),
            dx.data<scalar_t>(),
            d, n_col, h);
    }));
    THCudaCheck(cudaGetLastError());
    return {dedata, dx};
}
