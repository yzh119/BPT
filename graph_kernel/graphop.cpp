#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor maskedmm_csr_cuda_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& A,
    const at::Tensor& B);

at::Tensor maskedmm_csr_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& A,
    const at::Tensor& B) {
    CHECK_INPUT(row);
    CHECK_INPUT(indptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(indices);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    return maskedmm_csr_cuda_forward(row, indptr, eid, indices, A, B);
}

at::Tensor node_mul_edge_cuda_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& A,
    const at::Tensor& B);

at::Tensor node_mul_edge_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& A,
    const at::Tensor& B) {
    CHECK_INPUT(row);
    CHECK_INPUT(indptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    return node_mul_edge_cuda_forward(row, indptr, eid, A, B);
}

at::Tensor sparse_softmax_cuda_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& x);

at::Tensor sparse_softmax_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& x) {
    CHECK_INPUT(row);
    CHECK_INPUT(indptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(x);
    return sparse_softmax_cuda_forward(row, indptr, eid, x);
}

at::Tensor vector_spmm_cuda_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& edata,
    const at::Tensor& x);

at::Tensor vector_spmm_forward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& edata,
    const at::Tensor& x) {
    CHECK_INPUT(row);
    CHECK_INPUT(indptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(indices);
    CHECK_INPUT(edata);
    CHECK_INPUT(x);
    return vector_spmm_cuda_forward(row, indptr, eid, indices, edata, x);
}

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
    const at::Tensor& dy);

std::vector<at::Tensor> maskedmm_csr_backward(
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
    CHECK_INPUT(row);
    CHECK_INPUT(indptr_r);
    CHECK_INPUT(eid_r);
    CHECK_INPUT(indices_r);
    CHECK_INPUT(col);
    CHECK_INPUT(indptr_c);
    CHECK_INPUT(eid_c);
    CHECK_INPUT(indices_c);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    return maskedmm_csr_cuda_backward(row, indptr_r, eid_r, indices_r, col, indptr_c, eid_c, indices_c, A, B, dy);
}

std::vector<at::Tensor> node_mul_edge_cuda_backward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy);

std::vector<at::Tensor> node_mul_edge_backward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy) {
    CHECK_INPUT(row);
    CHECK_INPUT(indptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    return node_mul_edge_cuda_backward(row, indptr, eid, A, B, dy);
}

at::Tensor sparse_softmax_cuda_backward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& y,
    const at::Tensor& dy);

at::Tensor sparse_softmax_backward(
    const at::Tensor& row,
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& y,
    const at::Tensor& dy) {
    CHECK_INPUT(row);
    CHECK_INPUT(indptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(y);
    CHECK_INPUT(dy);
    return sparse_softmax_cuda_backward(row, indptr, eid, y, dy);
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
    const at::Tensor& x);

std::vector<at::Tensor> vector_spmm_backward(
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
    CHECK_INPUT(row);
    CHECK_INPUT(indptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(indices);
    CHECK_INPUT(col);
    CHECK_INPUT(indptr_t);
    CHECK_INPUT(eid_t);
    CHECK_INPUT(indices_t);
    CHECK_INPUT(edata);
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    return vector_spmm_cuda_backward(row, indptr, eid, indices, col, indptr_t, eid_t, indices_t, edata, dy, x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maskedmm_csr_forward", &maskedmm_csr_forward, "Masked Matrix Multiplication forward(CSR Format)");
    m.def("maskedmm_csr_backward", &maskedmm_csr_backward, "Masked Matrix Multiplication backward(CSR Format)");
    m.def("node_mul_edge_forward", &node_mul_edge_forward, "Node Multiply Edge forward");
    m.def("node_mul_edge_backward", &node_mul_edge_backward, "Node Multiply Edge backward");
    m.def("sparse_softmax_forward", &sparse_softmax_forward, "Sparse softmax forward");
    m.def("sparse_softmax_backward", &sparse_softmax_backward, "Sparse softmax backward");
    m.def("vector_spmm_forward", &vector_spmm_forward, "Vectorized SPMM forward");
    m.def("vector_spmm_backward", &vector_spmm_backward, "Vectorized SPMM backward");
}
