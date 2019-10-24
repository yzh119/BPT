import torch as th
from graphop import *
from torch.autograd import Function

def to_contiguous(args):
    def wrapper(func):
        return func(*[arg.contiguous() if th.is_tensor(arg) else arg for arg in args])
    return wrapper

class SparseSoftmax(Function):
    @staticmethod
    def forward(ctx, row, indptr, eid, x):
        y = sparse_softmax_forward(row, indptr, eid, x)
        ctx.save_for_backward(row, indptr, eid, y)
        return y

    @staticmethod
    def backward(ctx, dy):
        row, indptr, eid, y = ctx.saved_tensors
        return None, None, None, sparse_softmax_backward(row, indptr, eid, y, dy)


class MaskedMMCSR(Function):
    @staticmethod
    def forward(ctx, row, indptr_r, eid_r, indices_r, col, indptr_c, eid_c, indices_c, A, B):
        ctx.save_for_backward(row, indptr_r, eid_r, indices_r, col, indptr_c, eid_c, indices_c, A, B)
        return maskedmm_csr_forward(row, indptr_r, eid_r, indices_r, A, B)

    @staticmethod
    def backward(ctx, grad):
        row, indptr_r, eid_r, indices_r, col, indptr_c, eid_c, indices_c, A, B = ctx.saved_tensors
        dA, dB = maskedmm_csr_backward(row, indptr_r, eid_r, indices_r, col, indptr_c, eid_c, indices_c, A, B, grad)
        return None, None, None, None, None, None, None, None, dA, dB

class NodeMulEdge(Function):
    @staticmethod
    def forward(ctx, row, indptr, eid, A, B):
        ctx.save_for_backward(row, indptr, eid, A, B)
        return node_mul_edge_forward(row, indptr, eid, A, B)

    @staticmethod
    def backward(ctx, grad):
        row, indptr, eid, A, B = ctx.saved_tensors
        dA, dB = node_mul_edge_backward(row, indptr, eid, A, B, grad)
        return None, None, None, dA, dB

class VectorSPMM(Function):
    @staticmethod
    def forward(ctx, row, indptr, eid, indices, col, ptr_t, eid_t, indices_t, edata, x):
        y = vector_spmm_forward(row, indptr, eid, indices, edata, x)
        ctx.save_for_backward(row, indptr, eid, indices, col, ptr_t, eid_t, indices_t, edata, x)
        return y

    @staticmethod
    def backward(ctx, dy):
        row, indptr, eid, indices, col, ptr_t, eid_t, indices_t, edata, x = ctx.saved_tensors
        dedata, dx = vector_spmm_backward(row, indptr, eid, indices, col, ptr_t, eid_t, indices_t, edata, dy, x)
        return None, None, None, None, None, None, None, None, dedata, dx

def sparse_softmax(row, indptr, eid, x):
    return SparseSoftmax.apply(row, indptr, eid, x)

def masked_mm(row, indptr_r, eid_r, indices_r, col, indptr_c, eid_c, indices_c, A, B):
    return MaskedMMCSR.apply(row, indptr_r, eid_r, indices_r, col, indptr_c, eid_c, indices_c, A, B)

def node_mul_edge(row, indptr, eid, A, B):
    return NodeMulEdge.apply(row, indptr, eid, A, B)

def vec_spmm(row, indptr, eid, indices, col, ptr_t, eid_t, indices_t, edata, x):
    return VectorSPMM.apply(row, indptr, eid, indices, col, ptr_t, eid_t, indices_t, edata, x)
