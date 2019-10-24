import torch as th
from graphop import *
from torch.autograd import Function
from part_csr import partition_csr

chunk_size = 32

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

class MaskedMMSimple(Function):
    @staticmethod
    def forward(ctx, inc_x, inc_y, A, B):
        with th.no_grad():
            A_e = th.sparse.mm(inc_x.float(), A) # shape: (e, d)
            B_e = th.sparse.mm(inc_y.float(), B) # shape: (e, d)
            ctx.save_for_backward(A_e, B_e, inc_x, inc_y)
            y = (A_e * B_e).sum(-1) # shape: (e)
        assert y.requires_grad==False
        return y

    @staticmethod
    def backward(ctx, grad): # shape: (e)
        A_e, B_e, inc_x, inc_y = ctx.saved_tensors
        dAe = grad.unsqueeze(-1) * B_e
        dBe = grad.unsqueeze(-1) * A_e
        dA = th.sparse.mm(inc_x.t().float(), dAe)
        dB = th.sparse.mm(inc_y.t().float(), dBe)
        return None, None, dA, dB

if __name__ == '__main__':
    import os
    batch_size = 512
    l = 30
    n = batch_size * l
    e = batch_size * (l ** 2)
    v = th.ones(e, dtype=th.uint8)
    if not os.path.exists('i.pt'):
        i = th.zeros(2, e, dtype=th.long)
        eid_r = th.zeros(e, dtype=th.long)
        eid_c = th.zeros(e, dtype=th.long)
        indptr_r = th.zeros(n + 1, dtype=th.long)
        indptr_c = th.zeros(n + 1, dtype=th.long)
        indices_r = th.zeros(e, dtype=th.long)
        indices_c = th.zeros(e, dtype=th.long)    
        cnt = 0
        for b in range(batch_size):
            for x in range(b * l, (b + 1) * l):
                indptr_r[x] = cnt
                for y in range(b * l, (b + 1) * l):
                    i[0, cnt] = x
                    i[1, cnt] = y
                    indices_r[cnt] = y
                    eid_r[cnt] = cnt
                    cnt += 1
        indptr_r[n] = cnt

        cnt = 0
        for b in range(batch_size):
            for y in range(b * l, (b + 1) * l):
                indptr_c[y] = cnt
                for x in range(b * l, (b + 1) * l):
                    indices_c[cnt] = x
                    eid_c[cnt] = b * l * l + (x % l) * l + (y % l)
                    cnt += 1
        indptr_c[n] = cnt

        th.save((i, eid_r, eid_c, indptr_r, indptr_c, indices_r, indices_c), 'i.pt')
    else:
        i, eid_r, eid_c, indptr_r, indptr_c, indices_r, indices_c = th.load('i.pt')

    adj = th.sparse.ByteTensor(i, v, th.Size([n, n]))
    adj_1 = th.sparse.FloatTensor(i, th.rand(e), th.Size([n, n])).cuda(0).coalesce()
    adj_1.requires_grad_(True)

    if not os.path.exists('ix.pt'):
        i_x = th.zeros(2, e, dtype=th.long)
        i_y = th.zeros(2, e, dtype=th.long)
        cnt = 0
        for b in range(batch_size):
            for x in range(b * l, (b + 1) * l):
                for y in range(b * l, (b + 1) * l):
                    i_x[0, cnt] = cnt 
                    i_x[1, cnt] = x
                    i_y[0, cnt] = cnt 
                    i_y[1, cnt] = y
                    cnt += 1
        th.save((i_x, i_y), 'ixy.pt')
    else:
        i_x, i_y = th.load('ixy.pt')

    inc_x = th.sparse.ByteTensor(i_x, v, th.Size([e, n]))
    inc_y = th.sparse.ByteTensor(i_y, v, th.Size([e, n])) 

    import time
    inc_x = inc_x.cuda(0)
    inc_y = inc_y.cuda(0)
    adj = adj.cuda(0)
    eid_r, eid_c, indptr_r, indptr_c, indices_r, indices_c = eid_r.cuda(0), eid_c.cuda(0), indptr_r.cuda(0), indptr_c.cuda(0), indices_r.cuda(0), indices_c.cuda(0)
    th.cuda.synchronize()

    print('Single Head (batch size: 512, length: 30, dim: 1024)\n===========================================')
    print('MaskedNN(src_dot_dst)\nsimple implementation(copy to edge)')
    dim = 1024
    A = th.rand(n, dim, requires_grad=True, device='cuda:0')
    B = th.rand(n, dim, requires_grad=True, device='cuda:0')
    grad = th.rand(e, device='cuda:0')
    tic = time.time()
    A_e = th.sparse.mm(inc_x.float(), A)
    B_e = th.sparse.mm(inc_y.float(), B)
    y = (A_e * B_e).sum(-1)
    y_ori = y.clone()
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    A_grad_ori, B_grad_ori = A.grad.clone(), B.grad.clone()
    A.grad.zero_()
    B.grad.zero_()

    print('simple implementation, hand-crafted autograd')
    tic = time.time()
    y = MaskedMMSimple.apply(inc_x, inc_y, A, B)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y, y_ori)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad, A_grad_ori) and th.allclose(B.grad, B_grad_ori)
    A.grad.zero_()
    B.grad.zero_()

    print('vanilla bmm')
    tic = time.time()
    y = (A.view(batch_size, l, dim) @ B.view(batch_size, l, dim).transpose(-1, -2)).view(-1)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y, y_ori)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad, A_grad_ori) and th.allclose(B.grad, B_grad_ori)
    A.grad.zero_()
    B.grad.zero_()

    print('custom kernel(csr)')
    ROW, INDPTR_R = partition_csr(indptr_r, chunk_size=chunk_size)
    COL, INDPTR_C = partition_csr(indptr_c, chunk_size=chunk_size)
    tic = time.time()
    y = MaskedMMCSR.apply(ROW, INDPTR_R, eid_r, indices_r, COL, INDPTR_C, eid_c, indices_c, A, B)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y, y_ori)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad, A_grad_ori) and th.allclose(B.grad, B_grad_ori)

    # ------------------------------------------------------------------------
    # Test sparse softmax
    # ------------------------------------------------------------------------
    print('------------------------------------')
    print('vanilla softmax(scatter)')
    tic = time.time()
    x = th.rand(e, requires_grad=True, device='cuda:0')
    y = th.softmax(x.view(batch_size, l, l), -1).view(-1)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    y_ori = y.clone() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    x_grad_ori = x.grad.clone()
    x.grad.zero_()
    
    print('custom softmax(scatter)')
    tic = time.time()
    y = SparseSoftmax.apply(ROW, INDPTR_R, eid_r, x)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y_ori, y) 
    tic = time.time() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(x_grad_ori, x.grad, rtol=1e-3, atol=1e-6)
    x.grad.zero_()

    print('vanilla softmax(gather)')
    tic = time.time()
    x = th.rand(e, requires_grad=True, device='cuda:0')
    y = th.softmax(x.view(batch_size, l, l), -2).view(-1)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    y_ori = y.clone() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    x_grad_ori = x.grad.clone()
    x.grad.zero_()
    
    print('custom softmax(gather)')
    tic = time.time()
    y = SparseSoftmax.apply(COL, INDPTR_C, eid_c, x)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y_ori, y) 
    tic = time.time() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(x_grad_ori, x.grad, rtol=1e-3, atol=1e-6)
    x.grad.zero_()

    print('------------------------------------')
    print("spmm(pytorch coalesce)")
    A.grad.zero_()
    grad = th.rand(n, dim, device='cuda:0')
    tic = time.time()
    y = th.sparse.mm(adj_1, A)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    y_ori = y.clone()
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    A_grad_ori = A.grad.clone()
    adj_grad_ori = adj_1.grad._values()
    A.grad.zero_()
    adj_1.grad.zero_()

    print("vector-spmm(custom)")
    tic = time.time()
    val = adj_1._values()
    val.requires_grad_(True)
    y = VectorSPMM.apply(ROW, INDPTR_R, eid_r, indices_r, COL, INDPTR_C, eid_c, indices_c, val, A)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic)) 
    assert th.allclose(y_ori, y)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A_grad_ori, A.grad) and th.allclose(val.grad, adj_grad_ori)
    A.grad.zero_()
    val.grad.zero_()

    """
    Multi Head Test
    """
    print('\nMulti Head (batch size: 512, length: 30, head: 8, dim:64)\n===========================================')
    print('NodeMulEdge\nsimple implementation(copy to edge)')
    dim = 64
    h = 8
    A = th.rand(n, dim * h, requires_grad=True, device='cuda:0')
    B = th.rand(e, dim, requires_grad=True, device='cuda:0')
    grad = th.rand(e, h, device='cuda:0')
    tic = time.time()
    A_e = th.sparse.mm(inc_x.float(), A)
    y = (A_e.view(-1, h, dim) * B.view(-1, 1, dim)).sum(-1)
    y_ori = y.clone()
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    A_grad_ori, B_grad_ori = A.grad.clone(), B.grad.clone()
    A.grad.zero_()
    B.grad.zero_()
    
    print('custom kernel')
    tic = time.time()
    y = NodeMulEdge.apply(ROW, INDPTR_R, eid_r, A.view(-1, h, dim), B.view(-1, dim))
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y_ori, y)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A_grad_ori, A.grad) and th.allclose(B_grad_ori, B.grad)
    A.grad.zero_()
    B.grad.zero_()

    print('MaskedNN(src_dot_dst)\nsimple implementation(copy to edge)')
    dim = 64
    h = 8
    A = th.rand(n, dim * h, requires_grad=True, device='cuda:0')
    B = th.rand(n, dim * h, requires_grad=True, device='cuda:0')
    grad = th.rand(e, h, device='cuda:0')
    tic = time.time()
    A_e = th.sparse.mm(inc_x.float(), A)
    B_e = th.sparse.mm(inc_y.float(), B)
    y = (A_e.view(-1, h, dim) * B_e.view(-1, h, dim)).sum(-1)
    y_ori = y.clone()
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    A_grad_ori, B_grad_ori = A.grad.clone(), B.grad.clone()
    A.grad.zero_()
    B.grad.zero_()

    print('vanilla bmm')
    tic = time.time()
    y = (A.view(batch_size, l, h, dim).contiguous().transpose(1, 2) @ B.view(batch_size, l, h, dim).contiguous().permute(0, 2, 3, 1)).permute(0, 2, 3, 1).contiguous().view(-1, h)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y, y_ori)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad, A_grad_ori) and th.allclose(B.grad, B_grad_ori)
    A.grad.zero_()
    B.grad.zero_()

    print('custom kernel(csr)')
    tic = time.time()
    y = MaskedMMCSR.apply(ROW, INDPTR_R, eid_r, indices_r, COL, INDPTR_C, eid_c, indices_c, A.view(-1, h, dim), B.view(-1, h, dim))
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y, y_ori)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad, A_grad_ori) and th.allclose(B.grad, B_grad_ori)

    # ------------------------------------------------------------------------
    # Test sparse softmax
    # ------------------------------------------------------------------------
    print('------------------------------------')
    print('vanilla softmax(scatter)')
    tic = time.time()
    x = th.rand(e, h, requires_grad=True, device='cuda:0')
    y = th.softmax(x.view(batch_size, l, l, h), -2).view(-1, h)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    y_ori = y.clone() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    x_grad_ori = x.grad.clone()
    x.grad.zero_()
    
    print('custom softmax(scatter)')
    tic = time.time()
    y = SparseSoftmax.apply(ROW, INDPTR_R, eid_r, x)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y_ori, y) 
    tic = time.time() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(x_grad_ori, x.grad, rtol=1e-3, atol=1e-6)
    x.grad.zero_()

    print('vanilla softmax(gather)')
    tic = time.time()
    x = th.rand(e, h, requires_grad=True, device='cuda:0')
    y = th.softmax(x.view(batch_size, l, l, h), -3).view(-1, h)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    y_ori = y.clone() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    x_grad_ori = x.grad.clone()
    x.grad.zero_()
    
    print('custom softmax(gather)')
    tic = time.time()
    y = SparseSoftmax.apply(COL, INDPTR_C, eid_c, x)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y_ori, y) 
    tic = time.time() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(x_grad_ori, x.grad, rtol=1e-3, atol=1e-6)
    x.grad.zero_()

    adjs = []
    for index in range(8):
        adj_index = th.sparse.FloatTensor(i, th.rand(e), th.Size([n, n])).cuda(0).coalesce()
        adj_index.requires_grad_(True)
        adjs.append(adj_index)

    print('------------------------------------')
    print("spmm(pytorch coalesce)")
    A.grad.zero_()
    grad = [th.rand(n, dim, device='cuda:0') for _ in range(8)]
    tic = time.time()
    ys = []
    for index in range(8):
        ys.append(th.sparse.mm(adjs[index], A.view(n, h, dim)[:, index, :]))
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    y_ori = th.cat([y.clone().view(n, 1, dim) for y in ys], dim=-2)
    tic = time.time()
    for index in range(8):
        ys[index].backward(grad[index])
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    A_grad_ori = A.grad.clone()
    adj_grad_ori = th.cat([_.grad._values().view(e, 1) for _ in adjs], dim=-1)
    A.grad.zero_()
    for index in range(8):
        adjs[index].grad.zero_()

    print("vector-spmm(custom)")
    val = th.cat([_._values().view(-1, 1) for _ in adjs], dim=-1)
    val.requires_grad_(True)
    tic = time.time()
    y = VectorSPMM.apply(ROW, INDPTR_R, eid_r, indices_r, COL, INDPTR_C, eid_c, indices_c, val, A.view(n, h, dim))
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic)) 
    assert th.allclose(y_ori, y)
    tic = time.time()
    y.backward(th.cat([_.view(n, 1, dim) for _ in grad], dim=-2))
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
