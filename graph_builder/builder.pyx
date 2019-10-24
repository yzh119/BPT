#cython: boundscheck=False, wraparound=False, embedsignature=True, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport log2, ceil

cdef type DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t

"""
edges:
(src, dst, etype)

topdown/bottomup:
(eid)

etype:
TODO
"""

cdef class SegmentTree:
    cdef size_t length, n_nodes, n_lvl, n_edges, step, clip_dist
    cdef bool triu
    cdef vector[DTYPE_t] edges[3]
    cdef vector[DTYPE_t] n_nodes_arr, shift, pos

    def __cinit__(self, size_t length, bool triu=False, size_t step=1, size_t clip_dist=10):
        self.n_nodes = 0
        self.length = length
        self.triu = triu
        self.n_lvl = 1
        self.step = step
        self.clip_dist = clip_dist
        self.build_graph()

    def __reduce__(self):
        return SegmentTree, (self.length, self.triu, self.step, self.clip_dist)

    def build_graph(self):
        # count nodes and compute shift for each level
        cdef int n = self.length, i
        self.shift.push_back(0)
        while n >= 2:
            for i in range(n):
                self.pos.push_back((i + 1) * (1 << (self.n_lvl - 1)))
            self.n_nodes_arr.push_back(n)
            self.n_nodes += n
            self.shift.push_back(self.n_nodes)
            self.n_lvl += 1
            n = (n + 1) >> 1

        # add root
        self.n_nodes_arr.push_back(1)
        self.n_nodes += 1
        self.pos.push_back(self.length)

        for i in range(self.pos.size()):
            self.pos[i] = self.pos[i] - 1
            if self.pos[i] >= self.length:
                self.pos[i] = self.length - 1

        # add edges
        self.n_edges = 0
        cdef int j, v, l_ptr, r_ptr
        cdef bool l_overlap, r_overlap
        for i in range(self.length):
            v, shift = i, 0

            l_ptr, r_ptr = v - 1, v + 1
            l_overlap, r_overlap = False, False
            for j in range(self.n_lvl):
                # Add top down connection
                for k in range(self.step):
                    # right connection
                    if (not self.triu) and r_ptr < self.n_nodes_arr[j]:
                        if r_overlap:
                            self.edges[0].push_back(self.shift[j - 1] + ((r_ptr << 1) + 1))
                        else:
                            self.edges[0].push_back(self.shift[j] + r_ptr)
                        self.edges[1].push_back(i)
                        self.edges[2].push_back((2 * self.step + 1) * min(j, self.clip_dist) + 2 * (k + 1))
                        self.n_edges += 1
                        r_overlap = False

                    r_ptr += 1

                    # left connection
                    if l_ptr >= 0:
                        if l_overlap:
                            self.edges[0].push_back(self.shift[j - 1] + (l_ptr << 1))
                        else:
                            self.edges[0].push_back(self.shift[j] + l_ptr)
                        self.edges[1].push_back(i)
                        self.edges[2].push_back((2 * self.step + 1) * min(j, self.clip_dist) + 2 * k + 1)
                        self.n_edges += 1
                        l_overlap = False

                    l_ptr -= 1

                if l_ptr & 1 == 0:
                    l_overlap = True
                if r_ptr & 1 == 1:
                    r_overlap = True
                r_ptr >>= 1
                l_ptr >>= 1

                # Add self/bottom up connection
                self.edges[0].push_back(i)
                self.edges[1].push_back(shift + v)
                self.edges[2].push_back((2 * self.step + 1) * min(j, self.clip_dist))
                self.n_edges += 1

                shift += self.n_nodes_arr[j]
                v >>= 1

    @property
    def number_of_nodes(self):
        return self.n_nodes

    @property
    def number_of_edges(self):
        return self.n_edges

    @property
    def number_of_levels(self):
        return self.n_lvl

    @property
    def is_triu(self):
        return self.triu

    def get_pos(self):
        return np.asarray(self.pos)

    def get_edges(self, v_shift=0):
        return np.asarray(self.edges[0]) + v_shift,\
            np.asarray(self.edges[1]) + v_shift,\
            np.asarray(self.edges[2])

    def leaf_ids(self, v_shift=0, start=0):
        return np.arange(v_shift + start, v_shift + self.n_nodes_arr[0])

    def internal_ids(self, v_shift=0):
        return np.arange(v_shift + self.n_nodes_arr[0], v_shift + self.n_nodes)

    def root_id(self, v_shift=0):
        return v_shift + self.n_nodes - 1

    def number_of_nodes_at_lvl(self, i):
        return self.n_nodes_arr[i]

cdef class OpenAISparse:
    cdef size_t length, n_nodes, n_edges, stride, c
    cdef bool triu
    cdef vector[DTYPE_t] edges[3], pos
    
    def __cinit__(self, size_t length, size_t stride, size_t c, bool triu=True):
        self.n_nodes = length + 1
        self.length = length
        self.triu = triu
        self.stride = stride
        self.c = c
        assert triu, "bidirectional model is not supported"
        self.build_graph()
   
    def __reduce__(self):
        return OpenAISparse, (self.length, self.stride, self.c, self.triu)

    def build_graph(self):
        cdef size_t i, j, k
        for i in range(self.length):
            self.pos.push_back(i)
            # inside block
            for j in range((i // self.stride) * self.stride, i + 1):
                self.edges[0].push_back(j)
                self.edges[1].push_back(i)
                self.edges[2].push_back(0)
                self.n_edges += 1

            # outside block
            for j in range(1, i // self.stride + 1):
                for k in range(self.stride * j - self.c, self.stride * j):
                    self.edges[0].push_back(k)
                    self.edges[1].push_back(i)
                    self.edges[2].push_back(0)
                    self.n_edges += 1

            # bottomup edges
            self.edges[0].push_back(i)
            self.edges[1].push_back(self.length)
            self.edges[2].push_back(0)
            self.n_edges += 1

        self.pos.push_back(self.length - 1)

    @property
    def number_of_nodes(self):
        return self.n_nodes

    @property
    def number_of_edges(self):
        return self.n_edges

    @property
    def is_triu(self):
        return self.triu

    def get_pos(self):
        return np.asarray(self.pos)

    def get_edges(self, v_shift=0):
        return np.asarray(self.edges[0]) + v_shift,\
            np.asarray(self.edges[1]) + v_shift,\
            np.asarray(self.edges[2])

    def root_id(self, v_shift=0):
        return v_shift + self.n_nodes - 1

    def leaf_ids(self, v_shift=0, start=0):
        return np.arange(v_shift + start, v_shift + self.length)

    def internal_ids(self, v_shift=0):
        return np.arange(v_shift + self.length, v_shift + self.n_nodes)


cdef class FullyConnected:
    cdef size_t length, n_nodes, n_edges, window
    cdef bool triu
    cdef vector[DTYPE_t] edges[3], pos

    def __cinit__(self, size_t length, bool triu=False, size_t window=8192):
        self.n_nodes = length + 1
        self.length = length
        self.triu = triu
        self.window = window
        self.build_graph()

    def __reduce__(self):
        return FullyConnected, (self.length, self.triu, self.window)

    def build_graph(self):
        cdef size_t i, j
        for i in range(self.length):
            self.pos.push_back(i)
            # self loop
            self.edges[0].push_back(i)
            self.edges[1].push_back(i)
            self.edges[2].push_back(0)
            self.n_edges += 1

            # topdown edges
            for j in range(1, min(self.length, self.window + 1)):
                # left
                if i >= j:
                    self.edges[0].push_back(i - j)
                    self.edges[1].push_back(i)
                    self.edges[2].push_back(2 * j - 1)
                    self.n_edges += 1
                # right
                if not self.triu and i + j < self.length:
                    self.edges[0].push_back(i + j)
                    self.edges[1].push_back(i)
                    self.edges[2].push_back(2 * j)
                    self.n_edges += 1

            # bottomup edges
            self.edges[0].push_back(i)
            self.edges[1].push_back(self.length)
            self.edges[2].push_back(0)
            self.n_edges += 1

        self.pos.push_back(self.length)

    @property
    def number_of_nodes(self):
        return self.n_nodes

    @property
    def number_of_edges(self):
        return self.n_edges

    @property
    def is_triu(self):
        return self.triu

    def get_pos(self):
        return np.asarray(self.pos)

    def get_edges(self, v_shift=0):
        return np.asarray(self.edges[0]) + v_shift,\
            np.asarray(self.edges[1]) + v_shift,\
            np.asarray(self.edges[2])

    def root_id(self, v_shift=0):
        return v_shift + self.n_nodes - 1

    def leaf_ids(self, v_shift=0, start=0):
        return np.arange(v_shift + start, v_shift + self.length)

    def internal_ids(self, v_shift=0):
        return np.arange(v_shift + self.length, v_shift + self.n_nodes)

