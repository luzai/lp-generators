''' Randomly generate lhs matrices by varying degree and coefficient
statistics. Main function to be called is generate_lhs.

This process is a bit of a bottleneck in generation, since it is implemented
in pure python and uses a quadratic algorithm to achieve the required
expected frequencies of edges. This is fine for small scale experimental
purposes, but further work needed (equivalent efficient algorithm and/or C++
implementation) before using this for larger instances.
'''

import itertools
import collections
import operator

import numpy as np
import scipy.sparse as sparsemat
from numba.typed import List
from numba import njit, prange 

@njit
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
    

@njit
def degree_dist(vertices:int, edges:int, max_degree:int, param:float):
    ''' Gives degree values for :vertices vertices.
    For each iteration, deterministic weights are given by the current vertex
    degree, random weights are chosen uniformly.
    The final weights for the next choice are weighted by :param on [0, 1],
    after being normalised by their sum.
    Vertices are excluded once they reach :max_degree, so :edges can be at
    most :vertices x :max_degree '''
    if (edges > vertices * max_degree):
        raise ValueError('edges > vertices * max_degree')
    degree = [0] * vertices
    indices = np.arange(vertices)
    for _ in range(edges):
        deterministic_weights = np.array([degree[i] + 0.0001 for i in indices])
        random_weights = np.random.uniform(0, 1, len(indices))
        weights = (
            deterministic_weights / deterministic_weights.sum() * param +
            random_weights / random_weights.sum() * (1 - param))
        ind = rand_choice_nb(indices, weights)
        degree[ind] += 1
        if degree[ind] >= max_degree:
            indices=indices[indices!=ind]
    return degree

@njit 
def expected_bipartite_degree(degree1, degree2):
    # Generates edges with probability d1 * d2 / sum(d1), asserting that
    # sum(d1) = sum(d2).
    # There is a more efficient way, right?
    if abs(sum(degree1) - sum(degree2)) > 10 ** -5:
        raise ValueError('You\'ve unbalanced the force!')
    rho = 1 / sum(degree1)
    res=[]
    for i, di in enumerate(degree1):
        for j, dj in enumerate(degree2):
            if np.random.uniform(0, 1) < (di * dj * rho):
                res.append((i,j))
    return res

@njit
def generate_by_degree(n1, n2, density, p1, p2):
    ''' Join together two vertex distributions to create a bipartite graph. '''
    nedges = max(int(round(n1 * n2 * density)), 1)
    degree1 = List(degree_dist(n1, nedges, n2, p1))
    degree2 = List(degree_dist(n2, nedges, n1, p2))
    return expected_bipartite_degree(degree1, degree2)

@njit
def fill_res(missing1, missing2, degree1, degree2):
    n1,n2=degree1.shape[0], degree2.shape[0]
    m1,m2=len(missing1),len(missing2) 
    mx,mn = max(m1,m2), min(m1,m2) 
    res=np.empty((mx, 2), dtype=np.uint32) 
    res[:m1,0] = missing1
    res[:m2,1] = missing2 
    degree1[missing1]+=1 
    degree2[missing2]+=1 
    if m1<m2:
        for idx, v2 in enumerate(missing2[mn:]): 
            cand=np.where(degree1<n2)[0]
            v1 = np.random.choice(cand)
            res[mn+idx, 0] = v1 
            degree1[v1]+=1 
    elif m1>m2:
        for idx, v1 in enumerate(missing1[mn:]):
            cand=np.where(degree2<n1)[0] 
            v2 = np.random.choice(cand)
            res[mn+idx, 1] = v2 
            degree2[v2]+=1 
    return res 

def connect_remaining(n1, n2, edges):
    ''' Finds any isolated vertices in the bipartite graph and connects them. '''
    # edges = np.array(List(edges), dtype=int) 
    idx1, degree1_=np.unique(edges[:,0], return_counts=True)
    idx2, degree2_=np.unique(edges[:,1], return_counts=True)
    degree1 = np.zeros(n1); degree1[idx1]=degree1_ 
    degree2 = np.zeros(n2); degree2[idx2]=degree2_
    missing1 = np.setdiff1d(np.arange(n1), idx1) 
    missing2 = np.setdiff1d(np.arange(n2), idx2) 
    np.random.shuffle(missing1)
    np.random.shuffle(missing2)
    return fill_res(missing1, missing2, degree1, degree2) 

def generate_edges(n1, n2, density, p1, p2):
    ''' Generate edges using size and weight parameters. '''
    edges = np.array(generate_by_degree(n1, n2, density, p1, p2), dtype=int) 
    # assert edges.shape == np.unique(edges, axis=0).shape
    # print('\n --> nedges & nremains:', edges.shape, )
    remains = connect_remaining(n1, n2, edges) 
    # remains = np.array(list(remains))
    print('\n --> nedges & nremains:', edges.shape, remains.shape)
    if remains.shape[0]!=0:
        edges = np.concatenate((edges, remains),
                    axis=0)     
    # assert edges.shape == np.unique(edges, axis=0).shape
    return edges 

@njit
def trunct_normal(loc, scale, size=1, bound_mult=2, eps=1e-1): 
    res=np.empty(size, dtype=np.float)
    b_l = loc - scale * bound_mult 
    b_u = loc + scale * bound_mult 
    for i in prange(size):  
        while True: 
            s = np.random.normal(loc, scale) 
            if b_l <= s and s<=b_u and np.abs(s) > eps: 
                res[i]=s 
                break 
    return res 

def generate_lhs(variables, constraints, density, pv, pc,
                        coeff_loc, coeff_scale):
    ''' Generate lhs constraint matrix using sparsity parameters and
    coefficient value distribution. '''

    inds = generate_edges(
        variables, constraints, density,
        pv, pc) 
    ind_var, ind_cons = inds[:,0], inds[:,1] 
    data = trunct_normal(
        loc=coeff_loc, scale=coeff_scale, size=len(ind_var))
    return sparsemat.coo_matrix((data, (ind_cons, ind_var)))
