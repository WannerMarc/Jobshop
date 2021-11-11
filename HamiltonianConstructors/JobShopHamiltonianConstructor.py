from abc import ABCMeta
import sys
sys.path.append('..')
from HamiltonianConstructor import HamiltonianConstructor
from data import JobShopSchedulingData
import numpy as np
from pyqubo.array import Array

#Variable Guide:
#J: Number of jobs
#m: Number of machines
#T: Possible times of operations starting are in [0, T) \subset N
#M: Jxm matrix with machine index for every operation, i.e. job i operation o runs on machine M[i,o]
#P: Jxm matrix with processing time for every operation, i.e. job i operation o takes P[i,o] time steps to complete
#m_: running index for machines
#alpha, beta, gamma, eta: Biases for constraints/objective
#X: J*m x T matrix of qbit variables

class JobShopHamiltonianConstructor(HamiltonianConstructor):
    def __init__(self):
        super().__init__()

    def construct_hamiltonian(self, scheduling_data: JobShopSchedulingData, time_span: int, order_bias: float = 1,
                              machine_bias: float = 1, single_run_bias: float = 1, objective_bias: float = 1,
                              variable_pruning: bool = False):
        self._data = scheduling_data
        self._J, self._m = self._data.get_M.shape
        self._T = time_span
        self._X = Array.create("X_", shape=(self._m*self._J, self._T), vartype="BINARY")
        self._M = self._data.get_M
        self._P = self._data.get_P
        self._labels = [get_label(idx, t) for idx, t in np.ndindex(self._m*self._J, self._T)]

        H = order_bias*self.order_constraint() \
            + machine_bias*self.machine_constraint() \
            + single_run_bias*self.start_once_constraint() \
            + objective_bias*self.objective()

        self._bqm = H.compile().to_bqm()

        if variable_pruning:
            self.simple_variable_pruning()

        print(self._bqm.to_ising())

        self._hamiltonian = self._bqm.to_numpy_matrix(variable_order=self._labels)
        print(self._hamiltonian)

    def get_plottable_solution(self, solution_sampleset, energy_rank=0):
        plottable_solution = np.zeros((self._J, self._m, self._T), dtype=np.int)
        solution_sample = solution_sampleset.samples()[energy_rank]
        for i in range(len(self._labels)):
            idx = self.label_to_3d_idx(self._labels[i])
            if solution_sample[i] == 1:
                for k in range(self._P[idx[0], idx[1]]):
                    plottable_solution[idx[0], idx[1], idx[2] + k] = 1
        return plottable_solution

    def two_d_idx(self, idx):
        return (int(idx / self._m), idx % self._m)

    def label_to_3d_idx(self, label):
        rest, indices = label.split('_')
        index1, index2 = indices[1:-1].split('][')
        two_d_indices = self.two_d_idx(int(index1))
        return int(two_d_indices[0]), int(two_d_indices[1]), int(index2)

    def order_constraint(self):
        H = 0
        cnt = 0
        # h1
        for n in range(self._J):
            for i in range(n * self._m, (n + 1) * self._m - 1):  # I assume this is what k_(n-1)<i<k_n does
                for t in range(self._T):
                    for t_d in range(t + self._P[self.two_d_idx(i)]):
                        if t_d >= self._T:
                            break
                        H += self._X[i, t] * self._X[i + 1, t_d]
                        cnt += 1
        return H


    def machine_constraint(self):
        H = 0
        # construct index map
        I = [[] for i in range(self._m)]
        for i, o in np.ndindex((self._J, self._m)):
            I[self._M[i, o]].append(i * self._m + o)

        for m_ in range(self._m):
            Rm = self.get_Rm(I, m_)
            for i, t, k, t_d in Rm:
                H += self._X[i, t] * self._X[k, t_d]

        return H

    def start_once_constraint(self):
        H = 0
        for i in range(self._m * self._J):
            H += (np.sum([self._X[i, t] for t in range(self._T)]) - 1) ** 2

        return H

    def objective(self):
        H = 0
        for i in range(self._J):
            idx = self._m * (i + 1) - 1
            for t in range(self._T):
                H += self.lin_f(t) * self._X[idx, t]

        return H

    def get_Rm(self, I, m_):
        return self.get_Am(I, m_) + self.get_Bm(I, m_)

    def get_Am(self, I, m_):
        # Am: naive ugly looping
        Am = []
        for i, k in np.ndindex(self._J, self._J):
            if i == k:
                continue
            for t in range(self._T - 1):
                for t_d in range(t + 1, t + self._P[self.two_d_idx(I[m_][i])]):
                    if t_d >= self._T:
                        break
                    Am.append((I[m_][i], t, I[m_][k], t_d))
        return Am

    def get_Bm(self, I, m_):
        Bm = []
        for k in range(1, self._J):
            for i in range(k):
                for t in range(self._T):
                    Bm.append((I[m_][i], t, I[m_][k], t))
        return Bm

    def lin_f(self, t):
        return t / self._T

    def simple_variable_pruning(self):
        for i in range(self._J):
            temp_sum = 0
            job_sum = np.sum([self._P[i, o] for o in range(self._m)])
            for o in range(self._m):
                idx = i * self._m + o
                for t in range(temp_sum):
                    self._bqm.fix_variable(get_label(idx, t), 0)
                    self._labels.remove(get_label(idx, t))
                # include also that it can only be one at the start bit, so exclude also the pio-1 bits after the last possible start bit
                for t in range(self._T - (job_sum - temp_sum - 1), self._T):
                    self._bqm.fix_variable(get_label(idx, t), 0)
                    self._labels.remove(get_label(idx, t))
                temp_sum += self._P[i, o]

    def plottable_solution_to_pruned(self, plottable_solution):
        pruned = np.zeros(len(self._labels))
        for i in range(len(self._labels)):
            pruned[i] = plottable_solution[self.label_to_3d_idx(self._labels[i])]
        return pruned

    def get_name(self):
        return "JOBSHOP_HAMILTONIAN_CONSTRUCTOR"


def get_label(op_idx, t):
    return "X_[" + str(op_idx) + "][" + str(t) + "]"



