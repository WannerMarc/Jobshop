from Scheduler import JobShopScheduler, JobShopSchedulingData
from docplex.mp.model import Model
import numpy as np
import math


class CPLEXSolver(JobShopScheduler):
    def __init__(self, data: JobShopSchedulingData):
        super().__init__(data)
        self._solution = None
        self._model = None
        self._Tmin = None

    def solve(self):
        size = self._data.get_M.shape
        # convert p to dictionary for graph weights
        p = {(i, j): self._data.get_P[i, j] for i in range(size[0]) for j in range(size[1])}
        p[0] = 0
        p[1] = 0
        # initialize graph
        C = [((i, j), (i, j + 1)) for j in range(size[1] - 1) for i in range(size[0])]
        C.extend([(0, (i, 0)) for i in range(size[0])])
        C.extend([((i, size[1] - 1), 1) for i in range(size[0])])
        # helper for constructing D
        Machine_jobs = [[] for i in range(size[1])]
        for i in range(size[0]):
            for j in range(size[1]):
                Machine_jobs[self._data.get_M[i, j]].append((i, j))
        D = [(Machine_jobs[m][i], Machine_jobs[m][j]) for i in range(size[0]) for j in range(i + 1, size[0]) for m in
             range(size[1])]
        # initialize model and variables
        model = Model('JobShop')
        S = model.integer_var_matrix([i for i in range(size[0])], [i for i in range(size[1])])
        S[0] = model.integer_var()
        S[1] = model.integer_var()
        # objective function
        model.minimize(S[1])
        model.add_constraint(S[0] == 0)
        model.add_constraints(S[c[0]] + p[c[0]] <= S[c[1]] for c in C)
        for d in D:
            model.add_constraint(model.logical_or(S[d[0]] + p[d[0]] <= S[d[1]], S[d[1]] + p[d[1]] <= S[d[0]]))
        self._solution = model.solve()
        self._model = model

    def get_plottable_solution(self):
        assert self._solution is not None
        J, m = self._data.get_M.shape
        self._Tmin = math.ceil(self._solution.get_objective_value())
        self._plottable_solution = np.zeros((J, m, self._Tmin), dtype=np.int)
        i = 0
        o = 0
        for var in self._model.iter_integer_vars():
            start = int(var.solution_value)
            if o >= m:
                o = 0
                i += 1
            if i >= J:
                break
            for k in range(self._data.get_P[i, o]):
                self._plottable_solution[i, o, start+k] = 1
            o += 1
        print(self._plottable_solution)
        return self._plottable_solution

    def get_solver_name(self):
        return "CLPEX Solver"

    def get_Tmin(self):
        assert self._Tmin is not None
        return self._Tmin


