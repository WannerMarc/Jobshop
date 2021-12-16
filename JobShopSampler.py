import numpy as np
import os
import time


class JobShopSampler():
    def __init__(self, path, nJobs=None, nMachines=None, pmin=None, pmax=None):
        self._path = path
        self._J = nJobs
        self._m = nMachines
        self._pmin = pmin
        self._pmax = pmax
        self._rng = np.random.default_rng(round(time.time() * 1000))

    def sample(self, nJobs=None, nMachines=None, pmin=None, pmax=None, nsamples = 1):
        path = self._path
        self._J = nnull_condition(self._J, nJobs)
        self._m = nnull_condition(self._m, nMachines)
        self._pmin = nnull_condition(self._pmin, pmin)
        self._pmax = nnull_condition(self._pmax, pmax)

        if nsamples > 1:
            dirname = "Samples"+str(self._J)+"x"+str(self._m)
            path += os.sep + dirname
            if not os.path.exists(path):
                os.makedirs(path)
            path += os.sep

        num_existing_samples = get_num_existing_samples(path, self._J, self._m)

        for nsample in range(nsamples):
            filename = path + "js" + str(self._J) + "x" + str(self._m) + "_" + str(num_existing_samples+nsample) \
                       + ".txt"
            M = self.sample_M()
            P = self.sample_P()

            file = open(filename, 'w+')
            problem = str(self._J) + " " + str(self._m) + " " + str(self._pmin) + " " + str(self._pmax) + "\n"

            for i in range(self._J):
                problem += "".join([str(M[i, o])+" "+str(P[i, o])+" " for o in range(self._m)]) + "\n"
            print(problem)
            file.write(problem)
            file.close()

    def sample_M(self):
        return np.array([self._rng.permutation(np.arange(self._m)) for i in range(self._J)])

    def sample_P(self):
        return self._rng.integers(self._pmin, self._pmax+1, size=(self._J, self._m))


def nnull_condition(a, b):
    if b is None:
        assert a is not None
        return a
    else:
        return b


def get_num_existing_samples(path, nJobs, nMachines):
    num_existing_samples = 0
    for file in os.listdir(path):
        if not (file.startswith("js") and file.endswith(".txt") and "x" in file and "_" in file):
            continue
        probsizes = file[2:-4].split("_")
        probsize = probsizes[0].split("x")
        probsize.append(probsizes[1])
        sample_num = int(probsize[2])
        if probsize[0:2] == [str(nJobs), str(nMachines)] and sample_num > num_existing_samples:
            num_existing_samples = sample_num

    return num_existing_samples

