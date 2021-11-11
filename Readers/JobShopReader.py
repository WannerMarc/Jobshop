import sys
import numpy as np
sys.path.append('..')
from Reader import Reader
from data import JobShopSchedulingData


class JobShopReader(Reader):
    def read_problem_data(self, path: str):
        file = open(path)
        dims = [int(x) for x in file.readline().split(' ')]
        times = np.zeros((dims[0], dims[1]), dtype=np.int)
        sequences = np.zeros((dims[0], dims[1]), dtype=np.int)
        cnt = 0
        for line in file.readlines()[0:dims[0]]:
            numbers = [int(x) for x in line.split()]
            sequences[cnt, :] = numbers[::2]
            times[cnt, :] = numbers[1::2]
            cnt += 1

        self._data = JobShopSchedulingData(sequences, times, path)

