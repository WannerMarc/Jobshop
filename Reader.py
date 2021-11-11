from abc import ABCMeta, abstractmethod
from data import SchedulingData


class Reader(metaclass=ABCMeta):
    def __init__(self):
        self._data: SchedulingData = None

    @abstractmethod
    def read_problem_data(self, path: str):
        pass

    def get_data(self):
        return self._data

from Readers.JobShopReader import JobShopReader