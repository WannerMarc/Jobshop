from abc import ABCMeta, abstractmethod


class SchedulingData(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass


class JobShopSchedulingData(SchedulingData):
    def __init__(self, M, P, filename):
        self._M = M
        self._P = P
        self._filename = filename

    @property
    def get_M(self):
        return self._M

    @property
    def get_P(self):
        return self._P

    @property
    def get_filename(self):
        return self._filename