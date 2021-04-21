import random
from collections import defaultdict, deque
from multiprocessing.managers import BaseManager
from typing import Dict, List

import numpy as np

__all__ = ['MemPool', 'MemPoolManager']


class MemPool:
    def __init__(self, capacity: int = None, keys: List[str] = None) -> None:
        self._keys = keys
        if keys is None:
            self.data = defaultdict(lambda: deque(maxlen=capacity))
        else:
            self.data = {key: deque(maxlen=capacity) for key in keys}

    def push(self, data: Dict[str, np.ndarray]) -> None:
        """Push data into memory pool"""
        for key, value in data.items():
            self.data[key].extend(value)

        if self._keys is None:
            self._keys = list(self.data.keys())

    def sample(self, size: int = -1) -> Dict[str, np.ndarray]:
        """
        Sample training data from memory pool
        :param size: The number of sample data, default '-1' that indicates all data
        :return: The sampled and concatenated training data
        """

        num = len(self)
        indices = list(range(num))
        if 0 < size < num:
            indices = random.sample(indices, size)

        result = {}
        for key in self._keys:
            result[key] = np.stack([self.data[key][idx] for idx in indices])
        return result

    def clear(self) -> None:
        """Clear all data"""
        for key in self._keys:
            self.data[key].clear()

    def __len__(self):
        if self._keys is None:
            return 0
        return len(self.data[self._keys[0]])


class MemPoolManager(BaseManager):
    pass


MemPoolManager.register('MemPool', MemPool, exposed=['__len__', 'push', 'sample', 'clear'])