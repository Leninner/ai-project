from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class EmbeddingExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> Union[List[float], np.ndarray]:
        pass
