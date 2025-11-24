from abc import ABC, abstractmethod
from typing import Tuple


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, file_path: str, expected_name: str) -> Tuple[bool, str, float]:
        pass
