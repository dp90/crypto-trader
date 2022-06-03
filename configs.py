from abc import ABC, abstractmethod
import os
from dataclasses import dataclass


class DirectoryConfig(object):
    ROOT = os.path.dirname(__file__)
    VISUALS = os.path.join(ROOT, 'visuals')
    IMAGES = os.path.join(VISUALS, 'images')
    LOGS = os.path.join(ROOT, 'logs')
    RESULTS = os.path.join(LOGS, 'results')
    MODELS = os.path.join(ROOT, 'models')
    DATA = os.path.join(ROOT, 'data')


@dataclass
class Mode(object):
    name: str
    filename: str
    n_scenarios: int


class IMode(ABC):   
    @classmethod  # Ordering of decorators is important
    @property
    @abstractmethod
    def TRAIN(cls):
        raise NotImplementedError

    @classmethod  # Ordering of decorators is important
    @property
    @abstractmethod
    def TEST(cls):
        raise NotImplementedError

    @classmethod  # Ordering of decorators is important
    @property
    @abstractmethod
    def DEVELOP(cls):
        raise NotImplementedError
