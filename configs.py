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
