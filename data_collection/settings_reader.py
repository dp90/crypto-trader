import json
from typing import Dict


class SettingsReader(object):
    @staticmethod
    def read(filepath: str) -> Dict[str, str]:
        with open(filepath, 'r') as file:
            settings = json.load(file)
        return settings
