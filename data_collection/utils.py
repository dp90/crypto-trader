import time
from datetime import datetime


class TimeConverter(object):
    def __init__(self, settings):
        self.format = settings["timeFormat"]
        self.multiplier = 1000 \
            if (settings["exchange"] == "Bitvavo" or settings["exchange"] == "Binance")\
            else 1

    def to_timestamp(self, date_string: str, time_string: str):
        date_time = date_string + " " + time_string
        return int(self.multiplier * time.mktime(datetime.strptime(date_time, self.format).timetuple()))

    def from_timestamp(self, timestamp):
        return datetime.fromtimestamp( timestamp / self.multiplier)
