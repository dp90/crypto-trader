import time
from datetime import datetime, timedelta


class TimeConverter(object):
    def __init__(self, time_format, exchange):
        self.format = time_format
        self.multiplier = 1000 \
            if (exchange == "Bitvavo" or exchange == "Binance")\
            else 1

    def to_timestamp(self, date_string: str, time_string: str):
        date_time = date_string + " " + time_string
        return int(self.multiplier * time.mktime(datetime.strptime(date_time, self.format).timetuple()))

    def from_timestamp(self, timestamp):
        return datetime.fromtimestamp( timestamp / self.multiplier)

    def add_hours(self, start_time: int, n_hours: int):
        return start_time + n_hours * 3600 * self.multiplier
