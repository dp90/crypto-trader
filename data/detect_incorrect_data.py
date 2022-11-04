import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

if __name__ == "__main__":
    currency = "XRP"
    COIN = pd.read_csv(os.path.join(DATA_DIR, currency + ".csv"))
    COIN["TIME"] = pd.to_datetime(COIN["TIME"])
    COIN["time_delta"] = COIN["TIME"].diff()
    COIN["INCORRECT_DELTA"] = COIN["time_delta"].apply(lambda time: time.seconds != 300)
    incorrect = COIN[COIN["INCORRECT_DELTA"]]
    incorrect.to_csv(os.path.join(DATA_DIR, "incorrect_" + currency + ".csv"), index=False)
