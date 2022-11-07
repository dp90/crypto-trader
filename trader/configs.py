class TradingConfig:
    START_CAPITAL = 1000
    N_ASSETS = 5
    VARIABLES = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'N_TRADES']
    N_VARIABLES = len(VARIABLES)
    CURRENCY_ICS = [1, 2, 3]
    CURRENCIES = ['ADA', 'ATOM', 'BNB', 'BTC', 'ETH', 'LINK', 'LTC', 'TRX',
                  'VET', 'XLM', 'XRP']