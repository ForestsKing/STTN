import pandas as pd


def get_data():
    value = pd.read_csv('./PEMSD7/V_25.csv', header=None).values
    adj = pd.read_csv('./PEMSD7/W_25.csv', header=None).values

    train_data = value[:int(0.6 * len(value)), :]
    valid_data = value[int(0.6 * len(value)):int(0.8 * len(value)), :]
    test_data = value[int(0.8 * len(value)):, :]

    return train_data, valid_data, test_data, adj
