import os
import argparse
import numpy as np
import pandas as pd

def generate_data_txt(args, path_input, path_output):
    file = open(path_input)
    rawdata = np.loadtxt(file, delimiter=',')
    times, num_nodes = rawdata.shape
    print("times: ", times, ", num_nodes: ", num_nodes)
    P = args.window
    h = args.horizon
    
    train_end = int(times * args.train_rate)
    val_end = int(times * (args.train_rate + args.val_rate))
    train_set = range(0, train_end)
    val_set = range(train_end, val_end)
    test_set = range(val_end, times)
        
    # train
    x_train, y_train = split_x_y(rawdata, train_set, P, h)
    # val
    x_val, y_val = split_x_y(rawdata, val_set, P, h)
    # test
    x_test, y_test = split_x_y(rawdata, test_set, P, h)

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    # Write the data into npz file.
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(os.path.join(path_output, "%s.npz" % cat), x=_x, y=_y)
        
def split_x_y(rawdata, idx_set, P, h):
    x, y = [], []
    samples = len(idx_set) - P - h + 1
    for i in range(samples):
        start = idx_set[i]
        endx = start + P
        endy = endx + h
        x.append(rawdata[start:endx,...])
        y.append(rawdata[endx:endy,...])
    x = np.stack(x, axis=0) # (samples, P, num_nodes)
    y = np.stack(y, axis=0) # (samples, h, num_nodes)
    
    return np.expand_dims(x, axis = -1), np.expand_dims(y, axis = -1)


def generate_data_h5(args, path_input, path_output):
    df = pd.read_hdf(path_input)
    x_offsets = np.arange(-11, 1, 1) # array([-11,-10,...,0])
    y_offsets = np.arange(1, 13, 1)  # array([1,2,...,12])
    
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    # train/val/test: 7/1/2
    num_samples = x.shape[0]
    num_train = round(num_samples * args.train_rate)
    num_val = round(num_samples * args.val_rate)
    num_test = num_samples - num_train - num_val

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(path_output, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1])
        )

def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)) 
        # (1,1,times)->copy->(1,num_nodes,times)->transpose->(times,num_nodes,1)
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))    # (times,num_nodes,7)
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)
    
    data = np.concatenate(data_list, axis=-1) # (times,num_nodes,2)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - max(y_offsets))
    for t in range(min_t, max_t): # times-11-12 = samples
        x_t = data[t + x_offsets, ...] # (12,num_nodes,2)
        y_t = data[t + y_offsets, ...] # (12,num_nodes,2)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y
        
def main(args):
    print("Generating training data:")
    print("Traffic:")
    generate_data_txt(args, "./data/traffic/traffic.txt", "./data/traffic/")
    print("Electricity:")
    generate_data_txt(args, "./data/electricity/electricity.txt", "./data/electricity/")
    print("Solar_AL:")
    generate_data_txt(args, "./data/solar_AL/solar_AL.txt", "./data/solar_AL/")
    print("Exchange_rate:")
    generate_data_txt(args, "./data/exchange_rate/exchange_rate.txt", "./data/exchange_rate/")
    print("METR-LA:")
    generate_data_h5(args, "./data/METR-LA/metr-la.h5", "./data/METR-LA/")
    print("PEMS-BAY:")
    generate_data_h5(args, "./data/PEMS-BAY/pems-bay.h5", "./data/PEMS-BAY/")
    print("Finish!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train_rate", type=float, default=0.7)
    parser.add_argument("--val_rate", type=float, default=0.1)    
    args = parser.parse_args()
    main(args)
