import argparse
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import os
import pandas as pd
import pickle
from argparse import Namespace

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def tensor_threshold(tensor, n):
    thresholds = [sorted(arr)[-n] for arr in tensor]
    return [[0 if x < th else 1 for x in arr]
            for arr, th in zip(tensor, thresholds)]


def SMAPE(y_true, y_pred):
    """SMAPE loss function.

    Args:
    - y_true: true labels
    - y_pred: predictions

    Returns:
    - SMAPE: computed rmse loss
    """
    SMAPE = np.mean(np.abs(y_true - y_pred) /
                    ((np.abs(y_true) + np.abs(y_pred)) / 2)
                    )
    return SMAPE

def normalization_for_downstream_tasks(real_data, fake_data,real_history_data):
    # standardization for each sample of data, use the scaler of the real history data to transform both the real and fake data
    # for each sample, normalize them respectively

    #placeholder for the real data and fake data
    real_data_normalized = np.zeros((len(real_data), real_data[0].shape[0], real_data[0].shape[1]))
    fake_data_normalized = np.zeros((len(fake_data), fake_data[0].shape[0], fake_data[0].shape[1]))

    for i in range(real_data.shape[0]):
        scaler = StandardScaler()
        scaler.fit(real_history_data[i])
        real_data_normalized[i] = scaler.transform(real_data[i])
        fake_data_normalized[i] = scaler.transform(fake_data[i])
    return real_data_normalized, fake_data_normalized

def result_summary(result_path):
    model_path = os.path.dirname(result_path)
    # get the filename without extension name
    file_name= os.path.basename(result_path).split('.')[0]
    result_path_high_level = f'{model_path}/{file_name}_high_level.csv'
    # get the mean of every metric of each row and save it into a new row with index 'mean'
    df = pd.read_csv(result_path)
    df_mean = df.mean(axis=0)
    df_mean = df_mean.to_frame().transpose()
    df_mean.index = ['mean']
    df = df.append(df_mean)
    # keep the index
    df.to_csv(result_path_high_level)

def save_to_folder(data_dict, path):
    if not os.path.exists(path):
        os.makedirs(path)
    # for each key in the dict, save the value to the path with the key as the filename into pickle file
    for key, value in data_dict.items():
        file_path = f'{path}/{key}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(value, f)


def configure_network_args(args):
    dynamic_supervisor_args = Namespace(
        model=args.pretrain_model,
        input_size=args.feature_dim,
        output_size=args.dynamic_dim,
        hidden_size=64,
        num_layers=2,
        num_filters=64,
        filter_sizes=[2, 3, 4],
        num_channels=[32],
        kernel_size=3,
        dropout=0.1,
        task_name='classification',
        seq_len=args.max_seq_len,
        pred_len=0,
        e_layers=3,
        enc_in=args.feature_dim,
        hidden_dim=32,
        embed='timeF',
        freq='d',
        num_class=args.dynamic_dim,
    )
    label_supervisor_args = Namespace(
        model=args.pretrain_model,
        input_size=args.feature_dim,
        output_size=args.label_dim,
        hidden_size=64,
        num_filters=64,
        filter_sizes=[2, 3, 4],
        num_layers=2,
        num_channels=[32],
        kernel_size=3,
        dropout=0.1,
        task_name='classification',
        seq_len=args.max_seq_len,
        pred_len=0,
        e_layers=3,
        enc_in=args.feature_dim,
        hidden_dim=32,
        embed='timeF',
        freq='d',
        num_class=args.label_dim,
    )
    TimesNet_args = Namespace(
        use_TimesNet=args.use_TimesNet,
        use_RNN=args.use_RNN,
        add_history=args.add_history,
        input_size=args.feature_dim,
        output_size=args.label_dim,
        hidden_size=32,
        num_filters=64,
        filter_sizes=[2, 3, 4],
        num_layers=2,
        num_channels=[32],
        kernel_size=3,
        dropout=0.1,
        task_name='encoding',
        seq_len=args.max_seq_len,
        pred_len=0,
        e_layers=3,
        enc_in=args.feature_dim,
        hidden_dim=32,
        embed='timeF',
        freq='d',
        num_class=args.label_dim,
        feature_dim=args.feature_dim,
        condition_dim=args.dynamic_dim + args.label_dim
    )
    return dynamic_supervisor_args,label_supervisor_args,TimesNet_args