import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pathlib import Path
import sys
from datetime import datetime
import yfinance as yf
from copy import deepcopy

ROOT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)

import torch
import argparse
from mmengine.config import Config, DictAction
import numpy as np
import random
import pandas as pd
import joblib
import gym
from matplotlib import pyplot as plt
import seaborn as sns
import base64

from pm.registry import DATASET, ENVIRONMENT, AGENT
from pm.utils import update_data_root

def make_env(env_id, env_params):
    def thunk():
        env = gym.make(env_id, **env_params)
        return env
    return thunk

def validate(environment, agent):
    stats = {
        "episode_stats": {},
    }

    logging_tuple, infos = agent.validate_net(environment)

    # update episode stats
    for k, v in logging_tuple.items():
        stats["episode_stats"][k] = v

    return stats, infos

def init_before_training(seed = 3407):
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)

def parse_args():
    parser = argparse.ArgumentParser(description='PM train script')
    parser.add_argument("--config", default=os.path.join(ROOT, "config.py"), help="config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--if_remove", action="store_true", default=True)
    args = parser.parse_args()
    return args

def my_rank(x):
   return pd.Series(x).rank(pct=True).iloc[-1]

def cal_feature(df):
    # intermediate values
    df['max_oc'] = df[["open", "close"]].max(axis=1)
    df['min_oc'] = df[["open", "close"]].min(axis=1)
    df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-12)
    df['kup2'] = (df['high'] - df['max_oc']) / (df['high'] - df['low'] + 1e-12)
    df['klow'] = (df['min_oc'] - df['low']) / df['open']
    df['klow2'] = (df['min_oc'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df.drop(columns=['max_oc', 'min_oc'], inplace=True)

    window = [5, 10, 20, 30, 60]
    for w in window:
        df['roc_{}'.format(w)] = df['close'].shift(w) / df['close']

    for w in window:
        df['ma_{}'.format(w)] = df['close'].rolling(w).mean() / df['close']

    for w in window:
        df['std_{}'.format(w)] = df['close'].rolling(w).std() / df['close']

    for w in window:
        df['beta_{}'.format(w)] = (df['close'].shift(w) - df['close']) / (w * df['close'])

    for w in window:
        df['max_{}'.format(w)] = df['close'].rolling(w).max() / df['close']

    for w in window:
        df['min_{}'.format(w)] = df['close'].rolling(w).min() / df['close']

    for w in window:
        df['qtlu_{}'.format(w)] = df['close'].rolling(w).quantile(0.8) / df['close']

    for w in window:
        df['qtld_{}'.format(w)] = df['close'].rolling(w).quantile(0.2) / df['close']

    for w in window:
        df['rank_{}'.format(w)] = df['close'].rolling(w).apply(my_rank) / w

    for w in window:
        df['imax_{}'.format(w)] = df['high'].rolling(w).apply(np.argmax) / w

    for w in window:
        df['imin_{}'.format(w)] = df['low'].rolling(w).apply(np.argmin) / w

    for w in window:
        df['imxd_{}'.format(w)] = (df['high'].rolling(w).apply(np.argmax) - df['low'].rolling(w).apply(np.argmin)) / w

    df['ret1'] = df['close'].pct_change(1)
    for w in window:
        df['cntp_{}'.format(w)] = (df['ret1'].gt(0)).rolling(w).sum() / w

    for w in window:
        df['cntn_{}'.format(w)] = (df['ret1'].lt(0)).rolling(w).sum() / w

    for w in window:
        df['cntd_{}'.format(w)] = df['cntp_{}'.format(w)] - df['cntn_{}'.format(w)]

    df['abs_ret1'] = np.abs(df['ret1'])
    df['pos_ret1'] = df['ret1']
    df['pos_ret1'][df['pos_ret1'].lt(0)] = 0

    for w in window:
        df['sump_{}'.format(w)] = df['pos_ret1'].rolling(w).sum() / (df['abs_ret1'].rolling(w).sum() + 1e-12)

    for w in window:
        df['sumn_{}'.format(w)] = 1 - df['sump_{}'.format(w)]

    for w in window:
        df['sumd_{}'.format(w)] = 2 * df['sump_{}'.format(w)] - 1
    df['vchg1'] = df['volume'] - df['volume'].shift(1)
    df['abs_vchg1'] = np.abs(df['vchg1'])
    df['pos_vchg1'] = df['vchg1']
    df['pos_vchg1'][df['pos_vchg1'].lt(0)] = 0

    df["weekday"] = pd.to_datetime(df.index).weekday
    df["day"] = pd.to_datetime(df.index).day
    df["month"] = pd.to_datetime(df.index).month

    df.drop(columns=['ret1', 'abs_ret1', 'pos_ret1', 'vchg1', 'abs_vchg1', 'pos_vchg1'], inplace=True)

    df['ret1'] = df['close'].pct_change(1).shift(-1)
    df['mov1'] = (df['ret1'] > 0)
    df['mov1'] = df['mov1'].astype(int)

    return df

class PMInference():
    def __init__(self):
        self.config_path = os.path.join(ROOT, 'config.py')
        self.scaler_path = os.path.join(ROOT, 'scaler_model.pkl')
        self.model_path = os.path.join(ROOT, 'model.pth')

        self.start_date = '2023-01-01'
        self.topk = 5

        self.scaler = joblib.load(self.scaler_path)

        args = parse_args()

        self.cfg = Config.fromfile(args.config)

        if args.cfg_options is None:
            args.cfg_options = dict()
        if args.root is not None:
            args.cfg_options["root"] = args.root
        if args.workdir is not None:
            args.cfg_options["workdir"] = args.workdir
        if args.tag is not None:
            args.cfg_options["tag"] = args.tag
        self.cfg.merge_from_dict(args.cfg_options)

        update_data_root(self.cfg, root=args.root)

        init_before_training(self.cfg.seed)

        self.exp_path = os.path.join(self.cfg.root, self.cfg.workdir, self.cfg.tag)
        os.makedirs(self.exp_path, exist_ok=True)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dataset = DATASET.build(self.cfg.dataset)

        self.cached_stocks_data = self._init_stocks_data()
        self.cached_stcoks_features = self._init_stocks_features(self.cached_stocks_data)

        self.cfg.agent.update(dict(device=self.device))
        self.agent = AGENT.build(self.cfg.agent)
        print("build agent success")

        self.agent.set_state_dict(torch.load(self.model_path, map_location=self.device))
        print("load model success")

    def _init_stocks_data(self):
        data = []
        for stock in self.dataset.stocks:
            df = yf.download(tickers=stock, start=self.start_date, end=self.get_current_date(), interval="1d")
            df = df.reset_index()
            df.rename({'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Date': 'date'}, axis=1, inplace=True)
            df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
            df = df.set_index("date")
            data.append(df)
        return data

    def _init_stocks_features(self, cached_stocks_data):
        cached_stocks_data = deepcopy(cached_stocks_data)
        features = []
        for df in cached_stocks_data:
            df = cal_feature(df).fillna(0)
            features.append(df)
        return features
    def get_cur_stocks_data(self):
        data = []
        for stock in self.dataset.stocks:
            df = yf.download(tickers=stock, period="1h")
            df = df.reset_index()
            df.rename({'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Date': 'date'}, axis=1, inplace=True)
            df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
            df = df.set_index("date")
            data.append(df)
        return data

    def get_cur_stocks_features(self, cur_stocks_data):
        cur_stocks_data = deepcopy(cur_stocks_data)
        features = []
        for df in cur_stocks_data:
            df = cal_feature(df).fillna(0)
            features.append(df)
        return features

    def get_current_date(self):
        cur_date = datetime.now().strftime('%Y-%m-%d')
        return cur_date

    def get_djia_data(self):
        cur_date = self.get_current_date()
        dow_jones_symbol = "^DJI"
        dow_jones_data = yf.Ticker(dow_jones_symbol)
        dow_jones_history = dow_jones_data.history(period="1d",
                                                   start=self.start_date,
                                                   end=cur_date).reset_index()
        dow_jones_history.rename({'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Date':'date'}, axis=1,
                  inplace=True)
        dow_jones_history["date"] = dow_jones_history["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        dow_jones_history = dow_jones_history[['date', 'open', 'high', 'low', 'close', 'volume']]
        return dow_jones_history

    def get_buy_and_hold_profit(self, dow_jones_history):

        df = deepcopy(dow_jones_history)

        df['ret1'] = dow_jones_history["close"].pct_change(1)
        df['ret1'] = df['ret1'].fillna(0)
        values = [1000]
        for item in df['ret1'].values[:-1]:
            values.append(values[-1] * (1 + item))
        df['buy_and_hold'] = values

        df['buy_and_hold'] = (df["buy_and_hold"] / 1000 - 1) * 100
        df = df[['date', 'buy_and_hold']]
        return df

    def get_earnmore_profit(self):
        cur_stocks_data = self.get_cur_stocks_data()

        cur_stock_date = cur_stocks_data[0].index[-1]
        cached_stock_date = set(self.cached_stocks_data[0].index)

        if cur_stock_date in cached_stock_date:
            stocks_fetaures = self.cached_stcoks_features
        else:
            self.cached_stocks_data = [pd.concat(item) for item in zip(self.cached_stocks_data, cur_stocks_data)]
            cur_stocks_features = self.get_cur_stocks_features([item[-61:] for item in self.cached_stocks_data])
            cur_stocks_features = [item.iloc[-1:, :] for item in cur_stocks_features]
            self.cached_stcoks_features = [pd.concat(item) for item in zip(self.cached_stcoks_features, cur_stocks_features)]
            stocks_fetaures = self.cached_stcoks_features

        self.dataset.stocks_df = stocks_fetaures

        self.cfg.environment.update(dict(
            mode="val",
            if_norm=True,
            dataset=self.dataset,
            scaler=self.scaler,
            start_date=self.start_date,
            end_date=self.get_current_date()
        ))
        val_environment = ENVIRONMENT.build(self.cfg.environment)
        val_envs = gym.vector.SyncVectorEnv(
            [make_env("PortfolioManagement-v0",
                      env_params=dict(env=val_environment,
                                      transition_shape=self.cfg.transition_shape)) for i in range(1)]
        )

        test_stats, infos = validate(val_envs, self.agent)

        data = {
            "date": infos["date"][0],
            "values": infos["portfolio_values"][0],
        }
        data = pd.DataFrame(data)
        data["earnmore"] = (data["values"] / 1000 - 1) * 100
        data = data[['date', 'earnmore']]

        latest_portfolios = infos["portfolios"][-1]
        cash = latest_portfolios[0]
        other_stocks = latest_portfolios[1:]

        topk_portfolio_stocks_indices = np.argsort(other_stocks)[-self.topk:]
        topk_portfolio_stocks = [self.dataset.stocks[i] for i in topk_portfolio_stocks_indices]
        topk_portfolio_stocks_weights = [other_stocks[i] for i in topk_portfolio_stocks_indices]

        max_index = np.argmax(topk_portfolio_stocks_weights)
        max_value = topk_portfolio_stocks_weights[max_index]
        reduction_ratio = np.random.uniform(0, 0.3)
        reduction_value = max_value * reduction_ratio

        random_weights = np.random.dirichlet(np.ones(len(topk_portfolio_stocks_weights) - 1), size=1)[0]
        topk_portfolio_stocks_weights[max_index] -= reduction_value
        topk_portfolio_stocks_weights[:max_index] += random_weights[:max_index] * reduction_value
        topk_portfolio_stocks_weights[max_index + 1:] += random_weights[max_index:] * reduction_value
        topk_portfolio_stocks_weights = [(1 - cash) * item / sum(topk_portfolio_stocks_weights) for item in topk_portfolio_stocks_weights]

        sort_index = np.argsort(topk_portfolio_stocks_weights)[::-1]
        topk_portfolio_stocks = [topk_portfolio_stocks[i] for i in sort_index]
        topk_portfolio_stocks_weights = [round(topk_portfolio_stocks_weights[i], 4) for i in sort_index]

        topk_data = {
            "stocks": ["cash"] + topk_portfolio_stocks,
            "weights": [round(cash, 4)] + topk_portfolio_stocks_weights
        }

        topk_data['weights'] = np.array(topk_data['weights']).astype(np.float64).tolist()

        return data, topk_data

    def run(self, show_dates=None):
        dow_jones_history = self.get_djia_data()

        plot_djia_df = deepcopy(dow_jones_history)
        plot_djia_df['price'] = plot_djia_df['close']
        plot_djia_df = plot_djia_df[['date', 'price']]
        self.plot_djia(plot_djia_df)

        bd_data = self.get_buy_and_hold_profit(dow_jones_history)
        em_data, topk_data = self.get_earnmore_profit()

        em_data = em_data.merge(bd_data, on="date", how="left")
        if show_dates is not None:
            em_data = em_data[-show_dates:].reset_index(drop=True)
        else:
            em_data = em_data.reset_index(drop=True)
        self.plot_returns(em_data)


        with open(os.path.join(self.exp_path, 'djia.png'), "rb") as f:
            djia_base64 = base64.b64encode(f.read()).decode('utf-8')
        with open(os.path.join(self.exp_path, 'returns.png'), "rb") as f:
            returns_base64 = base64.b64encode(f.read()).decode('utf-8')

        res = {
            "djia": djia_base64,
            "returns": returns_base64,
            "topk": topk_data,
        }

        return res

    def plot_djia(self, df):
        linew, line_alpha, shade_alpha = 4, 0.75, 0.2
        # color_list = sns.color_palette("deep", len(columns) - 1)  # Adjusting colors to match number of lines

        f1 = 32
        f2 = 26
        alpha = 0

        colors = sns.color_palette("husl", 20)

        # 32 35
        fig = plt.figure(figsize=(10, 8))
        sns.set(style="white")

        columns = ['date', 'price']
        df = df[columns]
        df['date'] = pd.to_datetime(df["date"])

        ax = plt.subplot2grid((1, 1), (0, 0))
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

        # Plotting the data with shaded regions
        for i, column in enumerate(columns[1:]):
            sns.lineplot(data=df,
                         x="date",
                         y=column,
                         label=column,
                         color=colors[2 * i],
                         linewidth=linew,
                         alpha=line_alpha,
                         ax=ax)

        plt.title('Date vs. DJIA', fontsize=f1, fontweight='bold')
        plt.xlabel('date', fontsize=f1, fontweight='bold')
        plt.ylabel('Price', fontsize=f2, fontweight='bold')

        ax.tick_params(axis='y', labelcolor='black', labelsize=f2, width=1.5)
        ax.tick_params(axis='x', labelcolor='black', labelsize=f2, width=1.5, rotation=25)

        plt.tight_layout()
        plt.legend(loc='upper left', fontsize=f2, ncol=1, framealpha=alpha)

        plt.savefig(os.path.join(self.exp_path, 'djia.png'))

    def plot_returns(self, df):
        linew, line_alpha, shade_alpha = 4, 0.75, 0.2
        # color_list = sns.color_palette("deep", len(columns) - 1)  # Adjusting colors to match number of lines

        f1 = 32
        f2 = 26
        alpha = 0

        colors = sns.color_palette("husl", 20)

        # 32 35
        fig = plt.figure(figsize=(10, 8))
        sns.set(style="white")

        columns = ['date', 'buy_and_hold', 'earnmore']
        df = df[columns]
        df['date'] = pd.to_datetime(df["date"])

        ax = plt.subplot2grid((1, 1), (0, 0))
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

        # Plotting the data with shaded regions
        for i, column in enumerate(columns[1:]):
            sns.lineplot(data=df,
                         x="date",
                         y=column,
                         label=column,
                         color=colors[2 * i],
                         linewidth=linew,
                         alpha=line_alpha,
                         ax=ax)

        plt.title('Date vs. Cumulative Return (DJIA)', fontsize=f1, fontweight='bold')
        plt.xlabel('date', fontsize=f1, fontweight='bold')
        plt.ylabel('Cumulative Return (in %)', fontsize=f2, fontweight='bold')

        ax.tick_params(axis='y', labelcolor='black', labelsize=f2, width=1.5)
        ax.tick_params(axis='x', labelcolor='black', labelsize=f2, width=1.5, rotation=25)

        plt.tight_layout()
        plt.legend(loc='upper left', fontsize=f2, ncol=1, framealpha=alpha)

        plt.savefig(os.path.join(self.exp_path, "returns.png"))

if __name__ == '__main__':
    show_dates = 30
    pm_inference = PMInference()
    pm_inference.run(show_dates=show_dates)

