import random
import torch
import os
import numpy as np
import yaml
import pandas as pd
import copy

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_yaml(yaml_path):
    curPath = os.path.abspath('.')
    yaml_path = os.path.join(curPath, yaml_path)
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    d = yaml.load(cfg, Loader=yaml.FullLoader)
    return d

def load_style_yaml(yaml_path,style):
    curPath = os.path.abspath('.')
    yaml_path = os.path.join(curPath, yaml_path)
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    d = yaml.load(cfg, Loader=yaml.FullLoader)
    data=pd.read_csv(d["df_path"])
    # data['index_by_tick']=data.index
    data=data.reset_index()
    data=data.loc[data['label'] == style, :]
    def get_styled_intervals_and_gives_new_index(data):
        index_by_tick_list=[]
        index_by_tick=[]
        date=data['date']
        last_date=date[0]
        date_counter=0
        index = data['index'].to_list()
        last_value = index[0] - 1
        last_index = 0
        intervals = []
        for i in range(data.shape[0]):
            if last_value != index[i] - 1:
                date_counter = 0
                intervals.append([last_index, i])
                last_value = index[i]
                last_index = i
                index_by_tick_list.append(index_by_tick)
                index_by_tick=[]
            if date[i]!=last_date:
                date_counter+=1
            index_by_tick.append(date_counter)
            last_value = index[i]
            last_date = date[i]
        intervals.append([last_index, i])
        return intervals,index_by_tick
    intervals,index_by_tick_list=get_styled_intervals_and_gives_new_index(data)
    data.drop(columns=['index'])
    if not os.path.exists('temp'):
        os.makedirs('temp')
    d_list=[]
    for i,interval in enumerate(intervals):
        data_temp=data.iloc[interval[0]:interval[1],:]
        data_temp.set_index(index_by_tick_list[i])
        data_temp.to_csv('temp/'+str(style)+'_'+str(i)+'.csv')
        curPath = os.path.abspath('.')
        yaml_path = os.path.join(curPath, yaml_path)
        f = open(yaml_path, 'r', encoding='utf-8')
        cfg = f.read()
        d = yaml.load(cfg, Loader=yaml.FullLoader)
        d["df_path"]='temp/'+str(style)+'_'+str(i)+'.csv'
        d_list.append(d)
    return d_list