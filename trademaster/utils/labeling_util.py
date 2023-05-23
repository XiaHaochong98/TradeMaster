import math

import pandas as pd
import yfinance as yf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import statsmodels.api as sm
from scipy.signal import butter,filtfilt
from matplotlib import colors as mcolors
from sklearn.linear_model import LinearRegression
from random import sample
import fractions
import os
from sklearn.manifold import TSNE
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import pickle
import re
import matplotlib.font_manager as font_manager
from dtaidistance import dtw
import time
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class Node:
    def __init__(self, data=None,timestamp=None):
        self.data = data
        self.next = None
        self.timestamp = timestamp

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            current_node = self.head
            while current_node.next:
                current_node = current_node.next
            current_node.next = new_node

    def pop(self):
        if not self.head:
            return None
        if not self.head.next:
            data = self.head.data
            self.head = None
            return data
        else:
            current_node = self.head
            while current_node.next.next:
                current_node = current_node.next
            data = current_node.next.data
            current_node.next = None
            return data

    def to_list(self):
        lst = []
        current_node = self.head
        while current_node:
            lst.append(current_node.data)
            current_node = current_node.next
        return lst

    def set_head_by_timestamp(self, timestamp):
        previous_node = None
        current_node = self.head
        while current_node:
            if current_node.timestamp == timestamp:
                if previous_node:
                    previous_node.next = current_node.next
                current_node.next = self.head
                self.head = current_node
                return
            previous_node = current_node
            current_node = current_node.next

class Dynamic_labeler():
    def __init__(self,mode,dynamic_num,low,high,normalized_coef_list,data,turning_points):
        self.mode=mode
        self.dynamic_num=dynamic_num
        if self.mode == 'slope':
            low, _, high = sorted([low, high, 0])
            self.segments = []
            for i in range(1,self.dynamic_num):
                self.segments.append(low + (high - low) / (dynamic_num) * i)
        elif self.mode == 'quantile':
            self.segments = []
            # find the quantile of normalized_coef_list
            for i in range(1,self.dynamic_num):
                self.segments.append(np.quantile(normalized_coef_list, i / dynamic_num))
        elif self.mode == 'DTW':
            # segment the data by turning points
            self.segments = []
            for i in range(len(turning_points) - 1):
                self.segments.append(data['key_indicator_filtered'][turning_points[i]:turning_points[i + 1]])
            # self.segments.append(data['key_indicator_filtered'][turning_points[-1]:])
            # print('len of data: ',data['key_indicator_filtered'].shape)
            # print('last turning_points ',turning_points[-1])
            # print(self.segments)
            # run the DTW algorithm to cluster the segments into dynamic_num clusters
            self.labels = self.DTW_clustering(self.segments)
        else:
            raise Exception("Sorry, only slope,quantile and DTW mode are provided for now.")

    def DTW_clustering(self,data):
        fitting_data = to_time_series_dataset(data)
        km_stock = TimeSeriesKMeans(n_clusters=self.dynamic_num, metric="dtw", max_iter=100, max_iter_barycenter=100, n_jobs=50,
                                    verbose=0).fit(fitting_data)
        labels = km_stock.predict(fitting_data)
        return labels

    def get(self, coef):
        if self.mode == 'DTW':
            return self.labels[coef]
        elif self.mode == 'slope' or self.mode == 'quantile':
            # find the place where coef falls into in segments
            for i in range(self.dynamic_num - 1):
                if coef <= self.segments[i]:
                    flag = i
                    return flag
            return self.dynamic_num - 1

class Labeler():
    def __init__(self,data,method='linear',filter_strength=1,key_indicator='adjcp',timestamp='date',tic='tic',mode='slope',hard_length_limit=-1,merging_threshold=-1,merging_metric='DTW_distance'):
        plt.ioff()
        self.key_indicator=key_indicator
        self.timestamp=timestamp
        self.tic=tic
        self.mode=mode
        # the hard length limit is the hard constraint of the minium ticks of a continuous segment, which means that any volatility
        # with length less than the hard length limit will be considered as noise.
        self.hard_length_limit=hard_length_limit
        self.merging_metric=merging_metric
        self.merging_threshold=merging_threshold
        if method=='linear':
            self.method='linear'
            # calculate the parameters for filtering
            self.order=4
            self.Wn_key_indicator=self.filter_parameters_calculation(filter_strength)
            # print('Wn_key_indicator: ',self.Wn_key_indicator,' Wn_pct: ',self.Wn_pct,' order: ',self.order)
        else:
            raise Exception("Sorry, only linear model is provided for now.")
        self.preprocess(data)
        
    def filter_parameters_calculation(self,filter_strength):
        if self.hard_length_limit!=-1:
            filter_period=self.hard_length_limit
        else:
            filter_period=7 # default filter period

        # use the hard_length_limit to calculate the Wn_key_indicator and Wn_pct
        # the max Wn_key_indicator is 2, and the max Wn_pct is 2 for not filtering
        Wn_key_indicator=min(2/(filter_period*filter_strength),2)
        # Wn_pct=min(2,2/(filter_period*Wn_pct_factor))
        return Wn_key_indicator
        
    def fit(self,dynamic_number,length_limit,hard_length_limit):
        if self.method=='linear':
            self.turning_points_dict = {}
            self.coef_list_dict = {}
            self.norm_coef_list_dict = {}
            self.y_pred_dict = {}
            self.dynamic_num=dynamic_number
            self.length_limit=length_limit
            self.hard_length_limit=hard_length_limit
            for tic in self.tics:
                coef_list, turning_points, y_pred_list, norm_coef_list = self.linear_regession_turning_points(
                    data_ori=self.data_dict[tic], tic=tic,length_constrain=self.length_limit)
                self.turning_points_dict[tic] = turning_points
                self.coef_list_dict[tic] = coef_list
                self.y_pred_dict[tic] = y_pred_list
                self.norm_coef_list_dict[tic] = norm_coef_list

    def label(self,parameters,work_dir=os.getcwd()):
        # return a dict of label where key is the ticker and value is the label of time-series
        if self.method=='linear':
            try:
                low, high = parameters
            except:
                raise Exception(
                    "parameters shoud be [low,high] where the series would be split into 4 dynamics by low,high and 0 as threshold based on slope. A value of -0.5 and 0.5 stand for -0.5% and 0.5% change per step.")
            self.all_data_seg = []
            self.all_label_seg = []
            self.all_index_seg = []
            for tic in self.tics:
                turning_points = self.turning_points_dict[tic]
                norm_coef_list = self.norm_coef_list_dict[tic]
                label,data_seg,label_seg,index_seg = self.linear_regession_label(self.data_dict[tic],turning_points, low, high, norm_coef_list,tic,self.dynamic_num)
                self.data_dict[tic]['label'] = label
                self.all_data_seg.extend(data_seg)
                self.all_label_seg.extend(label_seg)
                self.all_index_seg.extend(index_seg)
            print('finish labeling')
            # interpolated_pct_return_data_seg = np.array(self.interpolation(self.all_data_seg))
            # try:
            #   self.TSNE_run(interpolated_pct_return_data_seg)
            # except:
            #   print('not able to do TSNE')
            if len(self.tics)>1:
                try:
                  self.stock_DWT(work_dir)
                except:
                  print('not able to do clustering')


    def linear_regession_label(self,data, turning_points, low, high, normalized_coef_list, tic,
                               dynamic_num=4):
        # data = data.reset_index(drop=True)['pct_return_filtered']
        data = data.reset_index(drop=True)
        data_seg = []


        label = []
        label_seg = []
        index_seg = []
        self.dynamic_flag = Dynamic_labeler(mode=self.mode, dynamic_num=dynamic_num, low=low, high=high,
                                            normalized_coef_list=normalized_coef_list,data=data,turning_points=turning_points)
        data=data['pct_return_filtered']
        for i in range(len(turning_points) - 1):
            if self.mode == 'slope' or self.mode=='quantile':
                coef = normalized_coef_list[i]
            elif self.mode == 'DTW':
                coef = i
            flag = self.dynamic_flag.get(coef)
            label.extend([flag] * (turning_points[i + 1] - turning_points[i]))
            if turning_points[i + 1] - turning_points[i] > 2:
                data_seg.append(data.iloc[turning_points[i]:turning_points[i + 1]].to_list())
                label_seg.append(flag)
                index_seg.append(tic + '_' + str(i))
        return label, data_seg, label_seg, index_seg



    def preprocess(self,data):
        # parse the extention of the data file
        if data.split('.')[-1] == 'csv':
            data = pd.read_csv(data)
        elif data.split('.')[-1] == 'feather':
            data = pd.read_feather(data)
            # slice the first 20000 rows for testing
            # TODO: remove this line
            data = data.iloc[:20000]
        # assign 'data' to tic if not exist
        if self.tic not in data.columns:
            data[self.tic] = 'data'
        self.tics = data[self.tic].unique()
        self.data_dict = {}

        for tic in self.tics:
            tic_data = data.loc[data[self.tic] == tic, [self.timestamp, self.tic, self.key_indicator]]
            tic_data.sort_values(by=self.timestamp, ascending=True)
            tic_data = tic_data.assign(pct_return=tic_data[self.key_indicator].pct_change().fillna(0))
            self.data_dict[tic] = tic_data.reset_index(drop=True)
        for tic in self.tics:
            self.adjcp_apply_filter(self.data_dict[tic], self.Wn_key_indicator, self.order)
            self.data_dict[tic] = self.data_dict[tic].assign(pct_return_filtered=self.data_dict[tic]['key_indicator_filtered'].pct_change().fillna(0))

    def stock_DWT(self,work_dir):
        data_by_tic = []
        data_by_tic_1 = []
        for tic in self.tics:
            try:
                data_by_tic.append(self.data_dict[tic].loc[:, ['open', 'high', 'low', 'close', self.key_indicator, 'pct_return']].values)
            except:
                data_by_tic.append(
                    self.data_dict[tic].loc[:, [self.key_indicator, 'pct_return']].values)
            data_by_tic_1.append(self.data_dict[tic].loc[:, 'pct_return'].values)
        fitting_data = to_time_series_dataset(data_by_tic)
        fitting_data_1=to_time_series_dataset(data_by_tic_1)
        km_stock = TimeSeriesKMeans(n_clusters=6, metric="dtw", max_iter=50, max_iter_barycenter=100, n_jobs=50,
                                    verbose=0).fit(fitting_data)
        label_stock = km_stock.predict(fitting_data)
        output = open(os.path.join(work_dir,'DWT_stocks.pkl'), 'wb')
        pickle.dump(km_stock, output, -1)
        output.close()
        output = open(os.path.join(work_dir,'DWT_label_stocks.pkl'), 'wb')
        pickle.dump(label_stock, output, -1)
        output.close()
        for i in range(len(self.tics)):
            self.data_dict[self.tics[i]]['stock_type']=label_stock[i]
        tsne_model = TSNE(n_components=3, perplexity=25, n_iter=300)
        tsne_results = tsne_model.fit_transform(fitting_data_1.reshape(fitting_data_1.shape[0],fitting_data_1.shape[1]))
        self.TSNE_plot(tsne_results,label_stock,'_stock_cluster',folder_name=self.plot_path)

    def plot_ori(self,data, name):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
        if isinstance(data[self.timestamp][0], str):
            date = data[self.timestamp].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data[self.timestamp]
        ax.plot(date, data[self.key_indicator])
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax.set_title(name + '_adjcp', fontsize=20)
        ax.grid(True)
        if not os.path.exists('res/'):
            os.makedirs('res/')
        fig.savefig('res/' + name + '_adjcp' + '.png')

    def plot_pct(self,data, name):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
        if isinstance(data[self.timestamp][0], str):
            date = data[self.timestamp].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data[self.timestamp]
        ax.plot(date, data['pct_return'])
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax.set_title(name + 'pct_return', fontsize=20)
        ax.grid(True)
        if not os.path.exists('res/'):
            os.makedirs('res/')
        fig.savefig('res/' + name + '_pct_return' + '.png')

    def plot_both(self,data, name):
        self.plot_ori(data, name)
        self.plot_pct(data, name)

    def plot_filter(self,data, name, low=6, high=32, K=12):
        # see sm.tsa.filters.bkfilter for more detail, this method is not applied to the pipline for now
        filtered_data = sm.tsa.filters.bkfilter(data[[self.key_indicator, 'pct_return']], low, high, K)
        if isinstance(data[self.timestamp][0], str):
            date = data[self.timestamp][K:-K].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data[self.timestamp][K:-K]
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        ax[0].plot(date, filtered_data['adjcp_cycle'], label='adjcp_cycle')
        ax[1].plot(date, filtered_data['pct_return_cycle'], label='pct_return_cycle')
        ax[0].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[1].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[0].set_title(name + '_adjcp_cycle', fontsize=20)
        ax[1].set_title(name + '_pct_return_cycle', fontsize=20)
        ax[0].grid(True)
        ax[1].grid(True)

    def butter_lowpass_filter(self,data, Wn, order):
        # It is strongly recommended to adjust the Wn based on different data.
        # You can refer to https://en.wikipedia.org/wiki/Butterworth_filter for parameter setting
        # suppose the data is sample at 7Hz (7days / week) so fs=7 and fn=7/2, we would like to eliminate volatility on weekly scale, then we should have
        # Wn=1/(7/2)=2/7
        b, a = butter(order, Wn, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def adjcp_apply_filter(self,data, Wn_adjcp, order):
        data['key_indicator_filtered'] = self.butter_lowpass_filter(data[self.key_indicator], Wn_adjcp, order)
        # data['pct_return_filtered'] = self.butter_lowpass_filter(data['pct_return'], Wn_pct, order)
        # plot the filtered data and save it to res folder
        # self.plot_lowpassfilter(data, 'test')
        # print(data[['key_indicator_filtered', 'pct_return_filtered']])

    def plot_lowpassfilter(self,data, name):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        if isinstance(data[self.timestamp][0], str):
            date = data[self.timestamp].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data[self.timestamp]
        ax[0].plot(date, data['key_indicator_filtered'])
        ax[0].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[0].set_title(name + '_adjcp_filtered', fontsize=20)
        ax[1].plot(date, data['pct_return_filtered'])
        ax[1].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[1].set_title(name + '_pct_return_filtered', fontsize=20)
        if not os.path.exists('res/'):
            os.makedirs('res/')
        fig.savefig('res/' + name + 'filtered' + '.png')

    def find_index_of_turning(self,data):
        turning_points = [0]
        data = data.reset_index(drop=True)
        for i in range(data['pct_return_filtered'].size - 1):
            if data['pct_return_filtered'][i] * data['pct_return_filtered'][i + 1] < 0:
                turning_points.append(i + 1)
        if turning_points[-1] != data['pct_return_filtered'].size:
            turning_points.append(data['pct_return_filtered'].size)
        return turning_points

    def get_mdd(self,seg):
        # get max drawdown of a segment
        mdd = 0
        peak=seg[0]
        for value in seg:
            if value>peak:
                peak=value
            dd=(peak-value)/peak
            if dd>mdd:
                mdd=dd
        return mdd

    def calculate_distance(self,seg1,seg2,iteration_count,mode='default'):
        # calculate the distance between two segments
        if mode=='default':
            mode=self.merging_metric
        if mode=='DTW_distance':
            # the sampling time increase as the iteration_count increase
            distance=self.calculate_dtw_distance(seg1,seg2,iteration_count*3+10)
        return distance
    def calculate_dtw_distance(self,seg1,seg2,max_sample_number=3):
        # calculate the dynamic time warping distance between two segments
        # roll the shorter segment on the longer one with step_size, and calculate the mean distance

        # decide the step size and slice length based on the max_calulation_number
        # we want to include every point in the longer segment at least once/ the longer segment slice length is comparable to the shorter segment/ step size is not too small


        if len(seg1)>len(seg2):
            longer=seg1
            shorter=seg2
        else:
            longer=seg2
            shorter=seg1



        step_size=max(1,math.floor((len(longer)-len(shorter))/max_sample_number))
        # slice_length=int(len(longer)/max_sample_number)
        slice_length = len(shorter)


        distances=[]
        for i in range(0,len(longer)-len(shorter),step_size):
            # print(i)
            # print(shorter,longer[i:i+slice_length])
            distance, paths = fastdtw(shorter, longer[i:i+slice_length])
            distances.append(distance)
        #normalize the distance by the length of the shorter segment and mean value of the shorter segment
        return np.mean(distances)/(slice_length*np.mean(shorter))


    def linear_regession_turning_points(self,data_ori, tic,length_constrain=0):
        """
        1. segment the data into chunks based on turning points(where all neighbors have the opposite slope)
        2. if the length is smaller than hard_length_limit, merge the chunk with its neighbor
        3. Calculate the slope
        5. While the chunk does not satisfy the length limit, and metric satisfied the merging_threshold:
             1.merge the chunk with its neighbor
             2.recalculate the slope
        """


        # recalculate_flag = False
        data = data_ori.reset_index(drop=True)


        # 1. segment the data into chunks based on turning points(where all neighbors have the opposite slope)
        turning_points = self.find_index_of_turning(data)
        # print(turning_points)
        # make every element in turning_points as a list
        turning_points = [[i] for i in turning_points]
        turning_points_ori = turning_points.copy()
        turning_points_new = [[turning_points[0][0]]]
        # turning_points_new = [turning_points[0]]

        # 2.if the length is smaller than hard_length_limit, merge the chunk with its neighbor

        # if length_constrain != 0:
        #     for num,i in enumerate(range(1, len(turning_points) - 1)):
        #         if turning_points[i] - turning_points_new[-1] >= length_constrain:
        #             turning_points_new.append(turning_points[i])
        #     turning_points_new.append(turning_points[-1])
        #     turning_points = turning_points_new
        if length_constrain != 0:
            for i in range(1, len(turning_points) - 1):
                if turning_points[i][0] - turning_points_new[-1][0] >= self.hard_length_limit:
                    #no need to merge
                    turning_points_new.append(turning_points[i])
                else:
                    # merge this point into the current segment
                    turning_points_new[-1].extend(turning_points[i])
            turning_points_new.append(turning_points[-1])
            turning_points = turning_points_new

        # 2. Calculate the slope
        coef_list = []
        normalized_coef_list = []
        y_pred_list = []
        # for i in range(len(turning_points) - 1):
        #     x_seg = np.asarray([j for j in range(turning_points[i], turning_points[i + 1])]).reshape(-1, 1)
        #     adj_cp_model = LinearRegression().fit(x_seg,
        #                                           data['key_indicator_filtered'].iloc[turning_points[i]:turning_points[i + 1]])
        #     y_pred = adj_cp_model.predict(x_seg)
        #     normalized_coef_list.append(100 * adj_cp_model.coef_ / data['key_indicator_filtered'].iloc[turning_points[i]])
        #     coef_list.append(adj_cp_model.coef_)
        #     y_pred_list.append(y_pred)
        for i in range(len(turning_points) - 1):
            x_seg = np.asarray([j for j in range(turning_points[i][0], turning_points[i + 1][0])]).reshape(-1, 1)
            adj_cp_model = LinearRegression().fit(x_seg,
                                                  data['key_indicator_filtered'].iloc[turning_points[i][0]:turning_points[i + 1][0]])
            y_pred = adj_cp_model.predict(x_seg)
            normalized_coef_list.append(100 * adj_cp_model.coef_ / data['key_indicator_filtered'].iloc[turning_points[i][0]])
            coef_list.append(adj_cp_model.coef_)
            y_pred_list.append(y_pred)

        # 3. While the chunk does not satisfy the length limit, and metric satisfied the merging_threshold
        #     1.merge the chunk with its neighbor
        #     2.recalculate the slope
        merging_round = 0
        if self.merging_threshold!=-1:
            change = True
            while change and merging_round<20:
                merging_round += 1
                counter=0
                for i in range(len(turning_points) - 1):
                    if turning_points[i]!=[]:
                        counter+=1
                print('merging round: ', merging_round, 'current number of segments: ', counter)
                change = False
                # for every segment that does not reach self.length_limit, calculate the the DTW distance between the segment and its neighbor
                for i in tqdm(range(len(turning_points) - 1)):
                    # find the first non-empty segment on right side
                    if turning_points[i]==[]:
                        continue
                    have_next_index=False
                    for j in range(i + 1, len(turning_points) - 1):
                        if turning_points[j]!=[]:
                            next_index = j
                            have_next_index=True
                            break
                    if have_next_index==False:
                        break
                    if turning_points[next_index][0] - turning_points[i][0] < self.length_limit:
                        left_distance=float('inf')
                        right_distance=float('inf')
                        this_seg=data['key_indicator_filtered'].iloc[turning_points[i][0]:turning_points[next_index][0]].tolist()
                        if i>0:
                            # find the first non-empty segment on left side
                            left_index=None
                            for j in range(i-1,-1,-1):
                                if turning_points[j]!=[]:
                                    left_index=j
                                    break
                            if left_index is not None:
                                left_neighbor = data['key_indicator_filtered'].iloc[turning_points[left_index][0]:turning_points[i][0]].tolist()
                                left_distance=self.calculate_distance(left_neighbor,this_seg,merging_round)
                        if i<len(turning_points)-2:
                            # find the second non-empty segment on right side
                            next_index_2=None
                            for j in range(next_index+1,len(turning_points)-1):
                                if turning_points[j]!=[]:
                                    right_index_2=j
                                    break
                            if next_index_2 is not None:
                                right_neighbor = data['key_indicator_filtered'].iloc[turning_points[next_index][0]:turning_points[next_index_2][0]].tolist()
                                right_distance=self.calculate_distance(this_seg,right_neighbor,merging_round)
                            else:
                                right_neighbor = data['key_indicator_filtered'].iloc[turning_points[next_index][0]:].tolist()
                                right_distance=self.calculate_distance(this_seg,right_neighbor,merging_round)
                        # pick the min distance that is smaller than the threshold to merge
                        # may choose to merge with the shorter neighbor for balanced segment length
                        # print(min(left_distance,right_distance))
                        if min(left_distance,right_distance)<self.merging_threshold:
                            change=True
                            if left_distance<right_distance:
                                turning_points[left_index]=turning_points[left_index]+turning_points[i]
                            else:
                                turning_points[next_index]=turning_points[i]+turning_points[next_index]
                            turning_points[i]=[]
            counter = 0
            for i in range(len(turning_points) - 1):
                if turning_points[i] != []:
                    counter += 1
            print(f'merging_round in total: {merging_round}, number of segments: {counter}')
            # remove empty segments
            turning_points_new = []
            for i in range(len(turning_points)):
                if turning_points[i]!=[]:
                    turning_points_new.append(turning_points[i])
            turning_points=turning_points_new
            # calculate the slope again
            coef_list = []
            normalized_coef_list = []
            y_pred_list = []
            for i in range(len(turning_points) - 1):
                x_seg = np.asarray([j for j in range(turning_points[i][0], turning_points[i + 1][0])]).reshape(-1, 1)
                adj_cp_model = LinearRegression().fit(x_seg,
                                                      data['key_indicator_filtered'].iloc[
                                                      turning_points[i][0]:turning_points[i + 1][0]])
                y_pred = adj_cp_model.predict(x_seg)
                normalized_coef_list.append(
                    100 * adj_cp_model.coef_ / data['key_indicator_filtered'].iloc[turning_points[i][0]])
                coef_list.append(adj_cp_model.coef_)
                y_pred_list.append(y_pred)





        # # 3. Get max drawdown of each segment
        # if self.merging_threshold!=-1:
        #     recalculate_flag=True
        #     mdd_list = []
        #     for i in range(len(turning_points) - 1):
        #         mdd_list.append(self.get_mdd(data['key_indicator_filtered'].iloc[turning_points[i][0]:turning_points[i + 1][0]].tolist()))

            # print('mdd_list',mdd_list)
            # print('coef_list',coef_list)
        # # 4. re-slice the segment if the if slope/ mdd is smaller than threshold
        #     turning_points_new = []
        #     # print('len(turning_points)',len(turning_points))
        #     for i in range(len(turning_points)-1):
        #         if abs(coef_list[i])/abs(mdd_list[i])<self.merging_threshold:
        #             # print(abs(coef_list[i])/abs(mdd_list[i]))
        #             for j in turning_points_ori[i]:
        #                 turning_points_new.append([j])
        #         else:
        #             turning_points_new.append(turning_points[i])
        # # 4.force merge if the hard constraint is not satisfied
        # if self.hard_length_limit!=-1:
        #     recalculate_flag=True
        #     turning_points_new = [[turning_points[0][0]]]
        #     for i in range(1, len(turning_points) - 1):
        #         if turning_points[i][0] - turning_points_new[-1][0] >= self.hard_length_limit:
        #             #no need to merge
        #             turning_points_new.append(turning_points[i])
        #         else:
        #             # merge this point into the current segment
        #             turning_points_new[-1].extend(turning_points[i])
        #     turning_points_new.append(turning_points[-1])
        #     turning_points = turning_points_new
        # print(len(turning_points))
        #
        # # 5. re-calculate the slope
        # if recalculate_flag:
        #     coef_list = []
        #     normalized_coef_list = []
        #     y_pred_list = []
        #     for i in range(len(turning_points) - 1):
        #         x_seg = np.asarray([j for j in range(turning_points[i][0], turning_points[i + 1][0])]).reshape(-1, 1)
        #         adj_cp_model = LinearRegression().fit(x_seg,
        #                                               data['key_indicator_filtered'].iloc[turning_points[i][0]:turning_points[i + 1][0]])
        #         y_pred = adj_cp_model.predict(x_seg)
        #         normalized_coef_list.append(100 * adj_cp_model.coef_ / data['key_indicator_filtered'].iloc[turning_points[i][0]])
        #         coef_list.append(adj_cp_model.coef_)
        #         y_pred_list.append(y_pred)


        # reshape turning_points to a 1d list
        turning_points = [i[0] for i in turning_points]

        # print('turning_points',turning_points)






        return np.asarray(coef_list), np.asarray(turning_points), y_pred_list, normalized_coef_list

    def plot(self,tics,parameters,data_path,model_id):
        self.plot_path =os.path.join(os.path.dirname(os.path.realpath(data_path)),'MDM_linear',model_id)
        self.plot_path_filtered = os.path.join(os.path.dirname(os.path.realpath(data_path)),'MDM_linear_filtered',model_id)
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        if self.method=='linear':
            try:
                low,high=parameters
            except:
                raise Exception("parameters shoud be [low,high] where the series would be split into 4 dynamics by low,high and 0 as threshold based on slope. A value of -0.5 and 0.5 stand for -0.5% and 0.5% change per step.")
            for tic in tics:
                paths=[]
                paths.append(self.linear_regession_plot(self.data_dict[tic],tic,self.y_pred_dict[tic],self.turning_points_dict[tic],low,high,normalized_coef_list=self.norm_coef_list_dict[tic],folder_name=self.plot_path,plot_feather=self.key_indicator))
                self.linear_regession_plot(self.data_dict[tic], tic, self.y_pred_dict[tic],
                                           self.turning_points_dict[tic], low, high,
                                           normalized_coef_list=self.norm_coef_list_dict[tic],
                                           folder_name=self.plot_path_filtered, plot_feather='key_indicator_filtered')
                return paths
            try:
              self.TSNE_plot(self.tsne_results,self.all_label_seg,folder_name=self.plot_path)
            except:
              print('not able to plot TSNE')
    def linear_regession_plot(self,data, tic, y_pred_list, turning_points, low, high, normalized_coef_list,folder_name=None,plot_feather=None):
        data = data.reset_index(drop=True)
        # every sub-plot is contained segment of at most 100000 data points
        # 1. split the data into segments

        plot_segments=[]
        counter=0
        segments_buffer=[turning_points[0]]
        for index,j in enumerate(range(len(turning_points) - 1)):
            counter+=turning_points[j + 1]-turning_points[j]
            segments_buffer.append(turning_points[j + 1])
            if counter>100000:
                plot_segments.append(segments_buffer)
                segments_buffer=[turning_points[j + 1]]
                counter=0
        plot_segments.append(segments_buffer)
        sub_plot_num=len(plot_segments)

        # fig, ax = plt.subplots(1, 1, figsize=(40, 10), constrained_layout=True)
        fig, axs = plt.subplots(sub_plot_num, 1, figsize=(50, 15*sub_plot_num), constrained_layout=True)
        if sub_plot_num==1:
            axs=[axs]
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

        counter=0
        for index,ax in enumerate(axs):
            turning_points_seg=plot_segments[index]
            for i in range(len(turning_points_seg) - 1):
                x_seg = np.asarray([j for j in range(turning_points_seg[i], turning_points_seg[i + 1])]).reshape(-1, 1)
                # y_pred = y_pred_list[i]
                coef = normalized_coef_list[i+counter]
                flag=self.dynamic_flag.get(coef[0])
                ax.plot(x_seg,data[plot_feather].iloc[turning_points_seg[i]:turning_points_seg[i + 1]], color=colors[flag], label='market style ' + str(flag))
            counter+=len(turning_points_seg)-1
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            font = font_manager.FontProperties(weight='bold',
                                               style='normal', size=16)
            # legend to every sub-plot
            ax.legend(by_label.values(), by_label.keys(), prop=font)
        # set the title
        plt.title(f"Dynamics_of_{tic}_linear_{self.mode}_{plot_feather}", fontsize=20)
        # plt.set_title(f"Dynamics_of_{tic}_linear_{self.mode}", fontsize=20)
        plot_path=folder_name
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig_path=plot_path+'_'+tic+'.png'
        fig.savefig(fig_path)
        plt.close(fig)
        return os.path.abspath(fig_path).replace("\\", "/")

    def linear_regession_timewindow(self,data_ori, tic, adjcp_timewindow):
        # This is the version of linear regession that does not use a turning point to segment. Instead, it applys a fixed-length time winodw.
        # This can be helpful to process data that is extremely volatile, or you simply want a long-term dynamic. However you can achieve similar result by applying stronger filter.
        data = data_ori.iloc[:adjcp_timewindow * (data_ori['key_indicator_filtered'].size // adjcp_timewindow), :]
        adjcp_window_data = [
            data[['key_indicator_filtered']][i * adjcp_timewindow:(i + 1) * adjcp_timewindow].to_numpy().reshape(-1) for i in
            range(data['key_indicator_filtered'].size // adjcp_timewindow)]
        coef_list = []
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        ax[0].plot([i for i in range(data.shape[0])], data['key_indicator_filtered'])
        for i, data_seg in enumerate(adjcp_window_data):
            x_seg = np.asarray([i * adjcp_timewindow + j for j in range(adjcp_timewindow)]).reshape(-1, 1)
            adj_cp_model = LinearRegression().fit(x_seg, data_seg)
            y_pred = adj_cp_model.predict(x_seg)
            ax[1].plot([i * adjcp_timewindow + j for j in range(adjcp_timewindow)], y_pred)
            coef_list.append(adj_cp_model.coef_)
        return coef_list

    def interpolation(self,data):
        max_len = max([len(d) for d in data])
        for i, d in enumerate(data):
            l = len(d)
            to_fill = max_len - l
            if to_fill != 0:
                interval = max_len // to_fill
                for j in range(to_fill):
                    idx = (interval + 1) * j + interval
                    data[i].insert(min(idx, len(data[i]) - 1), float('nan'))
            data[i] = pd.Series(data[i]).interpolate(method='polynomial', order=2)
        return data

    def TSNE_run(self,data_seg):
        interpolated_pct_return_data_seg = np.array(self.interpolation(data_seg))
        self.tsne = TSNE(n_components=2,perplexity=40, n_iter=300)
        self.tsne_results = self.tsne.fit_transform(interpolated_pct_return_data_seg)

    def TSNE_plot(self,data, label_list,title='',folder_name=None):
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
        for i in range(len(data) - 1):
            label = label_list[i]
            ax.scatter(data[i][0], data[i][1], color=colors[label], alpha=0.2, label='cluster' + str(label))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title('TSNE', fontsize=20)
        plot_path = folder_name
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(plot_path+'TSNE'+title+'.png')
        plt.close(fig)

