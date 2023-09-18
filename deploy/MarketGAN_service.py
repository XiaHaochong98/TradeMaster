from MarketGAN import MarketGAN
from MarketGAN.utils.util import *
from MarketGAN.models.utils import *
import torch
import numpy as np
import os
import pandas as pd
from MarketGAN.data.conditional_data_preprocess import *
from MarketGAN.metrics.visualization import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
class MarketGAN_utils():
    def __init__(self):
        torch.autograd.set_detect_anomaly(True)
        self.model_path='MarketGAN/model'
        variables = pd.read_pickle(f'{self.model_path}/preprocessed_data/variables.pkl')
        # create variables using the variables names in the variables list and read the data from '.output/preprocessed_data'
        for variable in variables:
            try:
                exec(f'{variable}=pd.read_pickle(f"{self.model_path}/preprocessed_data/{variable}.pkl")')
            except:
                continue
        # configure network parameters
        args = training_args
        # torch.manual_seed(args.seed)
        print(args)
        dynamic_supervisor_args, label_supervisor_args, TimesNet_args = configure_network_args(args)

        self.MarketGAN_instance = MarketGAN()
        data_path, args = self.MarketGAN_instance.init(args)

        # update the args
        self.MarketGAN_instance.dynamic_supervisor_args = dynamic_supervisor_args
        self.MarketGAN_instance.label_supervisor_args = label_supervisor_args
        self.MarketGAN_instance.TimesNet_args = TimesNet_args
        # generate the data
        args, model = self.MarketGAN_instance.load(args, self.model_path)
        args.batch_size=64
        self.model=model
        self.args=args
        # read the data from file
        data_path='MarketGAN/data/DJI_data.csv'
        self.data=pd.read_csv(data_path)

        return model



    def inference(self,date,ticker,dynamic,sample_number,work_dir):

        # dynamic: [bear_strength,flat_strength,bull_strength]
        # sample_number: the number of samples we want to generate
        # date: the date we want to generate the data
        # ticker: string of the ticker we want to generate the data

        # filter data by ticker ticker
        data=self.data[self.data['tic']==ticker]
        # sort the data by date, from old to new
        data=data.sort_values(by=['date'])
        features=self.args.feature
        # get the time, D, L, H, Last_h, scaler from the data

        # get the history data from the data, the history data is the data before the date with length args.max_seq_len
        history_sample= data[data['date'] < date].iloc[-self.args.max_seq_len:, :]
        history_sample = history_sample.loc[:, features]

        # reparameterization the history_sample
        # creat a dummy data_sample with the same shape and column names as history_sample
        data_sample = pd.DataFrame(np.ones(history_sample.shape), columns=history_sample.columns)
        _, history_sample = reparameterization(data_sample, history_sample)

        # differential features on 'low' for data_sample and history_sample

        # last_low_befor_data_sample = data.iloc[interval[0] + i - 1, :]['low']
        # do the same thing to history_sample
        last_low_befor_history_sample = data[data['date'] < date].iloc[-self.args.max_seq_len-1, :]
        # data_sample_low_diff = data_sample['low'].diff()
        # the first element of data_sample_low_diff is nan, we replace it with the first low in data sample minus last_low_befor_data_sample
        # data_sample_low_diff.iloc[0] = data_sample['low'].iloc[0] - last_low_befor_data_sample
        # data_sample['low'] = data_sample_low_diff
        # do the same thing to history_sample
        history_sample_low_diff = history_sample['low'].diff()
        history_sample_low_diff.iloc[0] = history_sample['low'].iloc[0] - last_low_befor_history_sample
        history_sample['low'] = history_sample_low_diff
        last_hist_vec=np.array([None, last_low_befor_history_sample])

        # normalization
        # we use the 'low' feature of historical data to normalize the data
        normalized_data_sample, normalized_history_sample, low_scaler, O_minus_L_scaler, C_minus_L_scaler, H_minus_maxOC_scaler = normalization(
            data_sample, history_sample)
        T=self.args.max_seq_len
        scaler = np.vstack([low_scaler, O_minus_L_scaler, C_minus_L_scaler, H_minus_maxOC_scaler]).T  # dim: (n,4)
        print("scaler.shape: ", scaler.shape, " scaler.dtype: ", scaler.dtype)
        # save the order of the scaler
        scaler_order = ['low', 'O_minus_L', 'C_minus_L', 'H_minus_maxOC']
        H=history_sample

        # expand the dynamic to (sample_number,3)
        dynamic=np.array([dynamic]*sample_number)
        # reshape the ticker to (sample_number,feature_dim)
        tic_tokenizer = pd.read_pickle(f"{self.model_path}/preprocessed_data/tic_tokenizer.pkl")
        tic_token = tic_tokenizer.word_to_one_hot(ticker)
        # scaler_vector = [min_sclar_by_tic[tic], max_sclar_by_tic[tic]]
        tic_token = np.array(tic_token, dtype=float)
        tic_token = np.expand_dims(tic_token, axis=0)
        tic_token = np.repeat(tic_token, sample_number, axis=0)  # dim: (n,tic_token)
        # expand the T to (sample_number,1)
        T=np.array([T]*sample_number)
        # expand the scaler to (sample_number,4)
        scaler=np.expand_dims(scaler,axis=0)
        scaler=np.repeat(scaler,sample_number,axis=0)
        # expand the history to (sample_number,T,feature_dim)
        H=np.expand_dims(H,axis=0)
        H=np.repeat(H,sample_number,axis=0)
        # expand the last_low_befor_history_sample to (sample_number,feature_dim)
        last_low_befor_history_sample=np.expand_dims(last_low_befor_history_sample,axis=0)
        last_low_befor_history_sample=np.repeat(last_low_befor_history_sample,sample_number,axis=0)




        ############################# generate the data #########################################

        noisy_dynamic=True
        noisy_history=True
        noise_multiplier=1
        print('noisy_dynamic',noisy_dynamic,'noisy_history',noisy_history)

        # generate the data smaple_number times and save the generated data to a list, get the average of the list as the generated data
        # get a seed list of sample_number length as int
        seed_list=np.random.randint(0,65535,sample_number)
        seed_list=[int(i) for i in seed_list]
        # sperate the train_Last_h to train_Last_h_data and train_Last_h_history on the last dimension
        Last_h_history = last_low_befor_history_sample
        generated_samples=[]

        # apply a random noise to each sample in dynamic and history
        # for each sample, we apply a random noise to the dynamic and history
        # for each element in the dynamic, we sample the dynamic from a normal distribution with mean as the element and std is element/10 and the sample should be non-negative
        # create a dynamic_prossed_noised that has the same shape as dynamic
        dynamic_prossed_noised = np.zeros(dynamic.shape)
        history_prossed_noised = np.zeros(H.shape)
        for i,sample_seed in enumerate(seed_list):
            # istead of using the noise to create randomess, we apply noie on dynamics and history
            # for each element in the dynamic, we sample the dynamic from a normal distribution with mean as the element and std is element/10 and the sample should be non-negative
            # use sample_seed as random seed

            if noisy_dynamic:
                np.random.seed(sample_seed)
                dynamic_prossed_noised[i]=np.random.normal(dynamic[i], dynamic[i]/20)
                # dynamic_prossed_noised=np.random.normal(dynamic, dynamic/20)
                # # set the negative element to 0
                dynamic_prossed_noised[dynamic_prossed_noised<0]=0
                # dynamic_prossed_noised[dynamic_prossed_noised<0]=0
            else:
                # dynamic_prossed_noised=dynamic
                dynamic_prossed_noised[i]=dynamic[i]
            # do the sample for history
            if noisy_history:
                np.random.seed(sample_seed)
                # history_prossed_noised=np.random.normal(H, np.abs(H)/1000)
                history_prossed_noised[i]=np.random.normal(H[i], np.abs(H[i])/1000)
                # set the negative element to 0
                # history_prossed_noised[history_prossed_noised<0]=0
                history_prossed_noised[history_prossed_noised<0]=0
            else:
                # history_prossed_noised=H
                history_prossed_noised[i]=H[i]

            # generated_X_sample = conditional_timegan_generator(model=self.model, T=T, args=self.args,dynamics=dynamic_prossed_noised,labels=tic_token,history=history_prossed_noised,noise_multiplier=noise_multiplier,seed=sample_seed)
            #
            # # rescale the generated data to the original scale
            #
            #
            # generated_data_rescaled_sample = conditional_rescale_data(generated_X_sample, scaler, self.args.differential_features, H, self.args.scaler_order, original_feature_order=features)
            # generated_samples.append(generated_data_rescaled_sample)
            # if i>=1 and i<10:
            #     print('diff between different seeds',np.sum(np.abs(generated_samples[i]-generated_samples[i-1])))
            # print('rescale train')
        # average the generated samples

        generated_X = conditional_timegan_generator(model=self.model, T=T, args=self.args,
                                                           dynamics=dynamic_prossed_noised, labels=tic_token,
                                                           history=history_prossed_noised,
                                                           noise_multiplier=noise_multiplier, seed=sample_seed)

        # rescale the generated data to the original scale

        generated_data_rescaled = conditional_rescale_data(generated_X, scaler,
                                                                  self.args.differential_features, H,
                                                                  self.args.scaler_order,
                                                                  original_feature_order=features)

        generated_data_rescaled_summary=np.mean(generated_samples,axis=0)
        # data_rescaled = conditional_rescale_data(X, scaler, args.differential_features, Last_h_data, args.scaler_order, original_feature_order=args.feature)
        history_rescaled_one = conditional_rescale_data(H[0], scaler[0], self.args.differential_features, Last_h_history[0], self.args.scaler_order, original_feature_order=features)

        # Save generated data
        generated_data_path=f"{work_dir}/{ticker}_{date}_{dynamic}.npy"
        np.save(generated_data_path, generated_data_rescaled)

        data=np.concatenate((history_rescaled_one,generated_data_rescaled_summary),axis=0)
        # print('generated data shape',data.shape)
        figure_path=plot_OHLCcharts(data,features,work_dir,fig_suffix=f'{ticker}_{date}_{dynamic}_{sample_number}',title=f'Average of {sample_number} samples generated on {date} for {ticker} with dynamic {dynamic}')

        return generated_data_path,figure_path

