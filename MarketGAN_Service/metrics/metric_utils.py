# Necessary packages
import random
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch
from dataset.dataset import FeaturePredictionDataset, OneStepPredictionDataset, DiscriminatorDataset, \
    LabelPredictionDataset,PosthocDiscriminatorDataset
from metrics.Dynamics_prediction_TCN import TCN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from models.TimesNet import TimesNet
from models.utils import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from utils.util import *


def reidentify_score(enlarge_label, pred_label):
    """Return the reidentification score.

    Args:
    - enlarge_label: 1 for train data, 0 for other data
    - pred_label: 1 for reidentified data, 0 for not reidentified data

    Returns:
    - accuracy: reidentification score
    """  
    accuracy = accuracy_score(enlarge_label, pred_label > 0.5)  
    return accuracy

def feature_prediction(train_data, test_data, index,args_,epoch):
    """Use the other features to predict a certain feature.

    Args:
    - train_data (train_data, train_time): training time-series
    - test_data (test_data, test_data): testing time-series
    - index: feature index to be predicted

    Returns:
    - perf: average performance of feature predictions (in terms of AUC or MSE)
    """
    train_data, train_time = train_data
    test_data, test_time = test_data

    # Parameters
    no, seq_len, dim = train_data.shape

    # Set model parameters

    # args = {}
    # args["device"] = args_.device
    # args["task"] = "regression"
    # args["model_type"] = "gru"
    # args["bidirectional"] = False
    # args["epochs"] = 250
    # args["batch_size"] = 128
    # args["in_dim"] = dim-1
    # args["h_dim"] = int(int(dim-1)/2)
    # args["out_dim"] = 1
    # args["n_layers"] = 3
    # args["dropout"] = 0.5
    # args["padding_value"] = -1.0
    # args["max_seq_len"] = args_.max_seq_len
    # args["learning_rate"] = 1e-3
    # args["grad_clip_norm"] = 5.0
    # args['weight_decay']=1

    TimesNet_arg = Namespace(
        embedding_size=64,
        learning_rate=1e-3,
        batch_size=512,
        hidden_size=64,
        num_filters=64,
        filter_sizes=[2, 3, 4],
        num_layers=2,
        num_channels=[32],
        kernel_size=3,
        dropout=0.1,
        task_name='tick_wise_classification',
        seq_len=args_.max_seq_len,
        pred_len=0, # one step ahead prediction
        e_layers=3,
        enc_in=args_.feature_dim-1,# one feature is the target feature
        hidden_dim=32,
        embed='timeF',
        freq='d',
        num_class=1,
        epoch=epoch,
        device=args_.device
    )

    # Output initialization
    metric = list()

    scale=True
    # normalization
    if scale:
        # normalization the train data using MinMaxScaler
        scaler = MinMaxScaler()
        # train_data shape (no,seq_len,dim)
        scaler.fit(train_data.reshape(-1,train_data.shape[-1]))
        # transform the train data and test data
        # print('train_data',train_data.shape)
        # print('test_data',test_data.shape)
        train_data = scaler.transform(train_data.reshape(-1,train_data.shape[-1])).reshape(train_data.shape)
        test_data = scaler.transform(test_data.reshape(-1,test_data.shape[-1])).reshape(test_data.shape)
        # print('train_data',train_data.shape)
        # print('test_data',test_data.shape)




    # For each index
    for idx in index:
        # Set training features and labels
        train_dataset = FeaturePredictionDataset(
            train_data, 
            train_time, 
            idx
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=TimesNet_arg.batch_size,
            shuffle=True
        )

        # Set testing features and labels
        test_dataset = FeaturePredictionDataset(
            test_data, 
            test_time,
            idx
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=TimesNet_arg.batch_size,
            shuffle=False
        )

        # Initialize model
        model = TimesNet(TimesNet_arg)
        model.to(args_.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TimesNet_arg.learning_rate
        )

        logger = trange(TimesNet_arg.epoch, desc=f"Epoch: 0, Loss: 0")
        for epoch in logger:
            running_loss = 0.0

            for train_x, train_t, train_y in train_dataloader:
                train_x = train_x.to(args_.device)
                train_y = train_y.to(args_.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # print('train_x',train_x.shape)
                train_p = model(train_x, train_t)
                # print('train_p',train_p.shape)
                # print('train_y',train_y.shape)
                loss = criterion(train_p, train_y)
                # backward
                loss.backward()
                # optimize
                optimizer.step()

                running_loss += loss.item()

            logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")

        
        # Evaluate the trained model
        with torch.no_grad():
            running_metric = 0
            for test_x, test_t, test_y in test_dataloader:
                test_x = test_x.to(TimesNet_arg.device)
                test_p = model(test_x, test_t).cpu().numpy()

                # if scale:
                #     print('test_p',test_p.shape)
                #     print('test_y',test_y.shape)
                #     test_p = scaler.inverse_transform(test_p.reshape(-1,1)).reshape(test_p.shape)
                #     test_y = scaler.inverse_transform(test_y.reshape(-1,1)).reshape(test_y.shape)
                #     test_p = np.reshape(test_p, [-1])
                #     test_y = np.reshape(test_y, [-1])
                # else:
                #     test_p = np.reshape(test_p, [-1])
                #     test_y = np.reshape(test_y.numpy(), [-1])
                test_p = np.reshape(test_p, [-1])
                test_y = np.reshape(test_y.numpy(), [-1])

                
                # Compute SMAPE
                running_metric += SMAPE(test_y, test_p)
      
        metric.append((100*running_metric)/len(test_dataset))
    
    return metric
      
def one_step_ahead_prediction(train_data, test_data,args_,epochs=100):
    """Use the previous time-series to predict one-step ahead feature values.

    Args:
    - train_data: training time-series
    - test_data: testing time-series

    Returns:
    - perf: average performance of one-step ahead predictions (in terms of AUC or MSE)
    """
    os.environ['PYTHONHASHSEED'] = str(args_.seed)
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    train_data, train_time = train_data
    test_data, test_time = test_data
    
    # Parameters
    no, seq_len, dim = train_data.shape


    if_TimeNet = True
    TimesNet_arg = Namespace(
        c_out=args_.feature_dim,
        learning_rate=1e-3,
        batch_size=512,
        embedding_size=64,
        hidden_size=64,
        num_filters=64,
        filter_sizes=[2, 3, 4],
        num_layers=2,
        num_channels=[32],
        kernel_size=3,
        dropout=0.1,
        task_name='short_term_forecast',
        seq_len=args_.max_seq_len-1,
        pred_len=1, # one step ahead prediction
        e_layers=3,
        enc_in=args_.feature_dim,
        hidden_dim=32,
        embed='timeF',
        freq='d',
        num_class=args_.label_dim,
        epochs=epochs,
        device=args_.device
    )

    # Set model parameters
    # args = {}
    # args["device"] = args_.device
    # args["task"] = "regression"
    # args["model_type"] = "gru"
    # args["bidirectional"] = False
    # args["epochs"] = epochs
    # args["batch_size"] = 128
    # args["in_dim"] = dim
    # args["h_dim"] = int(int(dim)/2)
    # args["out_dim"] = dim
    # args["n_layers"] = 3
    # args["dropout"] = 0.5
    # args["padding_value"] = -1.0
    # args["max_seq_len"] = args_.max_seq_len - 1   # only 99 is used for prediction
    # args["learning_rate"] = 1e-3
    # args["grad_clip_norm"] = 5.0
    # args['weight_decay']=1

    #noramlization
    scale=True
    if scale:
        # normalization the train data using MinMaxScaler
        scaler = MinMaxScaler()
        # train_data shape (no,seq_len,dim)
        scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
        # transform the train data and test data
        train_data = scaler.transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
        test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)

    # Set training features and labels
    train_dataset = OneStepPredictionDataset(train_data, train_time,if_TimeNet)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args_.batch_size,
        shuffle=True
    )
    # print('train_set shuflle True')

    # Set testing features and labels
    test_dataset = OneStepPredictionDataset(test_data, test_time,if_TimeNet)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args_.batch_size,
        shuffle=False
    )
    # Initialize model
    model = TimesNet(TimesNet_arg)
    model.to(args_.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TimesNet_arg.learning_rate
    )

    # Train the predictive model
    logger = trange(TimesNet_arg.epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        running_loss = 0.0
        for train_x, train_t, train_y in train_dataloader:
            train_x = train_x.to(args_.device)
            train_y = train_y.to(args_.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            train_p = model(train_x, train_t)
            loss = criterion(train_p, train_y)
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()
        # average loss
        running_loss /= len(train_dataloader)
        logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")
    # record the last epoch running_loss as training loss
    train_loss = running_loss

    # Evaluate the trained model
    with torch.no_grad():
        running_metric = 0
        running_metric_rescale = 0
        for test_x, test_t, test_y in test_dataloader:
            test_x = test_x.to(args_.device)
            test_p = model(test_x, test_t).cpu()
            if scale:
                # copy test_p and test_y as test_p_norm and test_y_norm
                test_p_norm = test_p.clone()
                test_y_norm = test_y.clone()
                # reshape test_p_norm and test_y_norm
                test_p_norm = np.reshape(test_p_norm.numpy(),[-1])
                test_y_norm = np.reshape(test_y_norm.numpy(),[-1])
                running_metric+= SMAPE(test_y_norm, test_p_norm)

                test_p = scaler.inverse_transform(test_p.reshape(-1, test_p.shape[-1])).reshape(test_p.shape)
                test_y = scaler.inverse_transform(test_y.reshape(-1, test_y.shape[-1])).reshape(test_y.shape)
                test_p = np.reshape(test_p, [-1])
                test_y = np.reshape(test_y, [-1])
                running_metric_rescale += SMAPE(test_y, test_p)
            else:
                test_p = np.reshape(test_p.numpy(), [-1])
                test_y = np.reshape(test_y.numpy(), [-1])
                running_metric += SMAPE(test_y, test_p)

    return (100*running_metric)/len(test_dataset),(100*running_metric_rescale)/len(test_dataset),train_loss


class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN
    """
    def __init__(self, args):
        super(DiscriminatorNetwork, self).__init__()
        self.input_size= args.input_size
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Discriminator Architecture
        self.dis_rnn = torch.nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.dis_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.dis_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information
        Returns:
            - logits: predicted logits (B x S x 1)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, H_t = self.dis_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100
        logits = self.dis_linear(H_o).squeeze(-1)
        classification=torch.sigmoid(logits)
        return logits,classification

class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN
    """

    def __init__(self, args_):
        super(DiscriminatorNetwork, self).__init__()
        TimesNet_arg = Namespace(
            batch_size=args_.batch_size,
            input_size=args_.input_size,
            output_size=1,
            train_rate=0.8,
            embedding_size=64,
            hidden_size=64,
            num_layers=2,
            num_filters=64,
            filter_sizes=[2, 3, 4],
            num_channels=[32],
            kernel_size=3,
            dropout=0.1,
            task_name='classification',
            seq_len=args_.max_seq_len,
            pred_len=0,
            e_layers=3,
            enc_in=args_.input_size,
            hidden_dim=32,
            embed='timeF',
            freq='d',
            num_class=1,
        )
        self.model = TimesNet(TimesNet_arg)

    def forward(self, x,t):
        try:
            x = self.model.forward(x)
        except:
            print('x shape is',x.shape)
            print('x' ,x)
        logit=x
        # get softmax of x
        x = torch.nn.functional.sigmoid(x)
        return logit,x

    # def __init__(self, args):
    #     super(DiscriminatorNetwork, self).__init__()
    #     self.input_size= args.input_size
    #     self.hidden_dim = args.hidden_dim
    #     self.num_layers = args.num_layers
    #     self.padding_value = args.padding_value
    #     self.max_seq_len = args.max_seq_len
    #
    #     # Discriminator Architecture
    #     self.dis_rnn = torch.nn.GRU(
    #         input_size=self.input_size,
    #         hidden_size=self.hidden_dim,
    #         num_layers=self.num_layers,
    #         batch_first=True
    #     )
    #     self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)
    #
    #     # Init weights
    #     # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
    #     # Reference:
    #     # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
    #     # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
    #     with torch.no_grad():
    #         for name, param in self.dis_rnn.named_parameters():
    #             if 'weight_ih' in name:
    #                 torch.nn.init.xavier_uniform_(param.data)
    #             elif 'weight_hh' in name:
    #                 torch.nn.init.xavier_uniform_(param.data)
    #             elif 'bias_ih' in name:
    #                 param.data.fill_(1)
    #             elif 'bias_hh' in name:
    #                 param.data.fill_(0)
    #         for name, param in self.dis_linear.named_parameters():
    #             if 'weight' in name:
    #                 torch.nn.init.xavier_uniform_(param)
    #             elif 'bias' in name:
    #                 param.data.fill_(0)
    #
    # def forward(self, H, T):
    #     """Forward pass for predicting if the data is real or synthetic
    #     Args:
    #         - H: latent representation (B x S x E)
    #         - T: input temporal information
    #     Returns:
    #         - logits: predicted logits (B x S x 1)
    #     """
    #     # Dynamic RNN input for ignoring paddings
    #     H_packed = torch.nn.utils.rnn.pack_padded_sequence(
    #         input=H,
    #         lengths=T,
    #         batch_first=True,
    #         enforce_sorted=False
    #     )
    #
    #     # 128 x 100 x 10
    #     H_o, H_t = self.dis_rnn(H_packed)
    #
    #     # Pad RNN output back to sequence length
    #     H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
    #         sequence=H_o,
    #         batch_first=True,
    #         padding_value=self.padding_value,
    #         total_length=self.max_seq_len
    #     )
    #
    #     # 128 x 100
    #     logits = self.dis_linear(H_o).squeeze(-1)
    #     classification=torch.sigmoid(logits)
    #     return logits,classification


def post_hoc_discriminator(ori_data, generated_data,args_,epoch=0):
    args = {}
    args["device"] = args_.device
    args["model_type"] = "gru"
    args["epochs"] = epoch
    args["batch_size"] = args_.batch_size
    print(f'batch size is {args["batch_size"]}')
    args["num_layers"] = 6
    args["padding_value"] = -1.0
    args["max_seq_len"] = args_.max_seq_len
    args["train_rate"] = 0.8
    args["learning_rate"] = 1e-3
    args['weight_decay']=1
    print('seed')
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)

    ori_data,ori_time=ori_data
    generated_data,generated_time=generated_data
    # random_seed=random.randint(1, 100000)
    # print('random_seed',random_seed)
    ori_train_data, ori_test_data, ori_train_time, ori_test_time = train_test_split(
        ori_data, ori_time, test_size=1-args['train_rate'],random_state=args_.seed
    )
    generated_train_data, generated_test_data, generated_train_time, generated_test_time = train_test_split(
        generated_data, generated_time, test_size=1-args['train_rate'],random_state=args_.seed
    )
    no, seq_len, dim = ori_data.shape
    args["input_size"] = dim
    args["hidden_dim"] = int(int(dim)/2)
    args_tuple = namedtuple('GenericDict', args.keys())(**args)


    #normalize the data
    scale=True
    if scale:
        # normalization the train data using MinMaxScaler
        scaler = MinMaxScaler()
        # concatenate ori_train_data and generated_train_data
        full_data=np.concatenate((ori_train_data,generated_train_data),axis=0)
        scaler.fit(full_data.reshape(-1, dim))
        # transform the ori_train_data and generated_train_data
        ori_train_data = scaler.transform(ori_train_data.reshape(-1, dim)).reshape(ori_train_data.shape)
        generated_train_data = scaler.transform(generated_train_data.reshape(-1, dim)).reshape(generated_train_data.shape)
        # transform the ori_test_data and generated_test_data
        ori_test_data = scaler.transform(ori_test_data.reshape(-1, dim)).reshape(ori_test_data.shape)
        generated_test_data = scaler.transform(generated_test_data.reshape(-1, dim)).reshape(generated_test_data.shape)


    train_dataset=DiscriminatorDataset(ori_data=ori_train_data,generated_data=generated_train_data, ori_time=ori_train_time,generated_time=generated_train_time)
    test_dataset=DiscriminatorDataset(ori_data=ori_test_data, generated_data=generated_test_data,
                                         ori_time=ori_test_time, generated_time=generated_test_time)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False
    )

    #Train the post-host discriminator
    discriminator = DiscriminatorNetwork(args_tuple)
    discriminator.to(args["device"])
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=args['learning_rate'],weight_decay=args['weight_decay'])
    logger = trange(args["epochs"], desc=f"Epoch: 0,loss: 0, real_loss: 0, fake_loss: 0")
    for epoch in logger:
        running_real_loss = 0.0
        running_fake_loss = 0.0
        running_loss = 0.0
        for generated_data, generated_time, ori_data, ori_time in train_dataloader:
            generated_data=generated_data.to(args["device"])
            # generated_time=generated_time.to(args["device"])
            ori_data=ori_data.to(args["device"])
            # ori_time=ori_time.to(args["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            generated_logits,generated_label = discriminator(generated_data, generated_time)
            ori_logits,ori_label = discriminator(ori_data, ori_time)
            D_loss_real = torch.nn.functional.binary_cross_entropy(ori_label, torch.ones_like(ori_label))
            D_loss_fake = torch.nn.functional.binary_cross_entropy(generated_label, torch.zeros_like(generated_label))
            # print('ori_logits',ori_logits)
            # print('generated_logits',generated_logits)
            D_loss=D_loss_real+D_loss_fake
            # backward
            D_loss.backward()
            # optimize
            optimizer.step()
            # print(generated_data.shape,ori_data.shape)
            running_real_loss += D_loss_real.item()
            running_fake_loss += D_loss_fake.item()
            running_loss +=D_loss.item()

        logger.set_description(f"batchnum: {len(train_dataloader)}, Epoch: {epoch},loss: {running_loss/len(train_dataloader):.4f}, real_loss: {running_real_loss/len(train_dataloader):.4f}, fake_loss: {running_fake_loss/len(train_dataloader):.4f}")
    # Evaluate the discriminator on the test set
    with torch.no_grad():
        discriminative_score = []
        running_loss=0
        for generated_data, generated_time, ori_data, ori_time in test_dataloader:
            generated_data = generated_data.to(args["device"])
            # generated_time = generated_time.to(args["device"])
            ori_data = ori_data.to(args["device"])
            # ori_time = ori_time.to(args["device"])

            generated_logits,generated_label = discriminator(generated_data, generated_time)
            generated_logits=generated_logits.cpu()
            generated_label=generated_label.cpu()
            ori_logits,ori_label = discriminator(ori_data, ori_time)
            ori_logits=ori_logits.cpu()
            ori_label=ori_label.cpu()
            y_pred_final = torch.squeeze(torch.concat((ori_label, generated_label), axis=0))
            y_label_final = torch.concat((torch.ones_like(ori_label), torch.zeros_like(generated_label)),
                                           axis=0)
            # print(f"y_pred_final data preview:\n{y_pred_final}\n")
            # print(f"y_label_final data preview:\n{y_label_final}\n")
            # print(f"(y_pred_final > 0.5) data preview:\n{(y_pred_final > 0.5)}\n")
            # print(y_pred_final.shape,y_label_final.shape,generated_logits.shape,ori_logits.shape,(y_pred_final > 0.5).shape)
            # acc_1=accuracy_score(y_label_final, y_pred_final)
            acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
            discriminative_score.append(acc)
            # discriminative_score.append(np.abs(0.5 - acc))
            # print('acc,score,acc_ori: ',acc,np.abs(0.5 - acc))

            #calculate the loss of the discriminator
            # print(y_pred_final.shape,y_label_final.shape)
            y_pred_final = torch.squeeze(y_pred_final)
            y_label_final = torch.squeeze(y_label_final)
            running_loss += torch.nn.functional.binary_cross_entropy(y_pred_final, y_label_final)

        # print("discriminative_score by batch:",discriminative_score)

    return (sum(discriminative_score)*100)/len(discriminative_score),running_loss/len(test_dataloader)

# def post_hoc_discriminator(ori_data, generated_data,args_):
#     # we train a classifier to distinguish between real and fake data
#     TimesNet_arg = Namespace(
#         batch_size=args_.batch_size,
#         input_size=args_.feature_dim,
#         output_size=args_.dynamic_dim,
#         train_rate=0.8,
#         embedding_size=64,
#         hidden_size=64,
#         num_layers=2,
#         num_filters=64,
#         filter_sizes=[2, 3, 4],
#         num_channels=[32],
#         kernel_size=3,
#         dropout=0.1,
#         task_name='classification',
#         seq_len=args_.max_seq_len,
#         pred_len=0,
#         e_layers=3,
#         enc_in=args_.feature_dim,
#         hidden_dim=32,
#         embed='timeF',
#         freq='d',
#         num_class=1,
#     )
#
#     class Posthoc_discriminator(torch.nn.Module):
#         def __init__(self, args):
#             super(Posthoc_discriminator, self).__init__()
#             self.model = TimesNet(args)
#
#         def forward(self, x):
#             x = self.model.forward(x)
#             # get softmax of x
#             x = torch.nn.functional.softmax(x, dim=1)
#             return x
#
#     ori_data, ori_time = ori_data
#     generated_data, generated_time = generated_data
#     random_seed = random.randint(1, 100000)
#     print('random_seed', random_seed)
#     # split data to train test and validation
#
#     ori_train_data, ori_test_data, ori_train_time, ori_test_time = train_test_split(
#         ori_data, ori_time, test_size=1 - TimesNet_arg.train_rate, random_state=random_seed
#     )
#     generated_train_data, generated_test_data, generated_train_time, generated_test_time = train_test_split(
#         generated_data, generated_time, test_size=1 - TimesNet_arg.train_rate, random_state=random_seed
#     )
#     # set label of real data to 1 and fake data to 0
#     ori_train_label = np.ones(len(ori_train_data))
#     generated_train_label = np.zeros(len(generated_train_data))
#     ori_test_label = np.ones(len(ori_test_data))
#     generated_test_label = np.zeros(len(generated_test_data))
#     # cast time list ot numpy array
#     ori_train_time = np.array(ori_train_time)
#     generated_train_time = np.array(generated_train_time)
#     ori_test_time = np.array(ori_test_time)
#     generated_test_time = np.array(generated_test_time)
#     # convert to torch tensor
#     ori_train_data = torch.from_numpy(ori_train_data).float()
#     generated_train_data = torch.from_numpy(generated_train_data).float()
#     ori_test_data = torch.from_numpy(ori_test_data).float()
#     generated_test_data = torch.from_numpy(generated_test_data).float()
#     ori_train_time = torch.from_numpy(ori_train_time).long()
#     generated_train_time = torch.from_numpy(generated_train_time).long()
#     ori_test_time = torch.from_numpy(ori_test_time).long()
#     generated_test_time = torch.from_numpy(generated_test_time).long()
#     ori_train_label = torch.from_numpy(ori_train_label).float()
#     generated_train_label = torch.from_numpy(generated_train_label).float()
#     ori_test_label = torch.from_numpy(ori_test_label).float()
#     generated_test_label = torch.from_numpy(generated_test_label).float()
#
#     # create dataset
#     train_dataset = PosthocDiscriminatorDataset(real_data=ori_train_data, real_time=ori_train_time, real_label=ori_train_label,
#                                          fake_data=generated_train_data, fake_time=generated_train_time,
#                                          fake_label=generated_train_label)
#     test_dataset = PosthocDiscriminatorDataset(real_data=ori_test_data, real_time=ori_test_time, real_label=ori_test_label,
#                                         fake_data=generated_test_data, fake_time=generated_test_time,
#                                         fake_label=generated_test_label)
#     # create dataloader
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TimesNet_arg.batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TimesNet_arg.batch_size, shuffle=False)
#     # create model
#     model = Posthoc_discriminator(TimesNet_arg)
#     # to args_.device
#     model.to(args_.device)
#     # create optimizer
#     optimizer = torch.optim.Adam(model.parameters())
#
#
#     logger = trange(5, desc=f"Epoch: 0, Loss: 0")
#     for epoch in logger:
#         running_loss_real = 0.0
#         running_loss_fake = 0.0
#         running_loss_sum = 0.0
#         running_acc_real = []
#         running_acc_fake = []
#         for X_real, T_real, Y_real, X_fake, T_fake, Y_fake in tqdm(train_loader, position=1, desc="dataloader",
#                                                                    leave=False, colour='red',
#                                                                    ncols=80):
#             # we follow the discriminator training scheme, we train the discriminator on real data and fake data separately
#             X_real = X_real.to(args_.device)
#             T_real = T_real.to(args_.device)
#             Y_real = Y_real.to(args_.device)
#             X_fake = X_fake.to(args_.device)
#             T_fake = T_fake.to(args_.device)
#             Y_fake = Y_fake.to(args_.device)
#             # Reset gradients
#             optimizer.zero_grad()
#             # print('forward')
#
#         #     # concatenate X_real and X_fake to X
#         #     X = torch.cat((X_real, X_fake), 0)
#         #     # concatenate T_real and T_fake to T
#         #     T = torch.cat((T_real, T_fake), 0)
#         #     # concatenate Y_real and Y_fake to Y
#         #     Y = torch.cat((Y_real, Y_fake), 0)
#         #     predict_Y= model(X)
#         #     # reshape predict_Y to 1D
#         #     predict_Y = predict_Y.reshape(-1)
#         #     # print(predict_Y.shape, predict_Y_real.shape,Y_real.shape,Y_fake.shape)
#         #     loss = torch.nn.functional.binary_cross_entropy(predict_Y, Y).to(args_.device)
#         #     loss.backward()
#         #     optimizer.step()
#         #     running_loss_sum+= loss.item()
#         #     # print('backward')
#         #     # print statistics
#         # logger.set_description(f"Epoch: {epoch}, Loss: {running_loss_sum / len(train_loader)}")
#
#         #     # print('X_real',X_real,'X_fake',X_fake)
#         #
#         #     # Forward Pass
#         #     predict_Y_real = model(X_real)
#         #     predict_Y_fake = model(X_fake)
#         #
#         #     # reshape predict_Y to 1D
#         #     predict_Y_fake = predict_Y_fake.reshape(-1)
#         #     predict_Y_real = predict_Y_real.reshape(-1)
#         #     # print(predict_Y_fake.shape, predict_Y_real.shape,Y_real.shape,Y_fake.shape)
#         #     loss_real = torch.nn.functional.binary_cross_entropy(predict_Y_real, Y_real).to(args_.device)
#         #     loss_fake = torch.nn.functional.binary_cross_entropy(predict_Y_fake, Y_fake).to(args_.device)
#         #     loss =  loss_real + loss_fake
#         #     # calculate accuracy
#         #     acc_real = torch.sum(torch.round(predict_Y_real) == Y_real).float() / len(Y_real)
#         #     acc_fake = torch.sum(torch.round(predict_Y_fake) == Y_fake).float() / len(Y_fake)
#         #     # Backward Pass
#         #     loss.backward()
#         #     # Update model parameters
#         #     optimizer.step()
#         #     # log loss and accuracy
#         #     print(f"loss_real: {loss_real.item()}, loss_fake: {loss_fake.item()},loss_sum:{loss.item()} ,acc_real: {acc_real.item()}, acc_fake: {acc_fake.item()}")
#         #     running_loss_real += loss_real.item()
#         #     running_loss_fake += loss_fake.item()
#         #     running_loss_sum += loss.item()
#         #     running_acc_real.append(acc_real.item())
#         #     running_acc_fake.append(acc_fake.item())
#         # # Log loss for final batch of each epoch
#         # logger.set_description(
#         #     f"Epoch: {epoch}, loss_real: {running_loss_real / len(train_loader)}, loss_fake: {running_loss_fake / len(train_loader)},loss_sum:{running_loss_sum / len(train_loader)} ,acc_real: {np.mean(running_acc_real)}, acc_fake: {np.mean(running_acc_fake)}")
#     # run prediction on test data
#     test_predict_real = []
#     test_predict_fake = []
#     test_Y_real = []
#     test_Y_fake = []
#     for X_real, T_real, Y_real, X_fake, T_fake, Y_fake in tqdm(test_loader, position=1, desc="dataloader", leave=False,
#                                                                colour='red',
#                                                                ncols=80):
#         # to args_.device
#         X_real = X_real.to(args_.device)
#         T_real = T_real.to(args_.device)
#         Y_real = Y_real.to(args_.device)
#         X_fake = X_fake.to(args_.device)
#         T_fake = T_fake.to(args_.device)
#         Y_fake = Y_fake.to(args_.device)
#         # Forward Pass
#         predict_Y_real = model(X_real)
#         predict_Y_fake = model(X_fake)
#         test_predict_real.append(predict_Y_real)
#         test_predict_fake.append(predict_Y_fake)
#         test_Y_real.append(Y_real)
#         test_Y_fake.append(Y_fake)
#
#     # calculate loss and accuracy
#     test_predict_real = torch.cat(test_predict_real)
#     test_predict_fake = torch.cat(test_predict_fake)
#     test_loss_real = torch.nn.functional.binary_cross_entropy(test_predict_real, torch.cat(test_Y_real))
#     test_loss_fake = torch.nn.functional.binary_cross_entropy(test_predict_fake, torch.cat(test_Y_fake))
#     # process test result, if predict value >0.5, then label it as 1, otherwise 0
#     test_result_real = test_predict_real.detach().cpu().numpy()
#     test_result_real[test_result_real > 0.5] = 1
#     test_result_real[test_result_real <= 0.5] = 0
#     test_result_fake = test_predict_fake.detach().cpu().numpy()
#     test_result_fake[test_result_fake > 0.5] = 1
#     test_result_fake[test_result_fake <= 0.5] = 0
#     test_acc_real = accuracy_score(torch.cat(test_Y_real), test_result_real)
#     test_acc_fake = accuracy_score(torch.cat(test_Y_fake), test_result_fake)
#     print('test_loss_real:', test_loss_real.item(), 'test_loss_fake:', test_loss_fake.item(), 'test_acc_real:',
#           test_acc_real, 'test_acc_fake:', test_acc_fake)
#     test_loss = (test_loss_real + test_loss_fake) / (2 * len(test_loader))
#     test_acc = (test_acc_real + test_acc_fake) / 2
#     return test_loss, test_acc
#
#     # args = {}
#     # args["device"] = args_.device
#     # args["model_type"] = "gru"
#     # args["epochs"] = 100
#     # args["batch_size"] = 128
#     # args["num_layers"] = 6
#     # args["padding_value"] = -1.0
#     # args["max_seq_len"] = args_.max_seq_len
#     # args["train_rate"] = 0.8
#     # args["learning_rate"] = 1e-3
#     # args['weight_decay'] = 1
#     #
#     # TimesNet_arg = Namespace(
#     #     c_out=args_.feature_dim,
#     #     batch_size=128,
#     #     embedding_size=64,
#     #     hidden_size=64,
#     #     num_filters=64,
#     #     filter_sizes=[2, 3, 4],
#     #     num_layers=2,
#     #     num_channels=[32],
#     #     kernel_size=3,
#     #     dropout=0.1,
#     #     task_name='classification',
#     #     seq_len=args_.max_seq_len,
#     #     pred_len=0,
#     #     e_layers=3,
#     #     enc_in=args_.feature_dim,
#     #     hidden_dim=32,
#     #     embed='timeF',
#     #     freq='d',
#     #     num_class=1,  # real vs fake
#     # )
#     #
#     # ori_data, ori_time = ori_data
#     # generated_data, generated_time = generated_data
#     # random_seed = random.randint(1, 100000)
#     # print('random_seed', random_seed)
#     # ori_train_data, ori_test_data, ori_train_time, ori_test_time = train_test_split(
#     #     ori_data, ori_time, test_size=1 - args['train_rate'], random_state=random_seed
#     # )
#     # generated_train_data, generated_test_data, generated_train_time, generated_test_time = train_test_split(
#     #     generated_data, generated_time, test_size=1 - args['train_rate'], random_state=random_seed
#     # )
#     # no, seq_len, dim = ori_data.shape
#     # args["input_size"] = dim
#     # args["hidden_dim"] = int(int(dim) / 2)
#     # args_tuple = namedtuple('GenericDict', args.keys())(**args)
#     # train_dataset = DiscriminatorDataset(ori_data=ori_train_data, generated_data=generated_train_data,
#     #                                      ori_time=ori_train_time, generated_time=generated_train_time)
#     # test_dataset = DiscriminatorDataset(ori_data=ori_test_data, generated_data=generated_test_data,
#     #                                     ori_time=ori_test_time, generated_time=generated_test_time)
#     #
#     # train_dataloader = torch.utils.data.DataLoader(
#     #     train_dataset,
#     #     batch_size=args["batch_size"],
#     #     shuffle=True
#     # )
#     #
#     # test_dataloader = torch.utils.data.DataLoader(
#     #     test_dataset,
#     #     batch_size=len(test_dataset),
#     #     shuffle=True
#     # )
#     #
#     # class Posthoc_discriminator(torch.nn.Module):
#     #     def __init__(self, args):
#     #         super(Posthoc_discriminator, self).__init__()
#     #         self.model = TimesNet(args)
#     #
#     #     def forward(self, x):
#     #         x = self.model.forward(x)
#     #         # get softmax of x
#     #         x = torch.nn.functional.softmax(x, dim=1)
#     #         return x
#     #
#     # # Train the post-host discriminator
#     # # discriminator = DiscriminatorNetwork(args_tuple)
#     # discriminator = Posthoc_discriminator(TimesNet_arg)
#     # discriminator.to(args["device"])
#     # optimizer = torch.optim.Adam(discriminator.parameters(), lr=args['learning_rate'],
#     #                              weight_decay=args['weight_decay'])
#     # logger = trange(args["epochs"], desc=f"Epoch: 0,loss: 0, real_loss: 0, fake_loss: 0")
#     # for epoch in logger:
#     #     running_real_loss = 0.0
#     #     running_fake_loss = 0.0
#     #     running_loss = 0.0
#     #     for generated_data, generated_time, ori_data, ori_time in train_dataloader:
#     #         generated_data = generated_data.to(args["device"])
#     #         # generated_time=generated_time.to(args["device"])
#     #         ori_data = ori_data.to(args["device"])
#     #         # ori_time=ori_time.to(args["device"])
#     #         # zero the parameter gradients
#     #         optimizer.zero_grad()
#     #         # forward
#     #         generated_label = discriminator(generated_data)
#     #         ori_label = discriminator(ori_data)
#     #         # print(generated_label.shape,ori_label.shape)
#     #         # print(generated_label,ori_label)
#     #         # print(torch.nn.functional.binary_cross_entropy(ori_label, torch.ones_like(ori_label)))
#     #         # print(torch.nn.functional.binary_cross_entropy(generated_label, torch.zeros_like(generated_label)))
#     #         # negative value in it cause error
#     #         # TODO:add softmax to the output
#     #         D_loss_real = torch.nn.functional.binary_cross_entropy(ori_label,
#     #                                                                torch.ones_like(ori_label).to(args["device"])).to(args["device"])
#     #         D_loss_fake = torch.nn.functional.binary_cross_entropy(generated_label,
#     #                                                                torch.zeros_like(generated_label).to(args["device"])).to(args["device"])
#     #         D_loss = D_loss_real + D_loss_fake
#     #         # backward
#     #         D_loss.backward()
#     #         # optimize
#     #         optimizer.step()
#     #         # print(generated_data.shape,ori_data.shape)
#     #         running_real_loss += D_loss_real.item()
#     #         running_fake_loss += D_loss_fake.item()
#     #         running_loss += D_loss.item()
#     #
#     #     logger.set_description(
#     #         f"batchnum: {len(train_dataloader)}, Epoch: {epoch},loss: {running_loss / len(train_dataloader):.4f}, real_loss: {running_real_loss / len(train_dataloader):.4f}, fake_loss: {running_fake_loss / len(train_dataloader):.4f}")
#
#     # return sum(discriminative_score) / len(discriminative_score)
#




def predictive_score(ori_data, generated_data,args_):
    args = {}
    args["device"] = args_.device
    # args["task"] = "regression"
    # args["model_type"] = "gru"
    # args["bidirectional"] = False
    # args["epochs"] = 200
    # args["batch_size"] = 128
    # args["n_layers"] = 3
    # args["dropout"] = 0.5
    # args["padding_value"] = -1.0
    # args["max_seq_len"] = args_.max_seq_len - 1  # only 99 is used for prediction
    args["learning_rate"] = 1e-3
    # args["grad_clip_norm"] = 5.0
    #
    ori_data,ori_time=ori_data
    generated_data,generated_time=generated_data
    print('random_seed',random_seed)
    no, seq_len, dim = ori_data.shape
    # args["in_dim"] = dim
    # args["h_dim"] = dim
    # args["out_dim"] = dim

    TimesNet_arg = Namespace(
        model=args.pretrain_model,
        input_size=args.feature_dim,
        output_size=args.label_dim,
        embedding_size=64,
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

    args_tuple = namedtuple('GenericDict', args.keys())(**args)



    # Set training features and labels
    train_dataset = OneStepPredictionDataset(ori_data, ori_time)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True
    )

    # Set testing features and labels
    test_dataset = OneStepPredictionDataset(generated_data, generated_time)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        shuffle=False
    )
    # Initialize model
    model = TimesNet(args)
    model.to(args["device"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["learning_rate"]
    )

    # Train the predictive model
    logger = trange(args["epochs"], desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        running_loss = 0.0

        for train_x, train_t, train_y in train_dataloader:
            train_x = train_x.to(args["device"])
            train_y = train_y.to(args["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            train_p = model(train_x, train_t)
            loss = criterion(train_p, train_y)
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()

        logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")

    # Evaluate the trained model
    with torch.no_grad():
        perf = 0
        for test_x, test_t, test_y in test_dataloader:
            test_x = test_x.to(args["device"])
            test_p = model(test_x, test_t).cpu()

            test_p = np.reshape(test_p.numpy(), [-1])
            test_y = np.reshape(test_y.numpy(), [-1])

            perf += SMAPE(test_y, test_p)

    return perf/len(test_dataset)

def label_score(raw_data, label,args_,training=False,model_name='default',):
    train_rate=0.8
    data,time=raw_data
    no, seq_len, dim = data.shape
    batch_size = 32
    input_size = dim
    sequence_length = seq_len
    # Instantiate the model
    num_channels = [32, 64, 64]
    kernel_size = 3
    dropout = 0.2
    # dim of label
    output_size = label.shape[-1]
    args = {}
    args["device"] = args_.device
    args["task"] = "regression"
    args["epochs"] = 50
    args["padding_value"] = -1.0
    args["learning_rate"]=1e-3

    # Create some random input data



    # Set training features and labels

    if not training:
        test_dataset = LabelPredictionDataset(data, time,label)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        #load model from file
        try:
            model=torch.load(f'{args.model_path}/dynamics_prediction_model/{model_name}_dynamics_classification_model.pt')
        except:
            training=True
    if training:
        train_data, test_data, train_time, test_time = train_test_split(
            data, time, test_size=1 - train_rate, random_state=args_.seed
        )

        train_dataset = LabelPredictionDataset(train_data, train_time,label)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Set testing features and labels
        test_dataset = LabelPredictionDataset(test_data, test_time,label)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        # Initialize model
        model = TCN(input_size, output_size, num_channels, kernel_size, dropout)


    model.to(args["device"])
    # loss for one-hot label prediction
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["learning_rate"]
    )

    if training:
        # Train the predictive model
        logger = trange(args["epochs"], desc=f"Epoch: 0, Loss: 0")
        for epoch in logger:
            # running_loss = 0.0
            acc_list=[]
            for train_x, train_t, train_y in train_dataloader:
                train_x = train_x.to(args["device"])
                train_y = train_y.to(args["device"])
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                train_p = model(train_x)
                loss = criterion(train_p, train_y)
                # backward
                loss.backward()
                # optimize
                optimizer.step()

                running_loss=loss.item()
                # add acc here
                result = tensor_threshold(train_p, 1)
                acc = accuracy_score(train_y.detach().cpu().numpy(), result)
                acc_list.append(acc)

            logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}, mean acc: {sum(acc_list)/len(acc_list):.4f}")

        # Evaluate the trained model
        with torch.no_grad():
            acc_list=[]
            for test_x, test_t, test_y in test_dataloader:
                test_x = test_x.to(args["device"])
                test_p = model(test_x).cpu()

                result = tensor_threshold(test_p, 1)
                acc = accuracy_score(test_y.detach().cpu().numpy(), result)
                acc_list.append(acc)
                # perf += rmse_error(test_y, test_p)
            print(f'evaluation avg acc {sum(acc_list)/len(acc_list):.4f}')
    else:
        #Test the generated data on the trained model
        with torch.no_grad():
            acc_list=[]
            for test_x, test_t, test_y in test_dataloader:
                test_x = test_x.to(args["device"])
                test_p = model(test_x).cpu()

                result = tensor_threshold(test_p, 1)
                acc = accuracy_score(test_y.detach().cpu().numpy(), result)
                acc_list.append(acc)
            print(f'evaluation avg acc {sum(acc_list) / len(acc_list):.4f}')

    if training:
        #save model
        if not os.path.exists(f'{args.model_path}/dynamics_prediction_model'):
            os.makedirs(f'{args.model_path}/dynamics_prediction_model')
        torch.save(model,f'{args.model_path}/dynamics_prediction_model/{model_name}_dynamics_classification_model.pt')

    return sum(acc_list) / len(acc_list)

def feature_constraint_evaluaton(data,features):
    high_low_diff = data[:, :, features.index('high')] - data[:, :, features.index('low')]
    high_open_diff = data[:, :, features.index('high')] - data[:, :,
                                                                         features.index('open')]
    high_close_diff = data[:, :, features.index('high')] - data[:, :,
                                                                          features.index('close')]
    low_open_diff = data[:, :, features.index('low')] - data[:, :, features.index('open')]
    low_close_diff = data[:, :, features.index('low')] - data[:, :,features.index('close')]
    # get the percentage of the data that doesn't satisfy the logic constraints
    high_low_ratio_loss = np.sum(high_low_diff < 0) / (data.shape[0] * data.shape[1])
    high_open_ratio_loss = np.sum(high_open_diff < 0) / (data.shape[0] * data.shape[1])
    high_close_ratio_loss = np.sum(high_close_diff < 0) / (data.shape[0] * data.shape[1])
    low_open_ratio_loss = np.sum(low_open_diff > 0) / (data.shape[0] * data.shape[1])
    low_close_ratio_loss = np.sum(low_close_diff > 0) / (data.shape[0] * data.shape[1])
    print(
        f"Percentage of data that doesn't satisfy the logic constraints: {round(high_low_ratio_loss, 4)}, {round(high_open_ratio_loss, 4)}, {round(high_close_ratio_loss, 4)}, {round(low_open_ratio_loss, 4)}, {round(low_close_ratio_loss, 4)}")
    avg_ratio = (high_low_ratio_loss + high_open_ratio_loss + high_close_ratio_loss + low_open_ratio_loss + low_close_ratio_loss) / 5
    return avg_ratio
