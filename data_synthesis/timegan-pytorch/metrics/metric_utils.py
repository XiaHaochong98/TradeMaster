# Necessary packages
import torch
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from metrics.general_rnn import GeneralRNN
from metrics.dataset import FeaturePredictionDataset, OneStepPredictionDataset, DiscriminatorDataset
import random
from collections import namedtuple

def rmse_error(y_true, y_pred):
    """User defined root mean squared error.

    Args:
    - y_true: true labels
    - y_pred: predictions

    Returns:
    - computed_rmse: computed rmse loss
    """
    # Exclude masked labels
    idx = (y_true >= 0) * 1
    # Mean squared loss excluding masked labels
    computed_mse = np.sum(idx * ((y_true - y_pred)**2)) / np.sum(idx)
    computed_rmse = np.sqrt(computed_mse)
    return computed_rmse

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

def feature_prediction(train_data, test_data, index):
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

    args = {}
    args["device"] = "cuda"
    args["task"] = "regression"
    args["model_type"] = "gru"
    args["bidirectional"] = False
    args["epochs"] = 20
    args["batch_size"] = 128
    args["in_dim"] = dim-1
    args["h_dim"] = dim-1
    args["out_dim"] = 1
    args["n_layers"] = 3
    args["dropout"] = 0.5
    args["padding_value"] = -1.0
    args["max_seq_len"] = 100
    args["learning_rate"] = 1e-3
    args["grad_clip_norm"] = 5.0

    # Output initialization
    perf = list()
  
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
            batch_size=args["batch_size"],
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
            batch_size=no,
            shuffle=False
        )

        # Initialize model
        model = GeneralRNN(args)
        model.to(args["device"])
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["learning_rate"]
        )

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
            temp_perf = 0
            for test_x, test_t, test_y in test_dataloader:
                test_x = test_x.to(args["device"])
                test_p = model(test_x, test_t).cpu().numpy()

                test_p = np.reshape(test_p, [-1])
                test_y = np.reshape(test_y.numpy(), [-1])
        
                temp_perf = rmse_error(test_y, test_p)
      
        perf.append(temp_perf)
    
    return perf
      
def one_step_ahead_prediction(train_data, test_data):
    """Use the previous time-series to predict one-step ahead feature values.

    Args:
    - train_data: training time-series
    - test_data: testing time-series

    Returns:
    - perf: average performance of one-step ahead predictions (in terms of AUC or MSE)
    """
    train_data, train_time = train_data
    test_data, test_time = test_data
    
    # Parameters
    no, seq_len, dim = train_data.shape

    # Set model parameters
    args = {}
    args["device"] = "cuda"
    args["task"] = "regression"
    args["model_type"] = "gru"
    args["bidirectional"] = False
    args["epochs"] = 20
    args["batch_size"] = 128
    args["in_dim"] = dim
    args["h_dim"] = dim
    args["out_dim"] = dim
    args["n_layers"] = 3
    args["dropout"] = 0.5
    args["padding_value"] = -1.0
    args["max_seq_len"] = 100 - 1   # only 99 is used for prediction
    args["learning_rate"] = 1e-3
    args["grad_clip_norm"] = 5.0

    # Set training features and labels
    train_dataset = OneStepPredictionDataset(train_data, train_time)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True
    )

    # Set testing features and labels
    test_dataset = OneStepPredictionDataset(test_data, test_time)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=no,
        shuffle=True
    )
    # Initialize model
    model = GeneralRNN(args)
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

            perf += rmse_error(test_y, test_p)

    return perf


class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN
    """

    def __init__(self, args):
        super(DiscriminatorNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Discriminator Architecture
        self.dis_rnn = torch.nn.GRU(
            input_size=self.hidden_dim,
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


def post_hoc_discriminator(ori_data, generated_data):
    args = {}
    args["device"] = "cuda"
    args["model_type"] = "gru"
    args["epochs"] = 1000
    args["batch_size"] = 128
    args["num_layers"] = 6
    args["padding_value"] = -1.0
    args["max_seq_len"] = 24
    args["padding_value"]=-1.0
    args["train_rate"] = 0.8
    args["learning_rate"] = 1e-3

    ori_data,ori_time=ori_data
    generated_data,generated_time=generated_data
    random_seed=random.randint(1, 100000)
    print('random_seed',random_seed)
    ori_train_data, ori_test_data, ori_train_time, ori_test_time = train_test_split(
        ori_data, ori_time, test_size=args['train_rate'],random_state=random_seed
    )
    generated_train_data, generated_test_data, generated_train_time, generated_test_time = train_test_split(
        generated_data, generated_time, test_size=args['train_rate'],random_state=random_seed
    )
    no, seq_len, dim = ori_data.shape
    args["hidden_dim"] = dim
    args_tuple = namedtuple('GenericDict', args.keys())(**args)
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
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=args['learning_rate'],weight_decay=1)
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
            D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(ori_logits, torch.ones_like(ori_logits))
            D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(generated_logits, torch.zeros_like(generated_logits))
            D_loss=D_loss_real+D_loss_fake
            # backward
            D_loss.backward()
            # optimize
            optimizer.step()

            running_real_loss += D_loss_real.item()
            running_fake_loss += D_loss_fake.item()
            running_loss +=D_loss.item()

        logger.set_description(f"batchnum: {len(train_dataloader)}, Epoch: {epoch},loss: {running_loss/len(train_dataloader):.4f}, real_loss: {running_real_loss/len(train_dataloader):.4f}, fake_loss: {running_fake_loss/len(train_dataloader):.4f}")
    # Evaluate the discriminator on the test set
    with torch.no_grad():
        discriminative_score = []
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
            print(y_pred_final.shape,y_label_final.shape,generated_logits.shape,ori_logits.shape)
            # acc_1=accuracy_score(y_label_final, y_pred_final)
            acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
            discriminative_score.append(np.abs(0.5 - acc))
            print('acc,score,acc_ori: ',acc,np.abs(0.5 - acc))
        print("discriminative_score by batch:",discriminative_score)

    return sum(discriminative_score)/len(discriminative_score)