import torch
import torch.utils.data as dataUtils
from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor

import numpy as np
import pandas as pd
import argparse
from tempfile import TemporaryDirectory
import os
import time
from datetime import datetime

import model
import data
import utils


def list_of_strings(arg):
    return arg.split(',')

parser = argparse.ArgumentParser(description='Self-Attention Odometry Fusion')

parser.add_argument('--train', action='store_true',
                    help='whether or not to train the fusion model')
parser.add_argument('--test', action='store_true',
                    help='whether or not to test the fusion model')
parser.add_argument('--model_name', type=str,
                    help='name of the file to be loaded from ./models dir if --train is not selected')
parser.add_argument('--cuda', action='store_true',
                    help="use CUDA")
# MODEL PARAMS                    
parser.add_argument('--pos_encoder', default='dummy', nargs='?', const='dummy', choices=['dummy', 't2v'],
                    help='choose between a dummy and a t2v temporal representation (default = dummy)')
parser.add_argument('--activation', default='relu', nargs='?', const='relu', choices=['gelu', 'relu'],
                    help='choose between a gelu and a relu activation function (default = relu)')
parser.add_argument('--d_output', type=int, default=6,
                    help='output DoF (default = 6)')
parser.add_argument('--d_model', type=int, default=20,
                    help='input data length (default = 20 = 6*3+2). careful: dummy->+2 and t2v->*2')
parser.add_argument('--nhead', type=int, default=5,
                    help='number of attention heads. must be a divisor of d_model! (default = 6)')
parser.add_argument('--d_hid', type=int, default=128,
                    help='FNN dimension (default = 128)')
parser.add_argument('--nlayers', type=int, default=8,
                    help='number of transformer encoder layers (default = 8)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout rate (default = 0.0)')
parser.add_argument('--seed', type=int, default=42,
                    help='train and val dataset spliter seed (default = 42)')
parser.add_argument('--bsize', type=int, default=10,
                    help='batch size (default = 10)')
parser.add_argument('--criterion', default='L1', nargs='?', const='L1', choices=['L1', 'MSE', 'Huber'],
                    help='criterion. L1 seemed to produce the best results in translation')
parser.add_argument('--optimizer', default='Adam', nargs='?', const='Adam', choices=['SGD', 'Adam', 'AdamW'],
                    help='optimizer')
parser.add_argument('--warmup', type=int, default=10,
                    help='warmup epochs (default = 10)')
parser.add_argument('--epochs', type=int, default=20,
                    help='epochs (default = 20)')
parser.add_argument('--factor', type=float, default=0.05,
                    help='warmup factor (default = 0.05)')
parser.add_argument('-n_seq', type=int, default=2,
                    help='number of available sequences')
parser.add_argument('--test_sequence', type=int, default=1,
                    help='sequence used for testing')
parser.add_argument('--seq_len', type=int, default=5,
                    help='input sequence length (default=5)')
parser.add_argument('--data', type=str, default='./data/sample',
                    help='data location')
parser.add_argument('--target', type=str, default='rtk_xsens',
                    help='odom groundtruth')
parser.add_argument('--features', type=list_of_strings, default=['icp_rtabmap','gps_imu', 'ORB'],
                    help='modalities')
parser.add_argument('--DoF', type=list_of_strings, default=['tx','ty','tz','roll','pitch','yaw'],
                    help='DoF')
parser.add_argument('--stamp', type=str, default='stamp',
                    help='stamp')
                                        

args = parser.parse_args()
if args.d_model % args.nhead != 0:
    raise Exception('d_model must be divisible by nhead')
    
# print(args)
# exit()

device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')
print(f"Using {device}!")


# prepare data
# there are certainly smarter ways to handle data
sequences = list(range(1,args.n_seq))
test_sequences = [args.test_sequence]
train_sequences = [n for n in sequences if n not in test_sequences]
df_train_data = {}
df_test_data = {}

train_datasets = []
test_datasets = []

### training data
for n in train_sequences:
    df_train_data[n] = pd.read_csv(f'{args.data}/{n}/merged.csv', header=[0,1], index_col=0, skipinitialspace=True)

for key in df_train_data.keys():
    aux_dataset = data.SequenceDataset(
        df_train_data[key],
        target = args.target,
        features = args.features,
        fields = args.DoF,
        stamp = [args.stamp],
        sequence_length = args.seq_len,
        device = device
    )
    train_datasets.append(aux_dataset)

full_train_dataset, full_val_dataset = dataUtils.random_split(dataUtils.ConcatDataset(train_datasets), [0.9,0.1], torch.Generator().manual_seed(args.seed))

train_dataloader = DataLoader(
    full_train_dataset,
    batch_size = args.bsize,
    shuffle = True,
    num_workers = 0
)

### test data
for n in test_sequences:
    df_test_data[n] = pd.read_csv(f'{args.data}/{n}/merged.csv', header=[0,1], index_col=0, skipinitialspace=True)

for key in df_test_data.keys():
    aux_dataset = data.SequenceDataset(
        df_test_data[key],
        target = args.target,
        features = args.features,
        fields = args.DoF,
        stamp = [args.stamp],
        sequence_length = args.seq_len,
        device = device
    )
    test_datasets.append(aux_dataset)

full_test_dataset = dataUtils.ConcatDataset(test_datasets)

test_dataloader = DataLoader(
    full_test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0
)


fusion_model = model.TransformerModel(args.d_output, args.d_model, args.nhead, args.d_hid, args.nlayers, args.dropout).to(device)
if args.criterion == 'MSE':
    criterion = nn.MSELoss(reduction='none')
elif args.criterion == 'Huber':
    criterion = nn.HuberLoss(reduction='none')
else:
    criterion = nn.L1Loss(reduction='none')
lr = 1.
### TODO
# betas as parameters
if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(fusion_model.parameters(), lr=lr)
elif args.optimizer == 'AdamW':
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
else:
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
### warmup probably not very important as normalization is done before the attention mechanism
def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (model_size**(-0.5) * min(step**(-0.5), step * warmup**(-1.5)))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda step: rate(step, args.d_model, factor=args.factor, warmup=args.warmup)
)


def train(fusion_model: nn.Module) -> float:
    fusion_model.train()
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(args.seq_len).to(device)
    last_i = 0
    
    for i, batch in enumerate(train_dataloader):
        data = batch[0]
        stamp = batch[1]
        targets = batch[2]

        fusion_model.zero_grad()

        output = fusion_model(data, stamp, src_mask)
        loss = criterion(output, targets)
        loss.mean().backward()
        optimizer.step()

        total_loss += loss.mean().item()
        last_i = i

    return total_loss/last_i

def evaluate(fusion_model: nn.Module) -> float:
    fusion_model.eval()
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(args.seq_len).to(device)
    last_i = 0

    with torch.no_grad():
        for i, batch in enumerate(train_dataloader):
            data = batch[0]
            stamp = batch[1]
            targets = batch[2]

            output = fusion_model(data, stamp, src_mask)
            loss = criterion(output, targets)
            total_loss += loss.mean().item()

            last_i = i
    return total_loss/last_i



if args.train:
    print("Training...")
    writer = utils.SummaryWriter()
    epochs = args.epochs
    interval = 1

    train_losses = []
    val_losses = []

    min_val_loss = float('inf')
    min_loss = float('inf')

    with TemporaryDirectory() as tempdir:
        best_model_params = os.path.join(tempdir, 'best_model_params.pt')

        for epoch in range(1, epochs+1):
            print(epoch)
            epoch_start_time = time.time()
        
            loss = train(fusion_model)
            train_losses.append(loss)
            val_loss = evaluate(fusion_model)
            val_losses.append(val_loss)

            elapsed = time.time() - epoch_start_time

            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                f'valid loss {val_loss:5.2f}')
            print('-' * 89)

            scheduler.step()

            if(epoch % interval == 0):
                writer.add_scalars('Losses', {'Training loss': loss, 'Validation loss': val_loss}, epoch)
                writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)

            if(loss < min_loss):
                torch.save(fusion_model.state_dict(), best_model_params)
                min_loss = loss

        fusion_model.load_state_dict(torch.load(best_model_params))

    dt_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    torch.save(fusion_model.state_dict(), f'./models/{dt_str}_seq_{args.test_sequence}.pt')


if args.test:
    ## not really testing. here we export the poses predicted by the fusion model
    print("Testing...")

    # load the model
    # if(not args.train):
    #     fusion_model.load_state_dict(torch.load(f'./models/{args.model_name}'))

    fusion_model.eval()

    with torch.no_grad():
        full_stamp = np.empty((1,0))
        out_traj = np.empty((1,0,6))

        src_mask = model.generate_square_subsequent_mask(args.seq_len).to(device)

        for i, batch in enumerate(iter(test_dataloader)):
            # not doing this here, but we could get only the last N predictions, as the first seq_len-N predictions dont have the same amount of information to attent (or predict one pose at a time with a MLP at the end of the encoder).
            # here we dont do that
            last_N = args.seq_len
            if(i%last_N == 0):
                in_data = batch[0]
                stamp = batch[1]
                targets = batch[2]

                output = fusion_model(in_data, stamp, src_mask).cpu().numpy()

                out_traj = np.concatenate((out_traj, output[:,-last_N:]), 1)
                full_stamp = np.concatenate((full_stamp, stamp[-last_N:].cpu()), 1)

        df = utils.reconstruct_traj(out_traj, full_stamp)
        df.to_csv(f'./results/{args.model_name}.txt', header=None, index=False, sep=' ') ### rpg_trajectory_evaluation format
