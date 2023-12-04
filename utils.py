import pandas as pd
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R
import numpy as np

### tensorboard
def weight_histograms_linear(writter, step, weights, layer_number):
    flattened_weights = weights.flatten()
    tag = f'layer_{layer_number}'
    writter.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')

def weight_histograms(writer, step, model: nn.Module):
    print('Visualizing model weights...')
    if (isinstance(model, nn.Linear)):
        weights = model.weight
        weight_histograms_linear(writer, step, weights, 0)
    elif (isinstance(model, nn.TransformerEncoder)):
        # print(model.transformer_encoder.layers[0].linear1.weight)
        for i, layer in enumerate(model.layers):
            weight_histograms_linear(writer, step, layer.linear1.weight, i*2+1)
            weight_histograms_linear(writer, step, layer.linear2.weight, i*2+2)
    else:
        for layer_number in range(len(model.layers)):
        # print(layer_number)
        # Get layer
            layer = model.layers[layer_number]
            if isinstance(layer, nn.Linear):
                weights = layer.weight
                weight_histograms_linear(writer, step, weights, layer_number)



def compute_poses(last_pose: np.ndarray, current_pose: np.ndarray):
    '''
    gets an 1D array with 6 values (translation and rotation dof) and a matrix for a rigid transform

    returns a 2D array. updates the last_pose value 
    '''
    translation = np.zeros([4,1])
    translation[3] = 1.

    translation[0:3] = np.atleast_2d(current_pose[0:3]).T
    rotation = R.from_euler('xyz', [current_pose[3],current_pose[4],current_pose[5]], degrees=True)
    current_pose = np.concatenate((rotation.as_matrix(), np.zeros((1,3))), axis=0)
    current_pose = np.concatenate((current_pose, translation), axis=1)

    last_pose = last_pose@current_pose

    return last_pose


def reconstruct_traj(prediction: np.ndarray, stamps: np.ndarray):
    '''
    Converts a relative poses to absolute poses
    get poses as tensors with dimensions (n,6)
    where 6 is the number of dof
    and n the number of measurements
    '''
    df = pd.DataFrame({'timestamp':[],
                            'tx':[],
                            'ty':[],
                            'tz':[],
                            'qx':[],
                            'qy':[],
                            'qz':[],
                            'qw':[]
                            })
    last_pose = np.eye(4)

    r = R.from_matrix(last_pose[:3,:3])
    r_quat = r.as_quat()
    t = last_pose[:,3]

    # # add row to dataframe as stamp tx ty tz qx qy qz qw
    df.loc[len(df.index)] = [0.0, t[0], t[1], t[2], r_quat[0], r_quat[1], r_quat[2], r_quat[3]]

    for i, c in zip(range(prediction.shape[1]), range(stamps.shape[1])):
        last_pose = compute_poses(last_pose, prediction[0,i])
        # transform rotation to quaternion
        r = R.from_matrix(last_pose[:3,:3])
        r_quat = r.as_quat()
        t = last_pose[:,3]

        # add row to dataframe with timestamps stamp tx ty tz qx qy qz qw
        df.loc[len(df.index)] = [stamps[0,c].item(), t[0], t[1], t[2], r_quat[0], r_quat[1], r_quat[2], r_quat[3]]

    return df