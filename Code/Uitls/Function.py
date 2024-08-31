import os
import torch
import shutil
import random
import sys
import contextlib
import numpy as np
import scipy.io
from tqdm import tqdm

def safemakefile(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def deletemakefile(save_path):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    safemakefile(save_path)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

### tqdm   输出不错行
class DummyFile:
    def __init__(self, file):
        if file is None:
            file = sys.stderr
        self.file = file
    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)
@contextlib.contextmanager
def redirect_stdout(file=None):
    if file is None:
        file = sys.stderr
    old_stdout = file
    sys.stdout = DummyFile(file)
    yield
    sys.stdout = old_stdout


### 多步法错位异步生成训练集输入输出
def one_step_track_RK4GRU(track,args):
    sl = args.seq_len ; gs = args.gru_step
    input_np = np.zeros([track.shape[0]*(track.shape[2]+1-sl-gs),sl,track.shape[1]]) # [num,seq_len,SV_feature]
    output_np = np.zeros([track.shape[0]*(track.shape[2]+1-sl-gs),gs,track.shape[1]]) # [num,gru_step,SV_feature]
    idx = 0
    for mi in range(track.shape[0]):
        for ni in range(track.shape[2]+1-sl-gs):
            for ii in range(sl):
                 input_np[idx,ii,:] = track[mi,:,ni+ii]
            for oi in range(gs):
                output_np[idx,oi,:] = track[mi,:,ni+sl+oi]
            idx= idx + 1
    return torch.from_numpy(input_np) ,torch.from_numpy(output_np)


# Scheduled Sampling框架
class ScheduledSampler():
    def __init__(self, total_epochs, start, end):
        self.total_epochs = total_epochs
        self.start= start
        self.end = end
    def teacher_forcing_ratio(self, current_epoch):
        if current_epoch < self.total_epochs * self.start:
            return 1.0, 1
        elif current_epoch < self.total_epochs * self.end:
            alpha = (current_epoch - self.total_epochs * self.start) / (self.total_epochs * (self.end - self.start))
            return 1.0 - alpha, 2
        else:
            return 0.0, 3   # tf_ratio, stage

def finite_difference(n,dt):               # 中心有限差分 n为seq_len
    phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([n - 3, ])])  #
    temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate([np.zeros([n - 3, ]), np.array([1 / 2, -2, 3 / 2])])
    Phi_t0 = (1/dt) *np.concatenate([np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0)
    Phi_t0 = np.reshape(Phi_t0, [1, n, n])
    return  Phi_t0   # [1, n, n]


def split_gm_GRU(input_array,output_array):  # [x xdot exc xdot2]   # [1119, 1 , 4]   # [1119,100, 4]
    # 提取的数据                                  0   1   2    3
    gm = np.concatenate((input_array[:, :, -2], output_array[:, :, -2]), axis=1)
    gm = gm[:,:,np.newaxis]
    gm_tensor  = torch.from_numpy(gm)   # [1119,101,1]
    # 调整input和output的形状
    x_xdot_input = input_array[:, :, :-2]   # [1119,1,2]
    g_input= - input_array[:, :, 2] - input_array[:, :, 3]    # [1119,1]
    input_ = np.concatenate((x_xdot_input, g_input[:, :, np.newaxis]), axis=2)  # [1119,1,3]
    x_xdot_output = output_array[:, :, :-2]  # [1119,100,2]
    g_output= - output_array[:, :, 2] - output_array[:, :, 3]  # [1119,100]
    output_ = np.concatenate((x_xdot_output, g_output[:, :, np.newaxis]), axis=2) # [1119,100,3]
    input_tensor = torch.from_numpy(input_)  # [1119,1,3]
    output_tensor = torch.from_numpy(output_)  # [1119,100,3]
    return input_tensor, output_tensor,gm_tensor  # [1119,1,3]   # [1119,100,3]    # [1119,101,1]


def load_matdata(args, runmodel = None, test_random_indices = None):
    mat = scipy.io.loadmat(args.data_path)
    t = mat['time']
    n1 = 1    # int(dt / 0.005)
    if runmodel == 'train':
        if test_random_indices == None:
            num_examples = args.train_example
            total_examples = mat['input_pred_tf'].shape[0]
            random_indices = np.random.choice(total_examples, num_examples, replace=False)
            ag_data = mat['input_pred_tf'][random_indices, 2::n1]
            u_data = mat['target_pred_X_tf'][random_indices, 2::n1]
            ut_data = mat['target_pred_Xd_tf'][random_indices, 2::n1]
            utt_data = mat['target_pred_Xdd_tf'][random_indices, 2::n1]
            ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
            u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
            ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
            utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])
            return random_indices,np.concatenate((u_data,ut_data,ag_data,utt_data), axis=2)     # [x,1499,4]
        else:
            random_indices = test_random_indices
            ag_data = mat['input_pred_tf'][random_indices, 2::n1]
            u_data = mat['target_pred_X_tf'][random_indices, 2::n1]
            ut_data = mat['target_pred_Xd_tf'][random_indices, 2::n1]
            utt_data = mat['target_pred_Xdd_tf'][random_indices, 2::n1]
            ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
            u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
            ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
            utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])
            return random_indices,np.concatenate((u_data,ut_data,ag_data,utt_data), axis=2)     # [x,1499,4]
    elif runmodel == 'test':
        if test_random_indices == None:
            num_examples = args.test_example
            total_examples = mat['input_tf'].shape[0]
            random_indices = np.random.choice(total_examples, num_examples, replace=False)
            ag_pred = mat['input_tf'][random_indices, 2::n1]  # ag, ad, av
            u_pred = mat['target_X_tf'][random_indices, 2::n1]
            ut_pred = mat['target_Xd_tf'][random_indices, 2::n1]
            utt_pred = mat['target_Xdd_tf'][random_indices, 2::n1]
            ag_pred = ag_pred.reshape([ag_pred.shape[0], ag_pred.shape[1], 1])
            u_pred = u_pred.reshape([u_pred.shape[0], u_pred.shape[1], 1])
            ut_pred = ut_pred.reshape([ut_pred.shape[0], ut_pred.shape[1], 1])
            utt_pred = utt_pred.reshape([utt_pred.shape[0], utt_pred.shape[1], 1])
            return  random_indices,np.concatenate((u_pred,ut_pred,ag_pred,utt_pred), axis=2)    # [x,1499,4]
        else:
            random_indices = test_random_indices
            ag_pred = mat['input_tf'][random_indices, 2::n1]  # ag, ad, av
            u_pred = mat['target_X_tf'][random_indices, 2::n1]
            ut_pred = mat['target_Xd_tf'][random_indices, 2::n1]
            utt_pred = mat['target_Xdd_tf'][random_indices, 2::n1]
            ag_pred = ag_pred.reshape([ag_pred.shape[0], ag_pred.shape[1], 1])
            u_pred = u_pred.reshape([u_pred.shape[0], u_pred.shape[1], 1])
            ut_pred = ut_pred.reshape([ut_pred.shape[0], ut_pred.shape[1], 1])
            utt_pred = utt_pred.reshape([utt_pred.shape[0], utt_pred.shape[1], 1])
            return  random_indices,np.concatenate((u_pred,ut_pred,ag_pred,utt_pred), axis=2)    # [x,1499,4]








if __name__ == '__main__':
    from RK4PIGRU_for_BoucWen import Args
    args = Args()
    args.data_path =r'../Data/data_boucwen.mat'
    load_matdata(args, 'test')







