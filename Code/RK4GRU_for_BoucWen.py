################################################## Vesion Notes  #######################################################
#     RK4GRU               # space state vector[x  xdot  p ]  # RK GRU_step = 100   # input [x  xdot  g  xdot2 ]
#                            # test_example = 50
######################################################################################################
import torch
from Uitls.RK4GRU_main import main
######################################################################################################
class Args:
    def __init__(self) -> None:
        ### 结构模型参数
        self.dt = 0.02
        self.SV_feature = 3                        # space state  [x  xdot p ]
        ### 网络参数
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = [4,12,1] # 网络层数          # input [x  xdot  p  xdot2 ]   output [pdot ]  # [ xdot2=-p-exc]
        self.GRU_layers = [3,3]  # 网络GRU层数
        ### 训练参数
        self.batch_size = 64  #  batch size
        self.seq_len = 1  # RK GRU  输入状态
        self.gru_step = 100 # RK GRU 预测步长
        self.stage1 = 0.2;  self.stage2 = 0.5
        # 学习率
        self.lr = 0.1
        self.lr_step = 100
        self.lr_gamma = 0.9
        # Epoch
        self.epochs = 2000
        self.valper = 1
        ### 路径
        self.train_example = 1 ;  self.test_example = 50
        self.data_path =r'Data/data_boucwen.mat'
        self.modelsave_path = r'Results/RK4GRU/'
if __name__ == '__main__':
    args = Args()
    rho = main(args)
    print('CC average min{}'.format(rho))










