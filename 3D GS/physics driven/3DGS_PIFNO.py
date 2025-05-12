import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as scio
import time
import os
import scipy.io
import plotly.graph_objects as go

from torch.nn.parameter import Parameter
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import scipy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float32)

torch.manual_seed(66)
np.random.seed(66)





class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, depth, act):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # the activation function#
        self.act = act

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))

        # Hidden layers
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, num_classes))

    def forward(self, x, final_act=False):
        for i in range(len(self.layers) - 1):
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)  # No activation after the last layer

        if final_act == False:
            return x
        else:
            return torch.relu(x)


class ModifiedMLP(nn.Module):
    def __init__(self, layers, activation):
        super().__init__()
        self.activation = activation

        # 初始化 U1, U2 参数
        self.U1 = nn.Parameter(torch.Tensor(layers[0], layers[1]))
        self.b1 = nn.Parameter(torch.Tensor(layers[1]))
        self.U2 = nn.Parameter(torch.Tensor(layers[0], layers[1]))
        self.b2 = nn.Parameter(torch.Tensor(layers[1]))

        # 创建中间线性层
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            in_dim = layers[i]
            out_dim = layers[i + 1]
            self.linears.append(nn.Linear(in_dim, out_dim))

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化 U1, U2 使用 Xavier 正态分布
        init.xavier_normal_(self.U1)
        init.zeros_(self.b1)
        init.xavier_normal_(self.U2)
        init.zeros_(self.b2)

        # 初始化线性层参数
        for linear in self.linears:
            init.xavier_normal_(linear.weight)
            init.zeros_(linear.bias)

    def forward(self, x):
        # 计算 U 和 V（基于原始输入）
        U = self.activation(torch.matmul(x, self.U1) + self.b1)
        V = self.activation(torch.matmul(x, self.U2) + self.b2)

        # 处理中间层（最后一层之前的所有层）
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
            x = x * U + (1 - x) * V  # 混合操作

        # 处理最后一层（无激活函数）
        x = self.linears[-1](x)
        return x


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width

        self.fc0 = nn.Linear(5, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        # self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        # self.bn0 = torch.nn.BatchNorm3d(self.width)
        # self.bn1 = torch.nn.BatchNorm3d(self.width)
        # self.bn2 = torch.nn.BatchNorm3d(self.width)
        # self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x





class RONNCell(nn.Module):
    ''' Recurrent convolutional neural network Cell '''

    def __init__(self, modes1, modes2, modes3, width, k, dt):
        super(RONNCell, self).__init__()

        # the initial parameters
        self.FNO3d = FNO3d(modes1, modes2, modes3, width)


        self.k = k
        self.dt = dt
        self.kweights = nn.Parameter(torch.Tensor(1, k))
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 使用 Xavier 均匀分布初始化
        init.xavier_uniform_(self.kweights)

    def forward(self, h):

        result = torch.einsum("mi,ihwdj->mhwdj", self.kweights, self.FNO3d(h))

        return result


class RONN(nn.Module):
    ''' Recurrent convolutional neural network layer '''

    def __init__(self, modes1, modes2, modes3, width, k, dt, init_h0, truth,
                 step=1, effective_step=[1]):

        super(RONN, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.k = k
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        # self.init_state = torch.nn.Parameter(torch.tensor(ini_state, dtype=torch.float32).cuda(), requires_grad=False)
        self.init_h0 = init_h0  # k*N
        self.truth = truth
        self.init_state = []
        self.dt = dt

        name = 'ronn_cell'
        cell = RONNCell(
            modes1=self.modes1,
            modes2=self.modes2,
            modes3=self.modes3,
            width=self.width,
            k=self.k,
            dt=self.dt)

        setattr(self, name, cell)
        self._all_layers.append(cell)

    def forward(self):

        internal_state = []
        loss_t = []
        rloss_t = []  # 计算残差


        for step in range(self.step - self.k):
            name = 'ronn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_h0  # k*N
                internal_state = h

            # forward torch.cat((x, grid), dim=-1)
            h = internal_state  # k*N
            # hidden state + output
            o = getattr(self, name)(h)  # N*1
            u = o[0, :, :, :, 0]
            v = o[0, :, :, :, 1]
            Lu = self.fd_laplacian_3d(u)
            Lv = self.fd_laplacian_3d(v)

            fu = Du * Lu - 20 * u * v ** 2 + FF * (1 - u)
            fv = Dv * Lv + 20 * u * v ** 2 - (FF + kk) * v

            resu = 137 / 60 * u - 5 * internal_state[4, :, :, :, 0] + 5 * internal_state[3, :, :, :, 0] \
                   - 10 / 3 * internal_state[2, :, :, :, 0] + 5 / 4 * internal_state[1, :, :, :, 0] \
                   - 1 / 5 * internal_state[0, :, :, :, 0] - self.dt * fu

            resv = 137 / 60 * v - 5 * internal_state[4, :, :, :, 1] + 5 * internal_state[3, :, :, :, 1] \
                   - 10 / 3 * internal_state[2, :, :, :, 1] + 5 / 4 * internal_state[1, :, :, :, 1] \
                   - 1 / 5 * internal_state[0, :, :, :, 1] - self.dt * fv

            res = torch.cat([resu.unsqueeze(0), resv.unsqueeze(0)], dim=0)
            rloss = (res ** 2).mean().unsqueeze(0).unsqueeze(0)

            loss = ((o - self.truth[step:step+1]) ** 2).mean().unsqueeze(0).unsqueeze(0)
            internal_state = torch.cat((internal_state[1:], o), dim=0)


            # after many layers output the result save at time step t
            if step in self.effective_step:
                loss_t.append(loss)
                rloss_t.append(rloss)


        return rloss_t, loss_t

    def spectral_laplacian_3d(self, u, Lx=80.0, Ly=80.0, Lz=80.0):
        """
        使用傅里叶谱方法计算三维张量的拉普拉斯算子
        :param u: 输入三维张量 (Nx, Ny, Nz)
        :param Lx: x方向计算域长度
        :param Ly: y方向计算域长度
        :param Lz: z方向计算域长度
        :return: 拉普拉斯算子 ∇²u (Nx, Ny, Nz)
        """
        # 获取网格尺寸
        Nx, Ny, Nz = u.shape
        device = u.device

        # 生成波数分量 (符合FFT频率排序)
        kx = 2 * np.pi * torch.fft.fftfreq(Nx, d=Lx/Nx).to(device)
        ky = 2 * np.pi * torch.fft.fftfreq(Ny, d=Ly/Ny).to(device)
        kz = 2 * np.pi * torch.fft.fftfreq(Nz, d=Lz/Nz).to(device)

        # 创建三维波数网格 (索引顺序需与网格生成一致)
        Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')

        # 计算波数平方和
        K_squared = Kx ** 2 + Ky ** 2 + Kz ** 2

        # 执行三维傅里叶变换
        U = torch.fft.fftn(u)

        laplacian = torch.fft.ifftn(-K_squared * U).real

        return laplacian

    def fd_laplacian_3d(self, u, Lx=80.0, Ly=80.0, Lz=80.0):
        """
        使用六点差分法（finite difference）计算3D拉普拉斯算子
        :param u: 输入3D张量 (Nx, Ny, Nz)
        :param dx, dy, dz: x, y, z方向上的网格步长
        :return: ∇²u 的张量 (Nx, Ny, Nz)
        """
        # 周期性边界使用 torch.roll
        Nx, Ny, Nz = u.shape
        dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
        lap = (
                (torch.roll(u, shifts=+1, dims=0) + torch.roll(u, shifts=-1, dims=0) - 2 * u) / dx ** 2 +
                (torch.roll(u, shifts=+1, dims=1) + torch.roll(u, shifts=-1, dims=1) - 2 * u) / dy ** 2 +
                (torch.roll(u, shifts=+1, dims=2) + torch.roll(u, shifts=-1, dims=2) - 2 * u) / dz ** 2
        )
        return lap



    def test(self):

        internal_state = []
        outputs_u = []
        outputs_v = []

        for step in range(self.step - self.k):
            name = 'ronn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_h0  # k*N
                internal_state = h

            # forward torch.cat((x, grid), dim=-1)
            h = internal_state  # k*N
            # hidden state + output
            o = getattr(self, name)(h)  # N*1
            u = o[:, :, :, :, 0]
            v = o[:, :, :, :, 1]

            internal_state = torch.cat((internal_state[1:], o), dim=0)

            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs_u.append(u)
                outputs_v.append(v)

        return outputs_u, outputs_v


def train(model, n_iters, learning_rate, cont=True):
    # define some parameters

    # truth
    # model
    if cont:
        model, optimizer, scheduler, train_loss_list, train_rloss_list = load_model(model)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = learning_rate
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.99)
        train_loss_list = []
        train_rloss_list = []

    M = torch.triu(torch.ones(length, length), diagonal=1).cuda()

    for epoch in range(n_iters):
        # input: [time,. batch, channel, height, width]
        optimizer.zero_grad()
        rloss_t, loss_t = model()  # output is a list
        rloss_t = torch.cat(tuple(rloss_t), dim=1)
        loss_t = torch.cat(tuple(loss_t), dim=1)

        #  data loss
        loss = loss_t.mean()

        # physics loss
        rW = torch.exp(-500 * (rloss_t @ M)).detach()
        rloss_w = 1000 * (rW * rloss_t).mean()
        rloss = (rloss_t).mean()
        rloss_w = rloss_w

        rloss_w.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] lossuv: %.7f, rloss: %.7f, '
              'rloss_w: %.7f,' % (
                  (epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), loss.item(), rloss.item(),
                  rloss_w.item()))
        train_loss_list.append(loss.item())
        train_rloss_list.append(rloss.item())
        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print('save model at {}'.format(epoch+1))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_list': train_loss_list,
                'train_rloss_list': train_rloss_list,
            }, './3D GS/physics driven/model/checkpoint_pifno2.pt')
    return train_loss_list, train_rloss_list


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./3D GS/physics driven/model/checkpoint_pifno2.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
    train_loss_list = checkpoint['train_loss_list']
    train_rloss_list = checkpoint['train_rloss_list']
    return model, optimizer, scheduler, train_loss_list, train_rloss_list


def postprocess3D(data, x, y, z, isU=True, flag=None, num=None, t=None):

    appd = ['Prediction', 'Truth', 'Error']
    uv = ['v', 'u']
    values = data

    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=0.3 if isU else 0.1,
        isomax=0.5 if isU else 0.2,
        opacity=0.2,
        colorscale='RdBu' if isU else 'RdGy',  # 'RdBu', 'RdGy'
        surface_count=3,  # number of isosurfaces, 2 by default: only min and max
    ))

    fig.update_layout(
        # title='3D Isosurface',
        # title_font=dict(size=24, family='Arial', color='black'),  # 标题字体
        # font=dict(size=14, family='Arial'),  # 全局默认字体
        width=800,
        height=800,
        # margin=dict(l=100, r=100, t=100, b=100),
        scene_camera=dict(
            eye=dict(x=1.2, y=1.2, z=1.5)  # 调整 z 使主体上移
        ),

        scene=dict(
            xaxis=dict(
                title='X',
                title_font=dict(size=20, family='Arial', color='black'),  # X轴标题
                tickfont=dict(size=18, family='Arial')  # X轴刻度字体
            ),
            yaxis=dict(
                title='Y',
                title_font=dict(size=20, family='Arial', color='black'),  # Y轴标题
                tickfont=dict(size=18, family='Arial')
            ),
            zaxis=dict(
                title='Z',
                title_font=dict(size=20, family='Arial', color='black'),  # Z轴标题
                tickfont=dict(size=18, family='Arial')
            ),

        )
    )

    # fig.show()
    fig.write_image('./3D GS/figures/PIFNO/Iso_surf_pifno_%s_%s_%.2f.png' % (uv[isU], appd[flag], t[num]), engine='orca', scale=2)
    plt.close('all')

def postprocess3D_error(data, x, y, z, isU=True, flag=None, num=None, t=None):

    appd = ['Prediction', 'Truth', 'Error']
    uv = ['v', 'u']
    values = data

    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=0.003,
        isomax=0.03,
        opacity=0.2,
        colorscale='YlGnBu',
        surface_count=3,  # number of isosurfaces, 2 by default: only min and max
    ))

    fig.update_layout(
        # title='3D Isosurface',
        # title_font=dict(size=24, family='Arial', color='black'),  # 标题字体
        # font=dict(size=14, family='Arial'),  # 全局默认字体
        width=800,
        height=800,
        # margin=dict(l=100, r=100, t=100, b=100),
        scene_camera=dict(
            eye=dict(x=1.2, y=1.2, z=1.5)  # 调整 z 使主体上移
        ),

        scene=dict(
            xaxis=dict(
                title='X',
                title_font=dict(size=20, family='Arial', color='black'),  # X轴标题
                tickfont=dict(size=18, family='Arial')  # X轴刻度字体
            ),
            yaxis=dict(
                title='Y',
                title_font=dict(size=20, family='Arial', color='black'),  # Y轴标题
                tickfont=dict(size=18, family='Arial')
            ),
            zaxis=dict(
                title='Z',
                title_font=dict(size=20, family='Arial', color='black'),  # Z轴标题
                tickfont=dict(size=18, family='Arial')
            ),

        )
    )

    # fig.show()
    fig.write_image('./3D GS/figures/PIFNO/Iso_surf_pifno_%s_%s_%.2f.png' % (uv[isU], appd[flag], t[num]), engine='orca', scale=2)
    plt.close('all')



if __name__ == '__main__':

    ################# prepare the input dataset ####################

    # 加载数据
    uv_sol = scipy.io.loadmat('./data/3DGS.mat')['UV']  # (301,2,32,32,32)
    t = np.linspace(0, 300 * 0.2, 301)  # (301,)

    sub = 2
    uv_sol = uv_sol[::sub].transpose(0, 2, 3, 4, 1)
    t = t[::sub]

    #  equation coefficient
    Du = 4
    Dv = 2
    FF = 0.5
    kk = 1.1

    Lx, Ly, Lz = 80, 80, 80
    Nx, Ny, Nz = 32, 32, 32
    dx = Lx / Nx  # 空间步长
    dy = Ly / Ny  # 空间步长
    dz = Lz / Nz  # 空间步长
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    z = np.linspace(-Lz / 2, Lz / 2, Nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # grid = np.stack((X, Y, Z), axis=-1)


    # 假设以下是你为模型提供的输入参数
    k = 5  # num of input u
    d = 3  # 维数

    length = 40
    step = k+length  # 时间步长
    effective_step = list(range(0, step))

    modes1 = 8
    modes2 = 8
    modes3 = 8
    width = 32

    dt = t[1]-t[0]
    U0 = uv_sol[:k, :, :, :, :]
    U = uv_sol[:step, :, :, :, :]

    init_h0 = torch.tensor(U0, dtype=torch.float32).cuda()  # (k,128,128)


    target = torch.tensor(U, dtype=torch.float32).cuda()
    truth = target[k:]

    # 创建 RONN 模型
    model = RONN(modes1, modes2, modes3, width, k, dt, init_h0, truth, step, effective_step).cuda()

    n_iters = 0
    learning_rate = 1e-3


    # train the model
    start = time.time()
    cont = True  # if continue training (or use pretrained model), set cont=True
    train_loss_list, train_rloss_list = train(model, n_iters, learning_rate, cont=cont)
    end = time.time()

    print('The training time is: ', (end - start))


    plt.plot(train_loss_list)
    plt.plot(train_rloss_list)
    plt.yscale("log")
    plt.show()

    with torch.no_grad():
        outputs_u, outputs_v = model.test()

    outputs_u = torch.cat(tuple(outputs_u), dim=0)  # N*(T-k)
    outputs_u = outputs_u.cpu().detach().numpy()
    outputs_v = torch.cat(tuple(outputs_v), dim=0)  # N*(T-k)
    outputs_v = outputs_v.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()
    truth_u = truth[:, :, :, :, 0]
    truth_v = truth[:, :, :, :, 1]

    Lu_e = np.mean(np.sqrt((outputs_u - truth_u) ** 2), axis=(1, 2, 3))
    Lu = np.mean(np.sqrt(truth_u ** 2), axis=(1, 2, 3))
    Lv_e = np.mean(np.sqrt((outputs_v - truth_v) ** 2), axis=(1, 2, 3))
    Lv = np.mean(np.sqrt(truth_v ** 2), axis=(1, 2, 3))
    #
    erroru = Lu_e / Lu
    errorv = Lv_e / Lv
    plt.plot(erroru)
    plt.plot(errorv)
    plt.show()
    #
    #
    output = np.sqrt(outputs_u ** 2 + outputs_v ** 2)
    truth = np.sqrt(truth_u ** 2 + truth_v ** 2)
    L_e = np.mean(np.sqrt((output - truth) ** 2), axis=(1, 2, 3))
    L = np.mean(np.sqrt(truth ** 2), axis=(1, 2, 3))

    t = t[k:k + length]
    error = L_e / L
    plt.plot(t, error)
    plt.show()

