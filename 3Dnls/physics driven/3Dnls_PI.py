import torch
import torch.nn as nn
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
import torch.nn.init as init
import plotly.graph_objects as go

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float32)

torch.manual_seed(66)
np.random.seed(66)


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


class SinAct(nn.Module):
    def __init__(self):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class CNN(nn.Module):
    """卷积网络处理函数输入(如图像)"""

    def __init__(self, dt, k, input_channels=2, input_kernel_size=5, output_dim=2, width=3):
        super().__init__()
        padding = (input_kernel_size - 1) // 2
        self.conv_layers = nn.Sequential(
            # 输入形状: (batch_size, input_channels, 64, 64)
            nn.Conv3d(input_channels, input_channels, kernel_size=input_kernel_size, padding=padding),
            nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Linear(width ** 3, output_dim),
        )
        self.kweights = nn.Parameter(torch.Tensor(k, 1))

        torch.nn.init.xavier_normal_(self.kweights)
        self._initialize_weights()  # 初始化网络权重

        self.dt = dt

    def _initialize_weights(self):
        # 对卷积层权重和偏置进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 Xavier 正态分布初始化权重
                init.xavier_normal_(m.weight)
                m.weight.data = 0.02 * m.weight.data
                # 将偏置初始化为零
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 对全连接层使用 Xavier 正态分布初始化权重
                init.xavier_normal_(m.weight)
                # 将偏置初始化为零
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        # x形状: (batch_size, C, H, W)
        x1 = self.conv_layers(x)
        x = (self.kweights.view(5, 1, 1, 1, 1) * (self.dt * x1 + x)).sum(dim=0)

        return self.fc(x.reshape(2, -1))  # 输出形状: (batch_size, m)


class RCNNCell(nn.Module):
    ''' Recurrent convolutional neural network Cell '''

    def __init__(self, k, trunk_layers, width, dt, input_channels, input_kernel_size, output_dim):
        super(RCNNCell, self).__init__()

        self.CNN = CNN(dt, k, input_channels, input_kernel_size, output_dim, width)
        self.trunk_net = ModifiedMLP(trunk_layers, SinAct())
        self.p = output_dim

    def forward(self, h, y):
        # periodic padding, can also be achieved using 'circular' padding
        B = self.CNN(h)
        T = self.trunk_net(y)
        outputs = 1 / self.p * torch.matmul(T, B.T)
        # outputs = torch.sin(outputs)

        return outputs


class RCNN(nn.Module):
    ''' Recurrent convolutional neural network layer '''

    def __init__(self, k, trunk_layers, width, dt, output_dim, input_channels, input_kernel_size, init_h0, init_true, truth, x,
                 step=1, effective_step=[1]):

        super(RCNN, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.init_h0 = init_h0
        self.init_true = init_true
        self.truth = truth
        self.k = k
        self.x = x
        self.dt = dt
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []

        name = 'crnn_cell'
        cell = RCNNCell(
            k=k,
            trunk_layers=trunk_layers,
            width=width,
            dt=dt,
            input_channels=input_channels,
            input_kernel_size=input_kernel_size,
            output_dim=output_dim)

        setattr(self, name, cell)
        self._all_layers.append(cell)

    def forward(self, y):

        internal_state = []
        rlossu_t = []  # 计算残差
        rlossv_t = []  # 计算残差
        loss_t = []
        HO = y[:, 0:1]**2 + y[:, 1:2]**2 + y[:, 2:3]**2

        # 确保y需要梯度
        if not y.requires_grad:
            y.requires_grad_(True)

        for step in range(self.step - self.k):
            name = 'crnn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_h0
                internal_state = h
                internal_output = self.init_true

            # forward
            h = internal_state
            # hidden state + output
            o = getattr(self, name)(h, y)  # (N,2)
            loss = ((o - self.truth[step].T) ** 2).mean().unsqueeze(0).unsqueeze(0)

            u = o[:, 0:1]
            v = o[:, 1:2]

            # 计算梯度 ----------------------------------------------------
            if y.grad is not None:
                y.grad.zero_()  # 清空之前的梯度

            Lu = self.compute_laplacian_3d(u, y)
            Lv = self.compute_laplacian_3d(v, y)
            fu = -Lv + HO * v - (u ** 2 + v ** 2) * v - u + delta * v + P
            fv =  Lu - HO * u + (u ** 2 + v ** 2) * u - v - delta * u

            resu = 137 / 60 * u - 5 * internal_output[4, :, 0:1] + 5 * internal_output[3, :, 0:1] \
                   - 10 / 3 * internal_output[2, :, 0:1] + 5 / 4 * internal_output[1, :, 0:1] \
                   - 1 / 5 * internal_output[0, :, 0:1] - self.dt * fu

            resv = 137 / 60 * v - 5 * internal_output[4, :, 1:2] + 5 * internal_output[3, :, 1:2] \
                   - 10 / 3 * internal_output[2, :, 1:2] + 5 / 4 * internal_output[1, :, 1:2] \
                   - 1 / 5 * internal_output[0, :, 1:2] - self.dt * fv

            rlossu = (resu ** 2).mean().unsqueeze(0).unsqueeze(0)
            rlossv = (resv ** 2).mean().unsqueeze(0).unsqueeze(0)

            internal_output = torch.cat((internal_output[1:], o.unsqueeze(0)), dim=0)

            h1 = getattr(self, name)(h, self.x)  # m*1
            internal_state = torch.cat((internal_state[1:], self.transform(h1)), dim=0)

            # after many layers output the result save at time step t
            if step in self.effective_step:
                rlossu_t.append(rlossu)
                rlossv_t.append(rlossv)
                loss_t.append(loss)

        return rlossu_t, rlossv_t, loss_t

    def transform(self, h):
        channel_0 = h[:, 0:1]  # 提取第一个通道 (16384, 1)
        channel_1 = h[:, 1:2]  # 提取第二个通道 (16384, 1)

        reshaped_0 = channel_0.view(32, 32, 32)  # 第一个通道 (128, 128)
        reshaped_1 = channel_1.view(32, 32, 32)  # 第二个通道 (128, 128)

        stacked_tensor = torch.stack([reshaped_0, reshaped_1], dim=0)  # 沿通道维度堆叠
        final_tensor = stacked_tensor.unsqueeze(0)  # 或 stacked_tensor[None, ...]

        return final_tensor

    def compute_laplacian_3d(self, u, y):
        """
        计算三维空间中u对y的二阶导数之和（拉普拉斯算子）
        :param u: 形状为 (N, 1) 的张量，标量场值
        :param y: 形状为 (N, 3) 的张量，三维坐标 (y1, y2, y3)
        :return: 拉普拉斯算子值，形状为 (N, 1)
        """
        # 计算一阶导数（保留计算图）
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=y,
            grad_outputs=torch.ones_like(u),
            create_graph=True,  # 保留计算图用于二阶导数
            retain_graph=True  # 保留中间变量
        )[0]  # 形状变为 (N, 3)

        # 初始化拉普拉斯结果
        laplacian = torch.zeros_like(u)

        # 对每个坐标分量计算二阶导数（三维空间有3个分量）
        for i in range(y.shape[1]):
            # 提取第i个分量的梯度（x, y或z方向）
            grad_i = grad_u[:, i]  # 形状 (N,)

            # 计算该分量的二阶导数
            grad2_i = torch.autograd.grad(
                outputs=grad_i,
                inputs=y,
                grad_outputs=torch.ones_like(grad_i),
                retain_graph=True,  # 保持计算图完整
                create_graph=False  # 不需要更高阶导数
            )[0][:, i]  # 形状 (N,)

            # 累加到拉普拉斯结果
            laplacian[:, 0] += grad2_i

        return laplacian



    def test(self, y):

        internal_state = []
        outputs_u = []
        outputs_v = []
        # second_last_state = []

        for step in range(self.step - self.k):
            name = 'crnn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_h0
                internal_state = h

            # forward
            h = internal_state
            # hidden state + output
            o = getattr(self, name)(h, y)

            h1 = getattr(self, name)(h, self.x)  # m*1
            internal_state = torch.cat((internal_state[1:], self.transform(h1)), dim=0)

            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs_u.append(o[:, 0:1])
                outputs_v.append(o[:, 1:2])

        return outputs_u, outputs_v



def train(model, y, n_iters, learning_rate, cont=True):
    if cont:
        model, optimizer, scheduler, train_loss_list, train_rloss_list = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.99)
        train_loss_list = []
        train_rloss_list = []

    # for param_group in optimizer.param_groups:
    #    param_group['lr']=0.0012

    M = torch.triu(torch.ones(length, length), diagonal=1).cuda()

    for epoch in range(n_iters):
        # input: [time,. batch, channel, height, width]
        optimizer.zero_grad()
        # One single batch
        rlossu_t, rlossv_t, loss_t = model(y)  # output is a list
        rlossu_t = torch.cat(tuple(rlossu_t), dim=1)
        rlossv_t = torch.cat(tuple(rlossv_t), dim=1)
        loss_t = torch.cat(tuple(loss_t), dim=1)

        #  data loss

        loss = loss_t.mean()

        # physics loss
        rWu = torch.exp(-500 * (rlossu_t @ M)).detach()
        rWv = torch.exp(-500 * (rlossv_t @ M)).detach()
        rlossu_w = 1000 * (rWu * rlossu_t).mean()
        rlossv_w = 1000 * (rWv * rlossv_t).mean()
        rlossu = (rlossu_t).mean()
        rlossv = (rlossv_t).mean()
        rloss = rlossu + rlossv
        rloss_w = rlossu_w + rlossv_w

        rloss_w.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] lossuv: %.7f, rlossu: %.7f, '
              'rlossv: %.7f, rlossu_w: %.7f, rlossv_w: %.7f,' % (
              (epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), loss.item(), rlossu.item(),
              rlossv.item(), rlossu_w.item(), rlossv_w.item()))
        train_loss_list.append(loss.item())
        train_rloss_list.append(rloss.item())

        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print('save model at {}'.format(epoch + 1))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_list': train_loss_list,
                'train_rloss_list': train_rloss_list,
            }, './3Dnls/physics driven/model/checkpoint_PI.pt')
    return train_loss_list, train_rloss_list


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./3Dnls/physics driven/model/checkpoint_PI.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)
    train_loss_list = checkpoint['train_loss_list']
    train_rloss_list = checkpoint['train_rloss_list']
    return model, optimizer, scheduler, train_loss_list, train_rloss_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def postprocess3D(data, x, y, z, flag=None, num=None, t=None):



    appd = ['Prediction', 'Truth', 'Error']
    values = data

    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        # isomin=0.2,
        # isomax=1.5,
        isomin=0.01,
        isomax=0.05,
        opacity=0.2,
        colorscale='RdBu',  # 'BlueRed',
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
    fig.write_image('./3Dnls/figures/PIdeeponet/Iso_surf_PIdeeponet_%s_%.2f.png' % (appd[flag], t[num]), engine='orca', scale=2)
    plt.close('all')




if __name__ == '__main__':
    # 加载数据
    data = np.load('./data/3Dnls.npz')

    # 访问各个数组
    uv_sol = data['u_sol']  # (2501,2,32,32,32)
    t = data['t']  # (2501,)

    sub = 5
    uv_sol = uv_sol[::sub]
    t = t[::sub]

    #  equation coefficient
    delta = -2
    P = 0.75


    # equations
    # u_t = -Lap v + (x**2+y**2+z**2)*v - (u**2+v**2)v - u + delta * v + P
    # v_t =  Lap u - (x**2+y**2+z**2)*u + (u**2+v**2)u - v - delta * u

    Lx, Ly, Lz = 6, 6, 6
    Nx, Ny, Nz = 32, 32, 32
    dx = Lx / Nx  # 空间步长
    dy = Ly / Ny  # 空间步长
    dz = Lz / Nz  # 空间步长
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    z = np.linspace(-Lz / 2, Lz / 2, Nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X_all = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)



    # Adaptive Sample
    N = 20000
    distances = np.sqrt((X.flatten() - 0) ** 2 + (Y.flatten() - 0) ** 2 + (Z.flatten() - 0) ** 2)
    weights = 1 / (distances + 1e-6)
    indices = np.random.choice(len(X.flatten()), size=N, p=weights / weights.sum(), replace=False)
    # indices = np.random.choice(len(X.flatten()), size=N, replace=False)
    x = X_all[indices]



    # 假设以下是你为模型提供的输入参数
    k = 5  # num of input u

    U0 = uv_sol[:k, :, :, :, :]  # [k,2,128,128]
    width = U0.shape[-1]
    d = 3  # 维数

    length = 100
    step = k + length  # 时间步长
    effective_step = list(range(0, step))

    dt = t[1] - t[0]
    x_values = X_all
    y_values = x
    U = uv_sol[:step].reshape(step, 2, -1)  # [201, 2, 32*32*32]
    init_true = U[:k, :, indices].transpose(0, 2, 1)
    init_true = torch.tensor(init_true, dtype=torch.float32).cuda()

    truth = U[k:, :, indices]
    truth = torch.tensor(truth, dtype=torch.float32).cuda()
    ################# build the model #####################
    # define the mdel hyperparameters

    # 创建初始化的隐藏状态 (k * m)
    init_h0 = torch.tensor(U0, dtype=torch.float32).cuda()

    # 创建 x 和 y 输入数据
    x = torch.tensor(x_values, dtype=torch.float32).cuda()  # m * d 的输入
    y = torch.tensor(y_values, dtype=torch.float32).cuda()  # N * d 的输入

    output_dim = 100
    model = RCNN(
        k=k,
        trunk_layers=[3, 100, 100, output_dim],  # 输入为 d，输出为 100
        width=width,
        dt=dt,
        output_dim=output_dim,
        input_channels=2,
        input_kernel_size=5,
        init_h0=init_h0,
        init_true=init_true,
        truth=truth,
        x=x,
        step=step,
        effective_step=effective_step).cuda()

    n_iters = 0
    learning_rate = 1e-3
    # train the model
    start = time.time()
    cont = True  # if continue training (or use pretrained model), set cont=True
    train_loss_list, train_rloss_list = train(model, y, n_iters, learning_rate, cont=cont)
    end = time.time()

    print('The training time is: ', (end - start))

    plt.plot(train_loss_list)
    plt.plot(train_rloss_list, '--')
    plt.yscale("log")
    plt.show()


    with torch.no_grad():
        output_u, output_v = model.test(x)
    output_u = torch.cat(tuple(output_u), dim=1).T
    output_v = torch.cat(tuple(output_v), dim=1).T

    truth = U[k:, :, :]
    truth_u = truth[:, 0, :]
    truth_v = truth[:, 1, :]
    truth_u = truth_u.reshape(length, 32, 32, 32)
    truth_v = truth_v.reshape(length, 32, 32, 32)


    output_u = output_u.reshape(length, 32, 32, 32).cpu().detach().numpy()
    output_v = output_v.reshape(length, 32, 32, 32).cpu().detach().numpy()

    L_e = np.mean(np.sqrt((output_u - truth_u) ** 2 + (output_v - truth_v) ** 2), axis=(1, 2, 3))
    L = np.mean(np.sqrt(truth_u ** 2 + truth_v ** 2), axis=(1, 2, 3))


    t = t[k:k + length] - t[0]
    erroru = L_e / L
    # errorv = Lv_e / Lv
    plt.plot(t,erroru)
    # plt.plot(errorv)
    plt.show()

