import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch.nn.init as init

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
        w = 2 * torch.pi / 10
        k_x = torch.arange(1, Mx + 1).cuda()
        k_y = torch.arange(1, My + 1).cuda()
        # k_xx, k_yy = torch.meshgrid(k_x, k_y)
        # k_xx = k_xx.flatten()
        # k_yy = k_yy.flatten()
        x = torch.cat([torch.cos(k_x * w * x[:, 0:1]), torch.cos(k_y * w * x[:, 1:2]),
                       torch.sin(k_x * w * x[:, 0:1]), torch.sin(k_y * w * x[:, 1:2])
                       ], dim=1)


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
            nn.Conv2d(input_channels, input_channels, kernel_size=input_kernel_size, padding=padding),
            SinAct(),
        )

        self.fc = nn.Sequential(
            nn.Linear(width**2, output_dim),
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
        x = (self.kweights.view(5, 1, 1, 1) * (self.dt * x1 + x)).sum(dim=0)

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
        outputs = torch.sin(outputs)

        return outputs


class RCNN(nn.Module):

    ''' Recurrent convolutional neural network layer '''

    def __init__(self, k, trunk_layers, width, dt, output_dim, input_channels, input_kernel_size, init_h0, x,
                       step=1, effective_step=[1]):

        super(RCNN, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.init_h0 = init_h0
        self.k = k
        self.x = x
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

    def transform(self, h):
        channel_0 = h[:, 0:1]  # 提取第一个通道 (16384, 1)
        channel_1 = h[:, 1:2]  # 提取第二个通道 (16384, 1)

        reshaped_0 = channel_0.view(128, 128)  # 第一个通道 (128, 128)
        reshaped_1 = channel_1.view(128, 128)  # 第二个通道 (128, 128)

        stacked_tensor = torch.stack([reshaped_0, reshaped_1], dim=0)  # 沿通道维度堆叠
        final_tensor = stacked_tensor.unsqueeze(0)  # 或 stacked_tensor[None, ...]

        return final_tensor



def train(model, y, truth, n_iters, learning_rate, cont=True):

    if cont:
        model, optimizer, scheduler, train_loss_list = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.99)
        train_loss_list = []

    #for param_group in optimizer.param_groups:
    #    param_group['lr']=0.0012

    M = torch.triu(torch.ones(truth.shape[0], truth.shape[0]), diagonal=1).cuda()
    truth_u = truth[:, 0, :].T
    truth_v = truth[:, 1, :].T

    for epoch in range(n_iters):
        # input: [time,. batch, channel, height, width]
        optimizer.zero_grad()
        # One single batch
        pred_u, pred_v = model(y)  # output is a list
        pred_u = torch.cat(tuple(pred_u), dim=1)  # N*(T-k)
        pred_v = torch.cat(tuple(pred_v), dim=1)
        mse_loss = nn.MSELoss()
        lossu_t = ((pred_u - truth_u) ** 2).mean(dim=0, keepdim=True)
        lossv_t = ((pred_v - truth_v) ** 2).mean(dim=0, keepdim=True)

        Wu = torch.exp(-200 * (lossu_t @ M)).detach()
        Wv = torch.exp(-200 * (lossv_t @ M)).detach()
        lossu = 1 * (Wu * lossu_t).mean()
        lossv = 1 * (Wv * lossv_t).mean()

        lossu0 = mse_loss(pred_u, truth_u)
        lossv0 = mse_loss(pred_v, truth_v)
        loss0 = lossu0 + lossv0

        loss = lossu + lossv
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] lossu: %.7f, lossv: %.7f, wlossu: %.7f, '
              'wlossv: %.7f' % ((epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), lossu0.item(), lossv0.item(), lossu.item(), lossv.item()))
        train_loss_list.append(loss0.item())
        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print('save model at {}'.format(epoch + 1))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_list': train_loss_list,
            }, './2DRD/data driven/model/checkpoint_deeponet_cnn.pt')
    return train_loss_list

def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./2DRD/data driven/model/checkpoint_deeponet_cnn.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)
    train_loss_list = checkpoint['train_loss_list']
    return model, optimizer, scheduler, train_loss_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':

    # 加载数据
    data = np.load('./data/2DRD.npz')

    # 访问各个数组
    uv_sol = data['uv_sol']  # (8193,256,256)
    t = data['t']  # (8193,)

    sub = 1
    uv_sol = uv_sol[::sub]
    t = t[::sub]

    Mx = My = 7

    L = 10.0  # 空间域大小
    Nx = 256  # 空间网格数
    Ny = 256  # 空间网格数
    dx = L / Nx  # 空间步长
    dy = L / Ny  # 空间步长
    x = np.linspace(0, L, Nx, endpoint=False)
    y = np.linspace(0, L, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    X_all = np.stack((X, Y), axis=-1).reshape(-1, 2)
    XX_all = np.stack((X[::2, ::2], Y[::2, ::2]), axis=-1).reshape(-1, 2)


    # Adaptive Sample
    N = 10000
    # 计算每个点到中心的距离
    distances = np.sqrt((X.flatten() - 5) ** 2 + (Y.flatten() - 5) ** 2)
    # 计算离中心的距离权重，距离越近，权重越大
    weights = 1 / (distances + 1e-6)  # 加小常数避免除零
    # 对点进行加权随机选择
    indices = np.random.choice(len(X.flatten()), size=N, p=weights / weights.sum(), replace=False)
    # selected_indices = np.random.choice(len(X_flat), size=N, replace=False)
    x = X_all[indices]

    # 假设以下是你为模型提供的输入参数
    k = 5  # num of input u

    U0 = uv_sol[:k, :, ::2, ::2]  # [k,2,128,128]
    width = U0.shape[-1]
    d = 2  # 维数

    length = 300
    step = k + length  # 时间步长
    effective_step = list(range(0, step))

    dt = t[1] - t[0]
    x_values = XX_all
    y_values = x
    U = uv_sol[:step].reshape(step, 2, -1)  # [201, 2, 100, 100]
    truth = U[k:, :, indices]
    truth = torch.tensor(truth, dtype=torch.float32).cuda()

    # 创建初始化的隐藏状态 (k * m)
    init_h0 = torch.tensor(U0, dtype=torch.float32).cuda()

    # 创建 x 和 y 输入数据
    x = torch.tensor(x_values, dtype=torch.float32).cuda()  # m * d 的输入
    y = torch.tensor(y_values, dtype=torch.float32).cuda()  # N * d 的输入

    output_dim = 100
    model = RCNN(
        k=k,
        trunk_layers=[2*(Mx + My), 100, 100, output_dim],  # 输入为 d，输出为 100
        width=width,
        dt=dt,
        output_dim=output_dim,
        input_channels=2,
        input_kernel_size=5,
        init_h0=init_h0,
        x=x,
        step=step,
        effective_step=effective_step).cuda()


    n_iters = 0
    learning_rate = 1e-3
    # train the model
    start = time.time()
    cont = True   # if continue training (or use pretrained model), set cont=True
    train_loss_list = train(model, y, truth, n_iters, learning_rate, cont=cont)
    end = time.time()

    print('The training time is: ', (end-start))

    plt.plot(train_loss_list)
    plt.yscale("log")
    plt.show()

    # Do the forward inference

    x = torch.tensor(X_all, dtype=torch.float32).cuda()
    with torch.no_grad():
        output_u, output_v = model(x)
    output_u = torch.cat(tuple(output_u), dim=1).T
    output_v = torch.cat(tuple(output_v), dim=1).T
    truth = U[k:, :, :]

    truth_u = truth[:, 0, :]
    truth_v = truth[:, 1, :]
    truth_u = truth_u.reshape(length, 256, 256)
    truth_v = truth_v.reshape(length, 256, 256)

    output_u = output_u.reshape(length, 256, 256).cpu().detach().numpy()
    output_v = output_v.reshape(length, 256, 256).cpu().detach().numpy()

    e_u = output_u - truth_u
    e_v = output_v - truth_v
    L2_e = np.mean(np.sqrt(e_u ** 2 + e_v ** 2), axis=(1, 2))
    L2 = np.mean(np.sqrt(truth_u ** 2 + truth_v ** 2), axis=(1, 2))

    Lu_e = np.mean(np.sqrt((output_u - truth_u) ** 2), axis=(1, 2))
    Lu = np.mean(np.sqrt(truth_u ** 2), axis=(1, 2))
    Lv_e = np.mean(np.sqrt((output_v - truth_v) ** 2), axis=(1, 2))
    Lv = np.mean(np.sqrt(truth_v ** 2), axis=(1, 2))

    t = t[k:k + length]
    erroru = Lu_e / Lu
    errorv = Lv_e / Lv
    error = L2_e / L2
    plt.plot(t,error)
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.imshow(output_u[-k, :, :], aspect='auto',
               extent=[0, L, 0, L], origin='lower',
               cmap='viridis')
    plt.colorbar()
    # plt.title('Heatmap of u = exp(-(x - at)^2)')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(output_v[-k, :, :], aspect='auto',
               extent=[0, L, 0, L], origin='lower',
               cmap='viridis')
    plt.colorbar()
    # plt.title('Heatmap of u = exp(-(x - at)^2)')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.show()

