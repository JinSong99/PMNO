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

def tonp(tensor):
    """ Torch to Numpy """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

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
            nn.Conv2d(input_channels, input_channels, kernel_size=input_kernel_size, padding=padding),
            nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Linear(width**2, output_dim),
        )
        self.kweights = nn.Parameter(torch.Tensor(k, 1))
        self.k = k

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
        x = (self.kweights.view(self.k, 1, 1, 1) * (self.dt * x1 + x)).sum(dim=0)

        return self.fc(x.reshape(1, -1))  # 输出形状: (batch_size, m)


class RCNNCell(nn.Module):
    ''' Recurrent convolutional neural network Cell '''
   
    def __init__(self, k, trunk_layers, width, dt, input_channels, input_kernel_size, output_dim):

        super(RCNNCell, self).__init__()

        self.CNN = CNN(dt, k, input_channels, input_kernel_size, output_dim, width)
        self.trunk_net = ModifiedMLP(trunk_layers, nn.Tanh())
        self.p = output_dim



    def forward(self, h, y):

        # periodic padding, can also be achieved using 'circular' padding
        B = self.CNN(h)
        T = self.trunk_net(y)
        outputs = 1 / self.p * torch.matmul(T, B.T)
        outputs = outputs * (y[:, 0:1] + L/2) * (L/2 - y[:, 0:1]) * (y[:, 1:2] + L/2) * (L/2 - y[:, 1:2])

        return outputs


class RCNN(nn.Module):

    ''' Recurrent convolutional neural network layer '''

    def __init__(self, k, trunk_layers, width, dt, t, output_dim, input_channels, input_kernel_size, init_h0, x, y0, x_sample,
                       step=1, effective_step=[1]):

        super(RCNN, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.init_h0 = init_h0
        self.k = k
        self.x = x
        self.y0 = y0
        self.x_sample = x_sample
        self.t = t
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


    def forward(self):

        internal_state = []
        loss_t = []
        y = self.y0
        t = self.t

        for step in range(self.step - self.k):
            name = 'crnn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_h0
                internal_state = h
                y = self.y0

            # forward
            h = internal_state
            # hidden state + output
            o = getattr(self, name)(h, y)

            t += self.dt
            u_true = u_ana(y[:, 0:1], y[:, 1:2], t)
            loss = ((o - u_true) ** 2).mean(dim=0, keepdim=True)

            h1 = getattr(self, name)(h, self.x)  # m*1
            internal_state = torch.cat((internal_state[1:], self.transform(h1)), dim=0)

            # after many layers output the result save at time step t
            if step in self.effective_step:
                loss_t.append(loss)

        return loss_t

    def transform(self, h):

        reshaped = h.view(100, 100)  # 第一个通道 (128, 128)
        final_tensor = reshaped.unsqueeze(0).unsqueeze(0)  # 或 stacked_tensor[None, ...]

        return final_tensor

    def test(self, y):

        internal_state = []
        outputs = []

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
                outputs.append(o)

        return outputs



def train(model, n_iters, learning_rate, cont=True):

    if cont:
        model, optimizer, scheduler, train_loss_list = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.99)
        train_loss_list = []

    # for param_group in optimizer.param_groups:
    #    param_group['lr']=0.0005

    M = torch.triu(torch.ones(length, length), diagonal=1).cuda()

    for epoch in range(n_iters):
        # input: [time,. batch, channel, height, width]
        optimizer.zero_grad()
        # One single batch
        loss_t = model()  # output is a list
        loss_t = torch.cat(tuple(loss_t), dim=1)  # N*(T-k)

        W = torch.exp(-500 * (loss_t @ M)).detach()
        loss_w = 100 * (W * loss_t).mean()

        loss = loss_t.mean()

        loss_w.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.7f, wloss: %.7f' % ((epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), loss.item(), loss_w.item()))
        train_loss_list.append(loss.item())
        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print('save model at {}'.format(epoch + 1))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_list': train_loss_list,
            }, './2Dadvection/data driven/model/checkpoint_deeponet.pt')
    return train_loss_list

def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./2Dadvection/data driven/model/checkpoint_deeponet.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)
    train_loss_list = checkpoint['train_loss_list']
    return model, optimizer, scheduler, train_loss_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def u_ana(x, y, t):
    u = 3*torch.exp(-A * ((x - R * np.cos(c * t)) ** 2 + (y - R * np.sin(c * t)) ** 2))
    return u

def Ada_X(u, kk, cc, N):

    Y = torch.abs(u)
    err_eq = torch.pow(Y, kk) / torch.pow(Y, kk).mean() + cc
    err_eq_normalized = (err_eq / err_eq.sum(dim=0))[:, 0]
    X_ids = torch.multinomial(err_eq_normalized, num_samples=N, replacement=False)
    return X_ids


if __name__ == '__main__':

    # 加载数据


    def generate_function_data(dt=0.02, grid_size=200, t_min=0.0, t_max=5.0):
        # 时间步长
        L=1.2
        t_steps = int(t_max / dt)+1
        t_vals = np.linspace(t_min, t_max, t_steps)  # 500个时间点
        # 空间网格
        x = np.linspace(-L/2, L/2, grid_size)  # 256个空间点，x方向
        y = np.linspace(-L/2, L/2, grid_size)  # 256个空间点，y方向
        X, Y = np.meshgrid(x, y)  # 生成二维空间网格

        # 创建一个存储结果的数组
        u_values = np.zeros((t_steps, grid_size, grid_size))

        A = 100
        c = 2*np.pi
        R = 0.25

        # 计算每个时间点的函数值
        for t_idx, t in enumerate(t_vals):
            # 计算每个网格点的函数值
            u = 3*np.exp(-A * ((X - R*np.cos(c*t)) ** 2 + (Y - R*np.sin(c*t)) ** 2))
            u_values[t_idx] = u  # 存储到数组中

        return u_values, t_vals


    # 生成数据
    u_sol, T = generate_function_data()

    u_sol = np.expand_dims(u_sol, axis=1)
    sub = 1
    u_sol = u_sol[::sub]
    T = T[::sub]

    L = 1.2  # 空间域大小
    Nx = 200  # 空间网格数
    Ny = 200  # 空间网格数
    x = np.linspace(-L/2, L/2, Nx)  # 200个空间点，x方向
    y = np.linspace(-L/2, L/2, Ny)  # 200个空间点，y方向
    X, Y = np.meshgrid(x, y)
    X_all = np.stack((X, Y), axis=-1).reshape(-1, 2)
    XX_all = np.stack((X[::2, ::2], Y[::2, ::2]), axis=-1).reshape(-1, 2)

    # solution parameters
    A = 100
    c = 2 * np.pi
    R = 0.25

    # 假设以下是你为模型提供的输入参数
    k = 5  # num of input u

    U0 = u_sol[:k, :, ::2, ::2]  # [k,2,128,128]
    width = U0.shape[-1]
    d = 2  # 维数

    length = 100
    step = k + length  # 时间步长
    effective_step = list(range(0, step))

    dt = T[1] - T[0]
    x_values = XX_all

    t = T[k-1]

    x_test = X_all
    x_test = torch.tensor(x_test, dtype=torch.float32).cuda()

    x = np.linspace(-L / 2, L / 2, 500)  # 200个空间点，x方向
    y = np.linspace(-L / 2, L / 2, 500)  # 200个空间点，y方向
    X, Y = np.meshgrid(x, y)
    X_all = np.stack((X, Y), axis=-1).reshape(-1, 2)
    x_sample = X_all
    x_sample = torch.tensor(x_sample, dtype=torch.float32).cuda()

    N = 8000
    indices = np.random.choice(len(tonp(x_sample[:, 0])), size=N, replace=False)
    y0 = x_sample[indices]

    ################# build the model #####################

    # 创建初始化的隐藏状态 (k * m)
    init_h0 = torch.tensor(U0, dtype=torch.float32).cuda()

    # 创建 x 和 y 输入数据
    x = torch.tensor(x_values, dtype=torch.float32).cuda()  # m * d 的输入


    output_dim = 100
    model = RCNN(
        k=k,
        trunk_layers=[2, 100, 100, output_dim],  # 输入为 d，输出为 100
        width=width,
        dt=dt,
        t=t,
        output_dim=output_dim,
        input_channels=1,
        input_kernel_size=5,
        init_h0=init_h0,
        x=x,
        y0=y0,
        x_sample=x_sample,
        step=step,
        effective_step=effective_step).cuda()

    n_iters = 1000  # 10000 for 200 steps, 5000 for 4000 steps, 5000 for 800 steps
    learning_rate = 1e-3
    # train the model
    start = time.time()
    cont = False  # if continue training (or use pretrained model), set cont=True
    train_loss_list = train(model, n_iters, learning_rate, cont=cont)
    end = time.time()

    print('The training time is: ', (end-start))

    plt.plot(train_loss_list)
    plt.yscale("log")
    plt.show()

    # Do the forward inference
    with torch.no_grad():
        output = model.test(x_test)
    output = torch.cat(tuple(output), dim=1).T
    U = u_sol[:step].reshape(step, 1, -1)  # [201, 1, 100, 100]
    truth = U[k:, 0, :]

    truth = truth.reshape(length, 200, 200)

    output = output.reshape(length, 200, 200).cpu().detach().numpy()

    Lu_e = np.mean(np.sqrt((output - truth) ** 2), axis=(1, 2))
    Lu = np.mean(np.sqrt(truth ** 2), axis=(1, 2))

    error = Lu_e / Lu
    plt.plot(error)
    plt.show()







