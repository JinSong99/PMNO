import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch.nn.init as init
from scipy.ndimage import zoom

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
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
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
            nn.Linear(width ** 2, output_dim),
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

        return outputs


class RCNN(nn.Module):
    ''' Recurrent convolutional neural network layer '''

    def __init__(self, k, trunk_layers, width, dt, output_dim, input_channels, input_kernel_size, init_h0, truth, x,
                 step, mask, effective_step=[1]):

        super(RCNN, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.init_h0 = init_h0
        self.truth = truth
        self.k = k
        self.x = x
        self.dt = dt
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.mask = mask

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
        loss_t = []
        mask_3d = self.mask[::2, ::2]
        mask_3d = torch.tensor(mask_3d, dtype=torch.float32).cuda()
        mask_3d = mask_3d.float().unsqueeze(0).unsqueeze(0)

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

            loss = ((o - self.truth[step].T) ** 2).mean().unsqueeze(0).unsqueeze(0)

            h1 = getattr(self, name)(h, self.x)  # m*1
            internal_state = torch.cat((internal_state[1:], self.transform(h1) * mask_3d), dim=0)

            # after many layers output the result save at time step t
            if step in self.effective_step:
                loss_t.append(loss)

        return loss_t

    def transform(self, h):

        reshaped = h.view(128, 128)  # 第一个通道 (128, 128)
        final_tensor = reshaped.unsqueeze(0).unsqueeze(0)  # 或 stacked_tensor[None, ...]

        return final_tensor

    def test(self, y):

        mask_3d = self.mask[::2, ::2]
        mask_3d = torch.tensor(mask_3d, dtype=torch.float32).cuda()
        mask_3d = mask_3d.float().unsqueeze(0).unsqueeze(0)

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
            internal_state = torch.cat((internal_state[1:], self.transform(h1) * mask_3d), dim=0)

            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs.append(o)

        return outputs


def train(model, y, n_iters, learning_rate, cont=True):
    if cont:
        model, optimizer, scheduler, train_loss_list = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.99)
        train_loss_list = []

    # for param_group in optimizer.param_groups:
    #    param_group['lr']=0.0012

    M = torch.triu(torch.ones(length, length), diagonal=1).cuda()

    for epoch in range(n_iters):
        # input: [time,. batch, channel, height, width]
        optimizer.zero_grad()
        # One single batch
        loss_t = model(y)  # output is a list
        loss_t = torch.cat(tuple(loss_t), dim=1)  # N*(T-k)

        W = torch.exp(-500 * (loss_t @ M)).detach()
        loss_w = 1000 * (W * loss_t).mean()

        loss = loss_t.mean()

        loss_w.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.7f, wloss: %.7f' % (
        (epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), loss.item(), loss_w.item()))
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
            }, './2Dheat/data driven/model/checkpoint_deeponet_cnn.pt')
    return train_loss_list


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./2Dheat/data driven/model/checkpoint_deeponet_cnn.pt')
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

    # 定义五角星参数
    def create_star_mask(size=501, R=0.38, r=0.1, center=(0.5, 0.5)):
        """创建五角星形状"""
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)

        # 转换到极坐标系
        theta = np.arctan2(Y - center[1], X - center[0])
        rho = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        # 五角星方程（极坐标形式）
        star_radius = R * (1 + 0.5 * np.sin(5 * theta)) / (1 + 0.2 * np.abs(np.sin(2.5 * theta)))
        mask_star = rho <= star_radius

        # 中心圆形扣除

        return mask_star


    # 加载数据
    data = np.load('./data/2Dheat_star.npz')

    # 访问各个数组
    u_sol = data['u_sol']  # (8193,128,128)
    t = data['t']  # (8193,)
    u_sol = u_sol[1000:, :, :]

    sub = 10
    u_sol = u_sol[::sub, :]
    t = t[::sub]
    u_sol = np.expand_dims(u_sol, axis=1)

    # 定义网格参数
    Lx, Ly = 1.0, 1.0  # 计算域总大小
    Nx, Ny = 256, 256  # 增加网格分辨率
    dx = Lx / Nx
    dy = Ly / Ny

    # 创建坐标系
    X, Y = np.meshgrid(np.linspace(0, Lx, Nx, endpoint=False), np.linspace(0, Ly, Ny, endpoint=False))
    # 创建复合掩码
    mask = create_star_mask(size=Nx, R=0.38, r=0.1, center=(0.5, 0.5))
    X_all = np.stack((X, Y), axis=-1).reshape(-1, 2)
    XX_all = np.stack((X[::2, ::2], Y[::2, ::2]), axis=-1).reshape(-1, 2)

    # 假设以下是你为模型提供的输入参数
    k = 5  # num of input u

    U0 = u_sol[:k, :, ::2, ::2]  # [k,2,128,128]
    width = U0.shape[-1]
    d = 2  # 维数

    length = 115
    step = k + length  # 时间步长
    effective_step = list(range(0, step))

    dt = t[1] - t[0]
    x_values = XX_all

    # -------------------------
    N = 5000
    X_flat = X[mask].flatten()
    Y_flat = Y[mask].flatten()

    distances = np.sqrt((X_flat - 0.5) ** 2 + (Y_flat - 0.5) ** 2)
    weights = 1 / (distances + 1e-6)
    selected_indices = np.random.choice(len(X_flat), size=N, p=weights / weights.sum(), replace=False)
    # selected_indices = np.random.choice(len(X_flat), size=N, replace=False)
    y = np.vstack((X_flat[selected_indices], Y_flat[selected_indices])).T

    ################# build the model #####################

    # 创建初始化的隐藏状态 (k * m)
    init_h0 = torch.tensor(U0, dtype=torch.float32).cuda()

    # 创建 x 和 y 输入数据
    x = torch.tensor(x_values, dtype=torch.float32).cuda()  # m * d 的输入
    y = torch.tensor(y, dtype=torch.float32).cuda()  # N * d 的输入

    U = u_sol[:step]
    U = U[:, :, mask]  # [201, 2, 32*32*32]
    truth = U[k:, :, selected_indices]
    truth = torch.tensor(truth, dtype=torch.float32).cuda()

    output_dim = 100
    model = RCNN(
        k=k,
        trunk_layers=[2, 100, 100, 100, output_dim],  # 输入为 d，输出为 100
        width=width,
        dt=dt,
        output_dim=output_dim,
        input_channels=1,
        input_kernel_size=5,
        init_h0=init_h0,
        truth=truth,
        x=x,
        step=step,
        mask=mask,
        effective_step=effective_step).cuda()

    n_iters = 0
    learning_rate = 1e-3
    # train the model
    start = time.time()
    cont = True  # if continue training (or use pretrained model), set cont=True
    train_loss_list = train(model, y, n_iters, learning_rate, cont=cont)
    end = time.time()

    print('The training time is: ', (end - start))

    plt.plot(train_loss_list)
    plt.yscale("log")
    plt.show()

    # Do the forward inference
    X_all = torch.tensor(X_all, dtype=torch.float32).cuda()
    with torch.no_grad():
        output = model.test(X_all)
    output = torch.cat(tuple(output), dim=1).T
    U = u_sol[:step].reshape(step, 1, -1)  # [201, 1, 100, 100]
    truth = U[k:, 0, :]

    truth = truth.reshape(length, 256, 256)

    output = output.reshape(length, 256, 256).cpu().detach().numpy()

    Lu_e = np.mean(np.sqrt((output - truth) ** 2), axis=(1, 2))
    Lu = np.mean(np.sqrt(truth ** 2), axis=(1, 2))

    t = t[k:k + length]
    error = Lu_e / Lu
    plt.plot(t, error)
    plt.show()




    #########################PLOT#############################

    # 定义网格参数
    Lx, Ly = 1.0, 1.0  # 计算域总大小
    Nx, Ny = 1024, 1024  # 增加网格分辨率

    # 创建坐标系
    X, Y = np.meshgrid(np.linspace(0, Lx, Nx, endpoint=False), np.linspace(0, Ly, Ny, endpoint=False))
    # 创建复合掩码
    mask = create_star_mask(size=Nx, R=0.38, r=0.1, center=(0.5, 0.5))

    X_all = np.stack((X, Y), axis=-1).reshape(-1, 2)

    X_all = torch.tensor(X_all, dtype=torch.float32).cuda()

    with torch.no_grad():
        outputs = model.test(X_all)
    outputs = torch.cat(tuple(outputs), dim=1).T  # N*(T-k)
    outputs = outputs.cpu().detach().numpy()
    outputs = outputs.reshape(length, Nx, Nx)

    outputs_mask = np.where(mask, outputs[-k, :, :], np.nan)

    plt.figure(figsize=(8, 6))
    plt.imshow(outputs_mask, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('$x$', fontsize=14, fontname='Times New Roman')
    plt.ylabel('$y$', fontsize=14, fontname='Times New Roman')
    plt.show()

