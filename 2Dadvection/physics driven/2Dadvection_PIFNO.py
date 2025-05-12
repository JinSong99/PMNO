import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os

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




class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x





class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), b(x,y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, h, x):
        x = torch.cat((h, x), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # x = F.pad(x, [0, self.padding, 0, self.padding])

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

        # x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x





class RONNCell(nn.Module):
    ''' Recurrent convolutional neural network Cell '''

    def __init__(self, modes, width, k, dt):
        super(RONNCell, self).__init__()

        # the initial parameters
        self.FNO2d = FNO2d(modes, modes, width)


        self.k = k
        self.dt = dt
        self.kweights = nn.Parameter(torch.Tensor(1, k))
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 使用 Xavier 均匀分布初始化
        init.xavier_uniform_(self.kweights)

    def forward(self, h, x):
        # periodic padding, can also be achieved using 'circular' padding
        # h: k*N*2

        result = torch.einsum("mi,ihwj->mhwj", self.kweights, self.FNO2d(h, x))

        return result


class RONN(nn.Module):
    ''' Recurrent convolutional neural network layer '''

    def __init__(self, modes, width, k, t, dt, init_h0, truth,
                 step=1, effective_step=[1]):

        super(RONN, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.modes = modes
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
        self.t = t

        name = 'ronn_cell'
        cell = RONNCell(
            modes=self.modes,
            width=self.width,
            k=self.k,
            dt=self.dt)

        setattr(self, name, cell)
        self._all_layers.append(cell)

    def forward(self, x):

        internal_state = []
        loss_t = []
        rloss_t = []  # 计算残差
        t = self.t

        for step in range(self.step - self.k):
            name = 'ronn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_h0  # k*N
                internal_state = h

            # forward torch.cat((x, grid), dim=-1)
            h = internal_state  # k*N
            # hidden state + output
            o = getattr(self, name)(h, x.repeat(k, 1, 1, 1))  # N*1
            u = o[0, :, :, 0:1]  # (128, 128)
            t += self.dt

            ux, uy = self.spectral_derivatives(u.squeeze(-1))
            ux = ux.unsqueeze(-1)
            uy = uy.unsqueeze(-1)

            f = c * R * np.sin(c * t) * ux - c * R * np.cos(c * t) * uy

            res = 137 / 60 * u - 5 * internal_state[4, :, :, 0:1] + 5 * internal_state[3, :, :, 0:1] \
                   - 10 / 3 * internal_state[2, :, :, 0:1] + 5 / 4 * internal_state[1, :, :, 0:1] \
                   - 1 / 5 * internal_state[0, :, :, 0:1] - self.dt * f


            rloss = (res ** 2).mean().unsqueeze(0).unsqueeze(0)
            loss = ((o - self.truth[step:step+1]) ** 2).mean().unsqueeze(0).unsqueeze(0)
            internal_state = torch.cat((internal_state[1:], o), dim=0)

            if step in self.effective_step:
                loss_t.append(loss)
                rloss_t.append(rloss)

        return rloss_t, loss_t

    def spectral_derivatives(self, u, Lx=1.2, Ly=1.2):
        """
        使用谱方法计算二维张量场 u 的空间导数
        :param u: 输入张量 (Nx, Ny)
        :param Lx: x方向计算域长度
        :param Ly: y方向计算域长度
        :return: (du_x, du_y) 两个导数场，形状均为 (Nx, Ny)
        """
        # 获取网格尺寸
        Nx, Ny = u.shape

        # 生成波数分量 (遵循FFT频率排序规则)
        kx = 2 * np.pi * torch.fft.fftfreq(Nx, d=Lx / Nx).to(u.device)  # (Nx,)
        ky = 2 * np.pi * torch.fft.fftfreq(Ny, d=Ly / Ny).to(u.device)  # (Ny,)

        # 扩展为二维波数网格
        Ky, Kx = torch.meshgrid(kx, ky)  # (Nx, Ny)

        # 执行二维傅里叶变换 (自动处理为复数张量)
        U = torch.fft.fft2(u)  # (Nx, Ny) complex

        # 计算频域导数
        dU_dx = 1j * Kx * U
        dU_dy = 1j * Ky * U

        # 逆变换回空间域
        du_dx = torch.fft.ifft2(dU_dx).real  # (Nx, Ny)
        du_dy = torch.fft.ifft2(dU_dy).real  # (Nx, Ny)

        return du_dx, du_dy

    def test(self, x):

        internal_state = []
        outputs_u = []

        for step in range(self.step - self.k):
            name = 'ronn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_h0  # k*N
                internal_state = h

            # forward torch.cat((x, grid), dim=-1)
            h = internal_state  # k*N
            # hidden state + output
            o = getattr(self, name)(h, x.repeat(k, 1, 1, 1))  # N*1
            u = o[:, :, :, 0]

            internal_state = torch.cat((internal_state[1:], o), dim=0)

            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs_u.append(u)

        return outputs_u


def train(model, x, n_iters, learning_rate, cont=True):
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
        # One single batch
        rloss_t, loss_t = model(x)  # output is a list
        loss_t = torch.cat(tuple(loss_t), dim=1)  # N*(T-k)
        rloss_t = torch.cat(tuple(rloss_t), dim=1)

        loss = loss_t.mean()

        rW = torch.exp(-500 * (rloss_t @ M)).detach()
        rloss_w = 500 * (rW * rloss_t).mean()
        rloss = (rloss_t).mean()

        rloss_w.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.7f, rloss: %.7f, rloss_w: %.7f' % (
        (epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), loss.item(), rloss.item(), rloss_w.item()))
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
            }, './2Dadvection/physics driven/model/checkpoint_pifno.pt')
    return train_loss_list


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./2Dadvection/physics driven/model/checkpoint_pifno.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
    train_loss_list = checkpoint['train_loss_list']
    train_rloss_list = checkpoint['train_rloss_list']
    return model, optimizer, scheduler, train_loss_list, train_rloss_list

if __name__ == '__main__':

    ################# prepare the input dataset ####################

    # 加载数据
    def generate_function_data(dt=0.02, grid_size=128, t_min=0.0, t_max=5.0):
        # 时间步长
        L = 1.2
        t_steps = int(t_max / dt) + 1
        t_vals = np.linspace(t_min, t_max, t_steps)  # 500个时间点
        # 空间网格
        x = np.linspace(-L / 2, L / 2, grid_size, endpoint=False)  # 256个空间点，x方向
        y = np.linspace(-L / 2, L / 2, grid_size, endpoint=False)  # 256个空间点，y方向
        X, Y = np.meshgrid(x, y)  # 生成二维空间网格

        # 创建一个存储结果的数组
        u_values = np.zeros((t_steps, grid_size, grid_size))

        A = 100
        c = 2 * np.pi
        R = 0.25

        # 计算每个时间点的函数值
        for t_idx, t in enumerate(t_vals):
            # 计算每个网格点的函数值
            u = 3 * np.exp(-A * ((X - R * np.cos(c * t)) ** 2 + (Y - R * np.sin(c * t)) ** 2))
            u_values[t_idx] = u  # 存储到数组中

        return u_values, t_vals

    # 访问各个数组
    u_sol, T = generate_function_data()

    sub = 1
    u_sol = u_sol[::sub, :, :]
    T = T[::sub]

    Lx, Ly = 1.2, 1.2  # 计算域总大小
    Nx, Ny = 128, 128  # 网格点数
    dx = Lx / Nx  # 空间步长
    dy = Ly / Ny

    # 创建坐标系
    X, Y = np.meshgrid(np.linspace(-Lx/2, Lx/2, Nx, endpoint=False), np.linspace(-Ly/2, Ly/2, Ny, endpoint=False))
    grid = np.stack([X, Y], axis=-1)

    # solution parameters
    A = 100
    c = 2 * np.pi
    R = 0.25

    # 假设以下是你为模型提供的输入参数
    k = 5  # num of input u
    d = 2  # 维数

    length = 150
    step = k+length  # 时间步长
    effective_step = list(range(0, step))


    modes = 8  # 输入为 m，输出为 100
    width = 32  # 输入为 d，输出为 100

    dt = T[1]-T[0]
    t = T[k - 1]
    U0 = u_sol[:k, :, :]
    U = u_sol[:step, :, :]

    grid = torch.tensor(grid, dtype=torch.float32, requires_grad=True).cuda()
    init_h0 = torch.tensor(U0, dtype=torch.float32).cuda()  # (k,128,128)
    init_h0 = init_h0.unsqueeze(-1)  # (k,128,128,1)

    target = torch.tensor(U, dtype=torch.float32).cuda()
    truth = target[k:]
    truth = truth.unsqueeze(-1)

    # 创建 RONN 模型
    model = RONN(modes, width, k, t, dt, init_h0, truth, step, effective_step).cuda()

    n_iters = 0
    learning_rate = 1e-3

    # train the model
    start = time.time()
    cont = True  # if continue training (or use pretrained model), set cont=True
    train_loss_list = train(model, grid, n_iters, learning_rate, cont=cont)
    end = time.time()

    print('The training time is: ', (end - start))


    plt.plot(train_loss_list)
    plt.yscale("log")
    plt.show()

    with torch.no_grad():
        outputs_u = model.test(grid)

    outputs_u = torch.cat(tuple(outputs_u), dim=0)  # N*(T-k)
    outputs_u = outputs_u.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()
    truth_u = truth[:, :, :, 0]

    Lu_e = np.mean(np.sqrt((outputs_u - truth_u) ** 2), axis=(1, 2))
    Lu = np.mean(np.sqrt(truth_u ** 2), axis=(1, 2))

    erroru = Lu_e / Lu
    t = T[k:k + length]
    plt.plot(t, erroru)
    plt.show()

    indices = np.where(erroru > 0.1)[0]

    if len(indices) > 0:
        first_index = indices[0]
        first_value = erroru[first_index]
        print(f"第一个大于0.1的位置是索引 {first_index+k}，对应值 {first_value}")
        print(f"第一个大于0.1的时间 {T[first_index+k]}")
    else:
        print("没有找到大于0.1的元素")


    plt.figure(figsize=(8, 6))
    plt.imshow(outputs_u[-5], aspect='auto', origin='lower', cmap='viridis', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    plt.colorbar()
    # plt.title('Heatmap of u = exp(-(x - at)^2)')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(truth_u[-5], aspect='auto', origin='lower',
               cmap='viridis', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    plt.colorbar()
    # plt.title('Heatmap of u = exp(-(x - at)^2)')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow((outputs_u[-5] - truth_u[-5]), aspect='auto', origin='lower',
               cmap='viridis', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
    plt.colorbar()
    # plt.title('Heatmap of u = exp(-(x - at)^2)')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.show()




