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


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
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

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
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
        # self.padding = 8  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

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

    def forward(self, h):
        # periodic padding, can also be achieved using 'circular' padding
        # h: k*N*2

        result = torch.einsum("mi,ihwj->mhwj", self.kweights, self.FNO2d(h))

        return result


class RONN(nn.Module):
    ''' Recurrent convolutional neural network layer '''

    def __init__(self, modes, width, k, dt, init_h0, truth,
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

        name = 'ronn_cell'
        cell = RONNCell(
            modes=self.modes,
            width=self.width,
            k=self.k,
            dt=self.dt)

        setattr(self, name, cell)
        self._all_layers.append(cell)

    def forward(self):

        internal_state = []
        loss_t = []
        mask_3d = torch.tensor(mask, dtype=torch.float32).cuda()
        mask_3d = mask_3d.float().unsqueeze(-1)

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

            loss = ((o* mask_3d.unsqueeze(0) - self.truth[step:step + 1]) ** 2).mean().unsqueeze(0).unsqueeze(0)

            internal_state = torch.cat((internal_state[1:], o * mask_3d.unsqueeze(0)), dim=0)

            # after many layers output the result save at time step t
            if step in self.effective_step:
                loss_t.append(loss)

        return loss_t

    def test(self):

        internal_state = []
        outputs_u = []
        mask_3d = torch.tensor(mask, dtype=torch.float32).cuda()
        mask_3d = mask_3d.float().unsqueeze(-1)

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
            o = o * mask_3d.unsqueeze(0)
            u = o[:, :, :, 0]

            internal_state = torch.cat((internal_state[1:], o * mask_3d.unsqueeze(0)), dim=0)

            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs_u.append(u)

        return outputs_u


def train(model, n_iters, learning_rate, k, cont=True):
    # define some parameters

    # truth
    # model
    if cont:
        model, optimizer, scheduler, train_loss_list = load_model(model)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = learning_rate
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.99)
        train_loss_list = []

    M = torch.triu(torch.ones(length, length), diagonal=1).cuda()

    for epoch in range(n_iters):
        # input: [time,. batch, channel, height, width]
        optimizer.zero_grad()
        # One single batch
        loss_t = model()  # output is a list
        loss_t = torch.cat(tuple(loss_t), dim=1)  # N*(T-k)

        W = torch.exp(-500 * (loss_t @ M)).detach()
        loss_w = 1000 * (W * loss_t).mean()

        loss = loss_t.mean()

        loss_w.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.7f, loss_w: %.7f' % (
        (epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), loss.item(), loss_w.item()))
        train_loss_list.append(loss.item())
        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print('save model at {}'.format(epoch+1))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_list': train_loss_list,
            }, './2Dheat/data driven/model/checkpoint_fno.pt')
    return train_loss_list


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./2Dheat/data driven/model/checkpoint_fno.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
    train_loss_list = checkpoint['train_loss_list']
    return model, optimizer, scheduler, train_loss_list

if __name__ == '__main__':

    ################# prepare the input dataset ####################

    # 定义五角星参数
    def create_star_mask(size=501, R=0.38, center=(0.5, 0.5)):
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

        return mask_star

    # 加载数据
    data = np.load('./data/2Dheat_star.npz')

    # 访问各个数组
    u_sol = data['u_sol']  # (6554,256,256)
    t = data['t']  # (6554,)
    u_sol = u_sol[1000:, :, :]

    sub = 10
    u_sol = u_sol[::sub, :, :]
    u_sol = np.expand_dims(u_sol, axis=-1)
    t = t[::sub]

    Lx, Ly = 1.0, 1.0  # 计算域总大小
    Nx, Ny = 256, 256  # 网格点数
    dx = Lx / Nx  # 空间步长
    dy = Ly / Ny

    # 创建坐标系
    X, Y = np.meshgrid(np.linspace(0, Lx, Nx, endpoint=False), np.linspace(0, Ly, Ny, endpoint=False))
    # 创建复合掩码
    mask = create_star_mask(size=Nx, R=0.38, center=(0.5, 0.5))


    # 假设以下是你为模型提供的输入参数
    k = 5  # num of input u
    d = 2  # 维数

    length = 115
    step = k+length  # 时间步长
    effective_step = list(range(0, step))

    # 定义网络层数（对于例子，我们使用简单的层数）
    modes = 16  # 输入为 m，输出为 100
    width = 64  # 输入为 d，输出为 100

    dt = t[1]-t[0]
    U0 = u_sol[:k, :, :]
    U = u_sol[:step, :, :]

    init_h0 = torch.tensor(U0, dtype=torch.float32).cuda()  # (k,128,128,1)

    target = torch.tensor(U, dtype=torch.float32).cuda()
    truth = target[k:]


    # 创建 RONN 模型
    model = RONN(modes, width, k, dt, init_h0, truth, step, effective_step).cuda()

    n_iters = 0
    learning_rate = 1e-3
    # 假设目标输出 (target) 是一个与 outputs 形状匹配的张量
    target = torch.tensor(U, dtype=torch.float32).cuda()


    # train the model
    start = time.time()
    cont = True  # if continue training (or use pretrained model), set cont=True
    train_loss_list = train(model, n_iters, learning_rate, k, cont=cont)
    end = time.time()

    print('The training time is: ', (end - start))


    plt.plot(train_loss_list)
    plt.yscale("log")
    plt.show()

    with torch.no_grad():
        outputs_u = model.test()

    outputs_u = torch.cat(tuple(outputs_u), dim=0)  # N*(T-k)
    outputs_u = outputs_u.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()
    truth_u = truth[:, :, :, 0]

    Lu_e = np.mean(np.sqrt((outputs_u - truth_u) ** 2), axis=(1, 2))
    Lu = np.mean(np.sqrt(truth_u ** 2), axis=(1, 2))

    t = t[k:k + length]
    erroru = Lu_e / Lu
    plt.plot(t, erroru)
    plt.show()




