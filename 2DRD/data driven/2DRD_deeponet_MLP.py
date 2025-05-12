import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float32)

torch.manual_seed(66)
np.random.seed(66)

class ModifiedMLP_p(nn.Module):
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


class RONNCell(nn.Module):
    ''' Recurrent convolutional neural network Cell '''

    def __init__(self, branch_layers, trunk_layers, k, p, dt):
        super(RONNCell, self).__init__()

        # the initial parameters
        self.branch_net = ModifiedMLP(branch_layers, SinAct())  # [m,100,100,p]
        self.trunk_net = ModifiedMLP_p(trunk_layers, SinAct())  # [d,100,100,p]

        self.k = k
        self.p = p
        self.dt = dt
        self.kweights = nn.Parameter(torch.Tensor(k, 1))
        self.W = nn.Parameter(torch.Tensor(m, p))
        # 初始化权重
        self._init_weights()
        self.softplus = nn.Softplus()

    def _init_weights(self):
        # 使用 Xavier 均匀分布初始化
        init.xavier_uniform_(self.kweights)
        init.xavier_uniform_(self.W)

    def forward(self, h, y):
        # periodic padding, can also be achieved using 'circular' padding
        # h: N*k*m   x: N*d

        B = self.dt*self.branch_net(h) + h @ self.W  # k*p
        B = torch.einsum('ijk,i->jk', B.reshape(k, 2, -1), self.kweights.squeeze(-1))
        T = self.trunk_net(y)  # N*p
        outputs = 1/self.p*torch.matmul(T, B.T)
        outputs = torch.sin(outputs)

        return outputs



class RONN(nn.Module):
    ''' Recurrent convolutional neural network layer '''

    def __init__(self, branch_layers, trunk_layers, k, p, dt, init_h0, truth, x,
                 step=1, effective_step=[1]):

        super(RONN, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.k = k  # no use, always 1
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        # self.init_state = torch.nn.Parameter(torch.tensor(ini_state, dtype=torch.float32).cuda(), requires_grad=False)
        self.init_h0 = init_h0  # k*m
        self.truth = truth
        self.x = x  # m sensor points m*d
        self.init_state = []
        self.dt = dt

        name = 'ronn_cell'
        cell = RONNCell(
            branch_layers=self.branch_layers,
            trunk_layers=self.trunk_layers,
            k=self.k,
            p=p,
            dt=self.dt)

        setattr(self, name, cell)
        self._all_layers.append(cell)

    def forward(self, y):

        # m = self.x.shape[0]
        # N = self.y.shape[0]
        internal_state = []
        loss_t = []

        for step in range(self.step - self.k):
            name = 'ronn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_h0  # k*m
                # h = h0.unsqueeze(0).repeat(N, 1, 1) # N*k*m
                internal_state = h

            # forward
            h = internal_state  # k*m
            # hidden state + output
            o = getattr(self, name)(h, y)  # N*1
            loss = ((o - self.truth[step].T) ** 2).mean().unsqueeze(0).unsqueeze(0)
            # ------------------------------------------------------------

            h1 = getattr(self, name)(h, self.x)  # m*1
            internal_state = self.cat(internal_state, h1)


            # after many layers output the result save at time step t
            if step in self.effective_step:
                loss_t.append(loss)

        return loss_t

    def cat(self, internal_state, h1):

        internal_state = internal_state.reshape(k, 2, -1)
        internal_state = torch.cat((internal_state[1:], h1.T.unsqueeze(0)), dim=0)

        return internal_state

    def test(self, y):

        internal_state = []
        outputs_u = []
        outputs_v = []

        for step in range(self.step - self.k):
            name = 'ronn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_h0  # k*m
                # h = h0.unsqueeze(0).repeat(N, 1, 1) # N*k*m
                internal_state = h

            # forward
            h = internal_state  # k*m
            # hidden state + output
            o = getattr(self, name)(h, y)  # N*1

            # ------------------------------------------------------------

            h1 = getattr(self, name)(h, self.x)  # m*1
            internal_state = self.cat(internal_state, h1)


            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs_u.append(o[:, 0:1])
                outputs_v.append(o[:, 1:2])

        return outputs_u, outputs_v



def train(model, y, n_iters, learning_rate, k, cont=True):
    # define some parameters

    # truth
    # model
    if cont:
        model, optimizer, scheduler, train_loss_list = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=500, gamma=0.99)
        train_loss_list = []


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
            }, './2DRD/data driven/model/checkpoint_mlp.pt')
    return train_loss_list


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./2DRD/data driven/model/checkpoint_mlp.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)
    train_loss_list = checkpoint['train_loss_list']
    return model, optimizer, scheduler, train_loss_list

if __name__ == '__main__':

    ################# prepare the input dataset ####################

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

    N = 10000
    distances = np.sqrt((X.flatten() - 5) ** 2 + (Y.flatten() - 5) ** 2)
    weights = 1 / (distances + 1e-6)
    indices = np.random.choice(len(X.flatten()), size=N, p=weights / weights.sum(), replace=False)
    # selected_indices = np.random.choice(len(X_flat), size=N, replace=False)
    x = X_all[indices]

    # 假设以下是你为模型提供的输入参数
    k = 5  # num of input u

    U0 = uv_sol[:k, :, ::2, ::2].reshape(k*2, -1)  # [k,2,128*128]
    m = U0.shape[-1]  # same points
    d = 2  # 维数

    length = 300
    step = k+length  # 时间步长
    effective_step = list(range(0, step))
    p = 100
    # 定义网络层数（对于例子，我们使用简单的层数）

    branch_layers = [m, 100, 100, p]  # 输入为 m，输出为 100
    trunk_layers = [2*(Mx + My), 100, 100, p]  # 输入为 d，输出为 100

    dt = t[1]-t[0]
    x_values = XX_all
    y_values = x

    U = uv_sol[:step].reshape(step, 2, -1)  # [step,128*128]
    truth = U[k:, :, indices]
    truth = torch.tensor(truth, dtype=torch.float32).cuda()

    # 创建初始化的隐藏状态 (k * m)
    init_h0 = torch.tensor(U0, dtype=torch.float32).cuda()

    # 创建 x 和 y 输入数据
    x = torch.tensor(x_values, dtype=torch.float32).cuda()  # m * d 的输入
    y = torch.tensor(y_values, dtype=torch.float32).cuda()  # N * d 的输入

    # 创建 RONN 模型
    model = RONN(branch_layers, trunk_layers, k, p, dt, init_h0, truth, x, step, effective_step).cuda()

    n_iters = 0
    learning_rate = 1e-3

    # train the model
    start = time.time()
    cont = True  # if continue training (or use pretrained model), set cont=True
    train_loss_list = train(model, y, n_iters, learning_rate, k, cont=cont)
    end = time.time()

    print('The training time is: ', (end - start))

    plt.plot(train_loss_list)
    plt.yscale("log")
    plt.show()

    x = torch.tensor(X_all, dtype=torch.float32).cuda()

    with torch.no_grad():
        output_u, output_v = model.test(x)

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
    # plt.plot(erroru)
    # plt.plot(errorv)
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
