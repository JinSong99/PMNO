import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.ndimage import zoom

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float32)

torch.manual_seed(66)
np.random.seed(66)


class MLP(nn.Module):
    def __init__(self, layers, activation):
        super(MLP, self).__init__()
        self.activation = activation

        # 创建中间线性层
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            in_dim = layers[i]
            out_dim = layers[i + 1]
            self.linears.append(nn.Linear(in_dim, out_dim))

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化线性层参数
        for linear in self.linears:
            init.xavier_normal_(linear.weight)
            init.zeros_(linear.bias)

    def forward(self, x):
        # 处理每一层（除了最后一层）
        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))  # 应用激活函数

        # 处理最后一层（无激活函数）
        x = self.linears[-1](x)
        return x


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
        self.branch_net = ModifiedMLP(branch_layers, nn.Tanh())  # [m,100,100,p]
        self.trunk_net = ModifiedMLP(trunk_layers, nn.Tanh())  # [d,100,100,p]

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
        # B = self.branch_net(h)  # N*k*p
        B = torch.sum(B * self.kweights, dim=0)  # (p,)
        B = B.unsqueeze(1)  # (p,1)
        T = self.trunk_net(y)  # N*p
        outputs = 1/self.p*torch.matmul(T, B)
        return outputs



class RONN(nn.Module):
    ''' Recurrent convolutional neural network layer '''

    def __init__(self, branch_layers, trunk_layers, k, p, dt, init_h0, init_true, truth, x,
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
        self.init_true = init_true
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


        internal_state = []
        loss_t = []
        rloss_t = []  # 计算残差



        if not y.requires_grad:
            y.requires_grad_(True)


        for step in range(self.step - self.k):
            name = 'ronn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_h0  # k*m
                # h = h0.unsqueeze(0).repeat(N, 1, 1) # N*k*m
                internal_state = h
                internal_output = self.init_true

            # forward
            h = internal_state  # k*m
            # hidden state + output
            o = getattr(self, name)(h, y)  # N*1

            # 计算梯度 ----------------------------------------------------
            if y.grad is not None:
                y.grad.zero_()  # 清空之前的梯度

            Lu = self.compute_laplacian(o, y)
            f = 0.001 * Lu

            res = 137 / 60 * o - 5 * internal_output[:, 4:5] + 5 * internal_output[:, 3:4] \
                   - 10 / 3 * internal_output[:, 2:3] + 5 / 4 * internal_output[:, 1:2] \
                   - 1 / 5 * internal_output[:, 0:1] - self.dt * f
            rloss = (res ** 2).mean().unsqueeze(0).unsqueeze(0)
            loss = ((o - self.truth[:, step:step + 1]) ** 2).mean().unsqueeze(0).unsqueeze(0)
            # ------------------------------------------------------------

            h1 = getattr(self, name)(h, self.x)  # m*1
            internal_state = torch.cat((internal_state[1:], h1.T), dim=0)


            # after many layers output the result save at time step t
            if step in self.effective_step:
                loss_t.append(loss)
                rloss_t.append(rloss)

        return loss_t, rloss_t

    def compute_laplacian(self, u, y):
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

        # m = self.x.shape[0]
        # N = self.y.shape[0]
        internal_state = []
        outputs = []

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
            internal_state = torch.cat((internal_state[1:], h1.T), dim=0)


            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs.append(o)

        return outputs

def train(model, y, n_iters, learning_rate, k, cont=True):
    # define some parameters

    # truth
    # model
    if cont:
        model, optimizer, scheduler, train_loss_list, train_rloss_list = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.99)
        train_loss_list = []
        train_rloss_list = []

    M = torch.triu(torch.ones(truth.shape[1], truth.shape[1]), diagonal=1).cuda()

    for epoch in range(n_iters):
        # input: [time,. batch, channel, height, width]
        optimizer.zero_grad()
        # One single batch
        loss_t, rloss_t = model(y)  # output is a list
        loss_t = torch.cat(tuple(loss_t), dim=1)  # N*(T-k)
        rloss_t = torch.cat(tuple(rloss_t), dim=1)
        loss = loss_t.mean()

        rW = torch.exp(-500 * (rloss_t @ M)).detach()
        rloss_w = 1000 * (rW * rloss_t).mean()
        rloss = rloss_t.mean()

        rloss_w.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.7f, rloss_w: %.7f, loss_w: %.7f' % (
            (epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), loss.item(), rloss_w.item(), rloss.item()))
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
            }, './2Dheat/physics driven/model/checkpoint_pideeponet.pt')
    return train_loss_list


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./2Dheat/physics driven/model/checkpoint_pideeponet.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)
    train_loss_list = checkpoint['train_loss_list']
    train_rloss_list = checkpoint['train_rloss_list']
    return model, optimizer, scheduler, train_loss_list, train_rloss_list

if __name__ == '__main__':

    ################# prepare the input dataset ####################

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

    #-------------sample-----------------------
    N = 5000
    X_flat = X[mask].flatten()
    Y_flat = Y[mask].flatten()

    distances = np.sqrt((X_flat - 0.5) ** 2 + (Y_flat - 0.5) ** 2)
    weights = 1 / (distances + 1e-6)  # 加小常数避免除零
    selected_indices = np.random.choice(len(X_flat), size=N, p=weights/weights.sum(), replace=False)
    # selected_indices = np.random.choice(len(X_flat), size=N, replace=False)
    x = np.vstack((X_flat[selected_indices], Y_flat[selected_indices])).T

    # 假设以下是你为模型提供的输入参数
    k = 5  # num of input u

    U0 = u_sol[:k, mask].reshape(k, -1)  # [k,256*256]
    U0 = U0[:, selected_indices]
    m = U0.shape[1]  # same points
    d = 2  # 维数

    length = 200
    step = k+length  # 时间步长
    effective_step = list(range(0, step))
    p = 10
    # 定义网络层数（对于例子，我们使用简单的层数）

    branch_layers = [m, 60, 60, 60, p]  # 输入为 m，输出为 100
    trunk_layers = [2, 60, 60, 60, p]  # 输入为 d，输出为 100

    dt = t[1]-t[0]
    x_values = x
    y_values = x

    U = u_sol[:step, mask].reshape(step, -1)  # [step,128*128]
    U = U[:, selected_indices].T

    target = torch.tensor(U, dtype=torch.float32).cuda()
    truth = target[:, k:]

    init_true = target[:, :k]

    # 创建初始化的隐藏状态 (k * m)
    init_h0 = torch.tensor(U0, dtype=torch.float32).cuda()

    # 创建 x 和 y 输入数据
    x = torch.tensor(x_values, dtype=torch.float32).cuda()  # m * d 的输入
    y = torch.tensor(y_values, dtype=torch.float32).cuda()  # N * d 的输入

    # 创建 RONN 模型
    model = RONN(branch_layers, trunk_layers, k, p, dt, init_h0, init_true, truth, x, step, effective_step).cuda()


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


    X_all = torch.tensor(X_all, dtype=torch.float32).cuda()

    with torch.no_grad():
        outputs = model.test(X_all)
    outputs = torch.cat(tuple(outputs), dim=1)  # N*(T-k)
    outputs = outputs.cpu().detach().numpy()
    outputs = outputs.reshape(256, 256, -1)

    mask_3d = mask[:, :, np.newaxis]
    outputs = outputs * mask_3d

    U = u_sol[:step].transpose(1, 2, 0)  # [128,128,step]
    truth = U[:, :, k:]

    L0_e = np.mean(np.sqrt((outputs - truth)**2), axis=(0, 1))
    L0 = np.mean(np.sqrt((truth)**2), axis=(0, 1))

    t = t[k:k + length]
    error = L0_e / L0
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
    outputs = torch.cat(tuple(outputs), dim=1)  # N*(T-k)
    outputs = outputs.cpu().detach().numpy()
    outputs = outputs.reshape(Nx, Ny, -1)

    outputs_mask = np.where(mask, outputs[:, :, -k], np.nan)

    plt.figure(figsize=(8, 6))
    plt.imshow(outputs_mask, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()

