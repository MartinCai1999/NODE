from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
# from scipy.integrate import odeint
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from scipy.optimize import curve_fit
np.random.seed(1999)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
def setupSeed(n):
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True
setupSeed(-999)

po = pd.read_csv("HK_Covid.csv")
po = po.iloc[40:85, :]
l = len(po)
df = po
df.dropna(inplace=True)
# df.columns = ["Month", "MOI"]
# df.set_index('Month', inplace=True)
# print(df)
x = po['Day']
# l = 36
t = np.linspace(0, l - 1, l)

df = df.iloc[:, [16]]
print(df)
t1 = range(l)
# timeseries = df.values.astype('float32')
timeseries = np.array(list(zip(df['Omicron(BA.2)'], t1)))
timeseries = timeseries[~np.isnan(timeseries).any(axis=1)]
timeseries = timeseries.astype('float32')
train_size = int(len(timeseries) * 0.8)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]
t_t = torch.tensor(t.astype('float32')).to(device)

def create_dataset(X, y):
    """Transform a time series into a prediction dataset
    """
    # X, y = [], []
    return torch.tensor(X).view(-1, 1).requires_grad_(True).to(device), torch.tensor(y).view(-1, 1).requires_grad_(True).to(device)

X_train, y_train = create_dataset(train[:, 1], train[:, 0])
X_test, y_test = create_dataset(test[:, 1], test[:, 0])
X, y = create_dataset(timeseries[:, 1], timeseries[:, 0])

t_train = torch.tensor(np.linspace(0, train_size - 1, train_size).astype('float32')).to(device)
t_test = torch.tensor(np.linspace(train_size, l - 1, test_size).astype('float32')).to(device)

N = 1.0
n = 7.413e6 
initial = 1000
y2_0 = torch.tensor([0.0001, 0.0001, 0.85]).to(device)
count = 0

def gradient(y, t, order = 1):
    if order == 1:
        return torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    else:
        return gradient(gradient(y, t), t, order = order - 1)
    
class Model(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.linear1 = nn.Linear(features, 16)
        self.linear2 = nn.Linear(16, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 128)
        self.linear5 = nn.Linear(128, 128)
        self.linear6 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = x.float().requires_grad_(True)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.linear3(x)
        x = nn.functional.relu(x)
        # x = self.dropout(x)
        # x = self.linear4(x)
        # x = nn.functional.relu(x)
        # x = self.linear5(x)
        # x = nn.functional.relu(x)
        x = self.linear6(x)
        return x 
    
def NODE(y2_0):

    model = Model(features=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loss_fn = nn.MSELoss(reduction='mean').to(device)

    def model3(t, y3):

        [E, I, pA] = y3

        [beta] = torch.reshape(model(t.view(-1, 1)), (-1, 1)).requires_grad_(True).to(device)

        eA = 10000
        eU = 50000
        gamma = 0.1
        p1 = 0.1
        p2 = 0.9
        CU = 10
        s = 0.01
        theta = 0.5
        sTheta = 0.001
        omega = 2500
        m = 1
        mu = 8.71e-3 

        epi = 1 / 2

        L = torch.reshape((1 - (1 - gamma) * pA) * CU * beta * I * (1 - E - I) / (1 - (1 - gamma) * pA * (1 - I)), (-1, )).requires_grad_(True).to(device)
        dEdt = L - epi * E - mu * E
        dIdt = torch.reshape(epi * E - mu * I, (-1, ))
        dpAdt = sTheta * (pA - p1) * (p2 - pA) * (- 1 + omega * L)

        dy3dt = torch.stack((dEdt, dIdt, dpAdt), 1).requires_grad_(True).view(-1).to(device)

        return dy3dt

    n_epochs1 = 12000 # 8000
    for epoch in range(n_epochs1):

        y_pred = odeint(model3, y2_0, t_train, method='euler')       

        y_pred = (y_pred[:, 1] * n).view(-1, 1).requires_grad_(True).to(device)
        if epoch % 100 == 0:
            print(y_pred)
        loss = loss_fn(y_pred, y_train)
        if epoch % 100 == 0:
            print(loss)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if epoch % 100 != 0:
            continue
        with torch.no_grad():
            train_rmse = np.sqrt(loss.cpu())
            y_pred1 = odeint(model3, y2_0, t_test, method='euler')[:, 1].view(-1, 1).to(device)
            loss1 = loss_fn(y_pred1 * n, y_test)
            test_rmse = np.sqrt(loss1.cpu())
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    
    return model

mo = NODE(y2_0)

def modelFinal3(t, y3):

    [E, I, pA] = y3

    [beta] = torch.reshape(mo(t.view(-1, 1)), (-1, 1)).requires_grad_(True).to(device)

    eA = 10000
    eU = 50000
    gamma = 0.1
    p1 = 0.1
    p2 = 0.9
    CU = 10
    s = 0.01
    theta = 0.5
    sTheta = 0.001
    omega = 2500
    m = 1
    mu = 8.71e-3 

    epi = 1 / 2

    L = torch.reshape((1 - (1 - gamma) * pA) * CU * beta * I * (1 - E - I) / (1 - (1 - gamma) * pA * (1 - I) - E), (-1, )).requires_grad_(True).to(device)
    dEdt = L - epi * E - mu * E
    dIdt = torch.reshape(epi * E - mu * I, (-1, ))
    dpAdt = sTheta * (pA - p1) * (p2 - pA) * (- 1 + omega * L)

    dy3dt = torch.stack((dEdt, dIdt, dpAdt), 1).requires_grad_(True).view(-1).to(device)

    return dy3dt

yFinal = (odeint(modelFinal3, y2_0, t_t, method='euler')[:, 1]).cpu().detach().numpy()

font = {'family':'Times New Roman'  #'serif', 
#         ,'style':'italic'
        ,'weight':'normal'  # 'bond' 
#         ,'color':'red'
        ,'size':18
       }

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

print(yFinal)

# from torcheval.metrics import R2Score
# metric = R2Score()
# metric.update(yFinal[2:] * n, y[2:].reshape(-1).cpu().detach())
# print(metric.compute())

# from scipy import stats
# res = yFinal[2:] * n - y[2:].reshape(-1).cpu().detach()
# v = torch.var(res)
# SSR = torch.sum(torch.square(res))
# s = torch.tensor(l - 2)
# lnLi = - s / 2 * torch.log(2 * torch.pi * v) - 1 / (2 * v) * SSR
# BIC = 8 * torch.log(s) - 2 * lnLi
# print(BIC)

plt.plot(t[2:], yFinal[2:] * n , 'b', label='Neural ODE (SEI)')
plt.scatter(t[2:], y[2:].cpu().detach().numpy(), color = 'r', label='Real BA.2 in HK')
plt.legend(loc='best', prop=font1)
plt.xlabel('$t$', font)
plt.ylabel('$N_I$', font, rotation=0)
xData = [2, 8, 16, 24, 32, 40]
xLim= ["2021-09","2022-01","2022-05","2022-09", "2023-01", "2023-04"]
plt.xticks(xData,xLim)
plt.grid()

plt.savefig('HK_SEI_PINN.png', dpi=600, format='png')
plt.show()
