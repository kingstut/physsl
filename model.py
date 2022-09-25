import torch 
import torch.nn as nn 
import torch.nn.functional as F
import scipy  
import resnet 

class CoherenceNet(nn.Module):
    def __init__(self, arch="resnet50", wave_coeff=1, diff_coeff=1, coh_coeff=1):
        super().__init__()
        self.backbone, self.embedding = resnet.__dict__[arch](
            zero_init_residual=True
        )
        self.projector = Projector(self.embedding)

        z_x = torch.randn(1)
        z_y = torch.randn(1)

        self.mu_x = nn.Parameter(z_x*10)
        self.k_x = nn.Parameter(z_x*1000)
        self.mu_y = nn.Parameter(z_y*10)
        self.k_y = nn.Parameter(z_x*1000)

    def forward(self, x, y):
        x_phy = self.backbone(x)
        y_phy = self.backbone(y)

        wave_x = self.projector(x_phy)
        wave_y = self.projector(y_phy)

        dx  = torch.autograd.grad(wave_x, x_phy, torch.ones_like(wave_x), create_graph=True)[0]# computes dw1/dx
        dx2 = torch.autograd.grad(dx,  x_phy, torch.ones_like(dx),  create_graph=True)[0]# computes d^2(w1)/dx^2

        dy  = torch.autograd.grad(wave_y, y_phy, torch.ones_like(wave_y), create_graph=True)[0]# computes dw2/dy
        dy2 = torch.autograd.grad(dy, y_phy, torch.ones_like(dx),  create_graph=True)[0]# computes d^2(w2)/dy^2

        wavy_x = dx2 + self.mu_x*dx + self.k_x*wave_x
        loss_wx = torch.mean(wavy_x**2)

        wavy_y = dy2 + self.mu_y*dy + self.k_y*wave_y
        loss_wy = torch.mean(wavy_y**2)

        loss_wave = loss_wx + loss_wy

        loss_diff =  - (F.mse_loss(self.mu_x , self.mu_y) + F.mse_loss(self.k_x , self.k_y))
        f, coh_x_y = scipy.signal.coherence(wave_x, wave_y)
        
        loss_coh = torch.mean((1 - coh_x_y)**2)

        loss = (
            self.wave_coeff * loss_wave
            + self.diff_coeff * loss_diff
            + self.coh_coeff * loss_coh
        )
        return loss

def Projector(embedding):
    N_INPUT = embedding
    N_HIDDEN = int(embedding*0.75)
    N_OUTPUT = embedding

    layers = []
    layers.append(nn.Linear(N_INPUT, N_HIDDEN))
    layers.append(nn.BatchNorm1d(N_HIDDEN))
    layers.append(nn.ReLU(True))
    layers.append(nn.Linear(N_HIDDEN, N_HIDDEN))
    layers.append(nn.BatchNorm1d(N_HIDDEN))
    layers.append(nn.ReLU(True))
    layers.append(nn.Linear(N_HIDDEN, N_OUTPUT))

    return nn.Sequential(*layers)
    
class ResonanceNet(nn.Module):
    def __init__(self, arch="resnet50", wave_coeff=1, res_coeff=1):
        super().__init__()
        self.backbone, self.embedding = resnet.__dict__[arch](
            zero_init_residual=True
        )
        self.projector = Projector(self.embedding)

        z_x = torch.randn(1)
        z_y = torch.randn(1)

        self.mu_x = nn.Parameter(z_x*10)
        self.k_x = nn.Parameter(z_x*1000)
        self.w_y = nn.Parameter(z_y*10)

    def forward(self, x, y):
        x_phy = self.backbone(x)
        y_phy = self.backbone(y)

        wave_x = self.projector(x_phy)
        wave_y = self.projector(y_phy)

        dx  = torch.autograd.grad(wave_x, x_phy, torch.ones_like(wave_x), create_graph=True)[0]# computes dw1/dx
        dx2 = torch.autograd.grad(dx,  x_phy, torch.ones_like(dx),  create_graph=True)[0]# computes d^2(w1)/dx^2

        dy  = torch.autograd.grad(wave_y, y_phy, torch.ones_like(wave_y), create_graph=True)[0]# computes dw2/dy
        dy2 = torch.autograd.grad(dy, y_phy, torch.ones_like(dx),  create_graph=True)[0]# computes d^2(w2)/dy^2

        wavy_x = dx2 + self.mu_x*dx + self.k_x*wave_x
        loss_wx = torch.mean(wavy_x**2)

        wavy_y = (self.w_y**2) * dy2 + wave_y
        loss_wy = torch.mean(wavy_y**2)

        loss_wave = loss_wx + loss_wy

        #loss_diff =  F.mse_loss(self.mu_x , self.mu_y) + F.mse_loss(self.k_x , self.k_y)
        f, coh_x_y = scipy.signal.coherence(wave_x, wave_y)
        
        w_x = (self.k_x)**0.5
        loss_res = 1 / ((self.w_y**2 - w_x**2)**2 + (self.w_y**2 * self.mu_x**2))**0.5

        loss = (
            self.wave_coeff * loss_wave
            #+ self.diff_coeff * loss_diff
            + self.res_coeff * loss_res
        )
        return loss