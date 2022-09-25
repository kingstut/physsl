import torch 
import torch.nn as nn 
import torch.nn.functional as F
import scipy  
import resnet 

class CohereNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(self.embedding)

        z_x = torch.randn(1)
        z_y = torch.randn(1)

        self.mu_x = nn.Parameter(z_x)
        self.k_x = nn.Parameter(z_x*100)
        self.mu_y = nn.Parameter(z_y)
        self.k_y = nn.Parameter(z_x*100)

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

        loss_diff =  F.mse_loss(self.mu_x , self.mu_y) + F.mse_loss(self.k_x , self.k_y)
        f, coh_x_y = scipy.signal.coherence(wave_x, wave_y)
        
        loss_coh = torch.mean((1 - coh_x_y)**2)

        loss = (
            self.args.wave_coeff * loss_wave
            + self.args.diff_coeff * loss_diff
            + self.args.coh_coeff * loss_coh
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
    