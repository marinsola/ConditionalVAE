import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CVAE(nn.Module):
    def __init__(self, hidden_size, lat_dim,ncond=0, dropout=0.0):
        super(CVAE, self).__init__()

        '''
        Conditional VAE
        hidden_size: list of layer widths to be used for all feedfowards here, includes in_dims, but not lat_dims
        lat_dim: Latent (Z) dimension, no. of Z features
        ncond: no. of enivornments + no. of Y features
        dropout: nn.Dropout(p = 0.0)
        '''

        self.hidden_size = hidden_size
        self.lat_dim = lat_dim
        self.ncond=ncond
        self.dropout=dropout


        self.encoder = self.construct_mlp(self.hidden_size, last_activation=True)

        self.vario_mean = nn.Sequential(nn.Linear(self.hidden_size[-1],self.lat_dim),nn.LeakyReLU(0.1))

        self.vario_logvar = nn.Sequential(nn.Linear(self.hidden_size[-1],self.lat_dim),nn.LeakyReLU(0.1))

        self.decoder = nn.Sequential(nn.Linear(self.lat_dim,self.hidden_size[-1]),nn.LeakyReLU(0.1),nn.Dropout(self.dropout),
                                     self.construct_mlp(self.hidden_size[:0:-1], last_activation=True)
                                     )

        self.x_mean = nn.Sequential(nn.Linear(self.hidden_size[1],self.hidden_size[0]),nn.LeakyReLU(0.1))
        self.x_logvar = nn.Sequential(nn.Linear(self.hidden_size[1],self.hidden_size[0]),nn.LeakyReLU(0.1))

        self.prioritize = nn.Sequential(nn.Linear(self.ncond,self.hidden_size[1]),nn.LeakyReLU(0.1),nn.Dropout(self.dropout),
                                        self.construct_mlp(self.hidden_size[1:])
                                        )

        self.prior_mean = nn.Sequential(nn.Linear(self.hidden_size[-1],self.lat_dim),nn.LeakyReLU(0.1))

        self.prior_logvar = nn.Sequential(nn.Linear(self.hidden_size[-1],self.lat_dim),nn.LeakyReLU(0.1))

    def construct_mlp(self, sizes, last_activation=True) :
        q = []
        for i in range(len(sizes)-1):
            in_dim = sizes[i]
            out_dim = sizes[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(sizes)-2) or ((i == len(sizes) - 2) and (last_activation)):
                q.append(("LeakyReLU_%d" % i, nn.LeakyReLU(0.1)))
                q.append(("Dropout_%d" % i, nn.Dropout(p=self.dropout)))

        return nn.Sequential(OrderedDict(q))

    def encode(self, x):
        '''
        Returns variational mean and variational log(variance)
        '''
        xnew = self.encoder(x)
        mean,logvar = self.vario_mean(xnew),self.vario_logvar(xnew)
        return mean,logvar

    def fortheprior(self, y, a):
        '''
        Returns prior mean and prior log(variance)
        '''
        conditions=torch.cat((y,a), dim = 1)
        raws = self.prioritize(conditions)
        prior_mean,prior_logvar = self.prior_mean(raws),self.prior_logvar(raws)
        return prior_mean,prior_logvar

    def reparametrize(self,mean,logvar):
        z_new = mean + torch.normal(mean=0,std=1,size=logvar.shape).to(device)*torch.exp(0.5*logvar)
        return z_new

    def decode(self, z):
        zz = self.decoder(z)
        x_mean,x_logvar = self.x_mean(zz),self.x_logvar(zz)
        return x_mean,x_logvar

    def forward(self, x):
        v_mu, v_logvar = self.encode(x)
        z_new = self.reparametrize(v_mu,v_logvar)
        x_mean, x_logvar = self.decode(z_new)
        return x_mean,x_logvar,v_mu,v_logvar