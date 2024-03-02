from functions import get_covariance 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

class MyDS(Dataset):
    def __init__(self,X,y,e=None,transform = None):
        self.X=X
        self.y=y
        self.e=e
        self.transform=transform

    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self,idx):
        if self.e!=None:
            covariable,target,environment = self.X[idx], self.y[idx], self.e[idx]
            return covariable,target,environment
        else:
            covariable,target=self.X[idx],self.y[idx]
            return covariable,target


class SyntheticData:
    def __init__(self,
                 n_per_env,
                 n_feat,
                 x_dim,
                 n_env,
                 eta,
                 device,
                 interventions = False,
                 n_batch = 10,
                 seed = None):
        '''
        Generates data for from Gaussians for given dimensions

        n_per_env : samples per environment
        n_env : no. of environments
        n_feat : latent dimension or no. of features in Z
        x_dim : no. of X features
        interventions: bool, with or without interventions on the target
        eta : interventions strength

        attributes:
        b : true causal vector
        X,Y,Z,E : datasets for training and their respective 'test' versions
        loader: dataloader for training , full_loader is a dataloader with batch_size = len(dataset)
        '''
        self.n_per_env = n_per_env
        self.n_env = n_env
        self.n_feat = n_feat   # features in Z
        self.X_dim = x_dim
        self.interventions = interventions
        self.standardize = standardize
        self.eta = eta
        self.seed = seed
        self.cov_Z = get_covariance(n_feat + 1) ## the +1 for Y dimension
        self.cov_v = get_covariance(n_feat + 1)

        self.latent_fn()

        self.b = torch.randint(-10,10,(self.n_feat,1),dtype = torch.float32)

        self.ncond = n_env + 1

        self.X,self.Y,self.Z,self.E = self.get_train_data()

        self.X_test,self.Y_test,self.Z_test = self.get_test_data()

        self.loader, self.full_loader = self.get_loader(batch_size=(self.n_per_env*self.n_env)//n_batch)



    def latent_fn(self):
        '''
        Nontrained feedfowards to generate X from given Z
        '''
        if self.seed is not None:
            torch.manual_seed(self.seed)
        with torch.no_grad():
            self.lat_fn = nn.Sequential(nn.Linear(self.n_feat,300),nn.ReLU(),nn.Linear(300,300),nn.ReLU(),
                                        nn.Linear(300,300),nn.ReLU(), nn.Linear(300,300),nn.ReLU(), nn.Linear(300,self.X_dim))

    def get_train_data(self):
        Z = torch.zeros(size=(self.n_per_env*self.n_env,self.n_feat))
        Y = torch.zeros(size=(self.n_per_env*self.n_env,1))
        X = torch.zeros(size=(self.n_per_env*self.n_env,self.X_dim))
        E = torch.zeros(size=(self.n_env*self.n_per_env,self.n_env))

        eps_ZY = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.n_feat + 1), self.cov_Z
        ).sample((self.n_per_env*self.n_env,))

        for i in range(self.n_env):
            Z[i *self.n_per_env : (i+1)*self.n_per_env, : ] = eps_ZY[i *self.n_per_env : (i+1)*self.n_per_env, :-1] + i * torch.normal(mean = 1, std = 1, size = (self.n_per_env, self.n_feat))

            if self.interventions == True:
                Y[i*self.n_per_env:(i+1)*self.n_per_env] = (
                    Z[i*self.n_per_env:(i+1)*self.n_per_env,:] @ self.b +
                    eps_ZY[i*self.n_per_env:(i+1)*self.n_per_env,-1].unsqueeze(1) +(1/4)*i*torch.normal(mean=1,std=3,size=(self.n_per_env,1))
                )
            else:
                Y[i*self.n_per_env:(i+1)*self.n_per_env] = (
                    Z[i*self.n_per_env:(i+1)*self.n_per_env,:] @ self.b + eps_ZY[i*self.n_per_env:(i+1)*self.n_per_env,-1].unsqueeze(1)
                )

            E[i*self.n_per_env:(i+1)*self.n_per_env, i] = torch.ones(self.n_per_env)
        with torch.no_grad():
            X = self.lat_fn(Z)

        return X,Y,Z,E

    def get_loader(self,batch_size):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        ds = MyDS(self.X,self.Y,self.E)
        dl = DataLoader(ds,shuffle=True,batch_size=batch_size)
        full_dl = DataLoader(ds,shuffle=True,batch_size=len(ds))
        return dl,full_dl

    def get_test_data(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        v = torch.distributions.multivariate_normal.MultivariateNormal(
            self.eta * torch.ones(self.n_feat + 1), self.eta**2 * self.cov_v
        ).sample((self.n_per_env,))
        eps_ZY = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.n_feat + 1), self.cov_Z
        ).sample((self.n_per_env,))
        Z_test = eps_ZY[:, :-1] + v[:,:-1] 
        with torch.no_grad():
            X_test = self.lat_fn(Z_test)
        if self.interventions == True:
            Y_test=(Z_test@self.b + eps_ZY[:,-1].unsqueeze(1)) + v[:,-1].unsqueeze(1)  
        else:
            Y_test=(Z_test@self.b + eps_ZY[:,-1].unsqueeze(1))

        return X_test,Y_test,Z_test