import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def recon_loss(x_mean,x_logvar,x):
    '''
    Takes mean and logvar from the encoder, but it learns to set logvar to minus infinity so
    I've put logvar times 0

    Outputs negative log likelihood
    '''
    var = torch.exp(x_logvar)
    NegLogLikeL = 0.5*torch.sum( x_logvar + torch.pow(x-x_mean,2)/var )  # -0.5*log(2*pi)
    return NegLogLikeL

def KLD(v_mean,v_logvar,prior_mean,prior_logvar):
    '''
    KL( q(z|x) || p(z|y,a) ) between variational (encoder) and prior (fortheprior function)

    Outputs the KLD
    '''
    logs=v_logvar - prior_logvar
    quads=torch.pow(v_mean-prior_mean,2) / torch.exp(prior_logvar)
    traces=torch.exp(v_logvar)/torch.exp(prior_logvar)

    return -0.5*torch.sum( 1 + logs - quads - traces )

def train_cvae(model,data,optimizer,n_iters,device,
               output_all = False, gamma = 5, plot = True):
    model.train()
    trains, validations = [], []
    for epoch in range(n_iters):
        epoch_loss = 0
        recon = 0
        dkl = 0
        for x,y,e in data.loader:
            x,y,e = x.to(device),y.to(device),e.to(device)
            optimizer.zero_grad()

            x_mean,x_logvar,v_mean,v_logvar = model(x)
            prior_mean,prior_logvar=model.fortheprior(y,e)

            dkl_loss = KLD(v_mean,v_logvar,prior_mean,prior_logvar)
            dkl += dkl_loss
            rec_loss = recon_loss(x_mean, x_logvar, x)
            recon += rec_loss

            loss = dkl_loss + rec_loss
            loss.backward()
            epoch_loss += loss
            optimizer.step()

        if torch.isnan(epoch_loss):
            return 'Blew up'

        elif output_all:
            print('{}/{}, epoch loss: {:.4f}, recons. loss: {:.4f}, DKL: {:.4f}'.format(epoch+1,n_iters,epoch_loss.item(),recon.item(),dkl.item()))
            trainerror, testerror = evaluate_c(model, data, device, gamma = gamma)
            if plot:
                trains.append(trainerror.item())
                validations.append(testerror.item())
            model.train()
            print('In sample total: {:.4f},   On test: {:.4f}'.format(trainerror, testerror),'\n')

    print('Done, final loss: {:.4f}'.format(epoch_loss))
    if plot:
        plt.figure(figsize = (6,4))
        plt.plot(trains, c = 'blue')
        plt.grid(True)
        plt.title('Train error')
        plt.show()
        plt.plot(validations, c = 'red')
        plt.grid(True)
        plt.title('Test error')
        plt.show()
    return trains, validations


    return None

def evaluate_c(model, data, device, gamma = 5):
    data.X, data.X_test = data.X.to(device), data.X_test.to(device)
    data.Y, data.Y_test = data.Y.to(device), data.Y_test.to(device)
    model.eval()
    V_mean = model(data.X)[2]

    center = torch.mean(V_mean[:data.n_per_env], dim = 0)
    y_center = torch.mean(data.Y[:data.n_per_env])

    V_mean_centered = V_mean - center
    Y_centered = data.Y - y_center

    drig = drig_est(V_mean_centered, Y_centered, gamma = gamma, m = data.n_env, d = data.n_feat)
    V_mean_test = model(data.X_test)[2]

    trainerror = F.mse_loss(V_mean_centered @ drig, Y_centered)
    testerror = F.mse_loss((V_mean_test-center) @ drig, (data.Y_test-y_center))
    return trainerror, testerror
def train(model,data,optimizer,n_iters,device, out_all, plot = True):
    '''
    For training baselines
    '''
    model.train()
    model.to(device)
    vals = []
    for epoch in range(n_iters):
        epoch_loss=0
        for x,y,e in data.loader:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad()
            y_hat=model(x)
            loss=F.mse_loss(y_hat,y)
            epoch_loss+=loss
            loss.backward()
            optimizer.step()
        if torch.isnan(epoch_loss):
            return 'Blew up'
        if out_all:
            model.eval()
            insample = F.mse_loss(model(data.X.to(device)), data.Y.to(device))
            test = F.mse_loss(model(data.X_test.to(device)), data.Y_test.to(device))
            if plot:
                vals.append((insample.item(),test.item()))
            print("Epoch: {}/{} In Sample: {:.4f},    test: {:.4f}   ".format(epoch+1,n_iters,insample,test))
            model.train()
    if plot:
        plt.plot(vals)
    print('Done, final loss: {:.4f}'.format(epoch_loss))
def drig_est(X,Y,gamma,m,d):
    '''
    m : number of environments
    d : number of features of X

    Estimtes DRIG , used for VAE
    Input: Y,        NOT Y_cond
    Output: DRIG
    '''
    size=len(Y)//m
    x_e=X.reshape(m,size,d)
    y_e=Y.reshape(m,size,1)
    G=(1-gamma)*(x_e[0].t()@x_e[0]/size) + (1/m)*gamma*sum([x_e[i].t()@x_e[i]/size for i in range(1,m)])
    Zz=(1-gamma)*(x_e[0].t()@y_e[0]/size) + (1/m)*gamma*sum([x_e[i].t()@y_e[i]/size for i in range(1,m)])
    return torch.inverse(G) @ Zz

def base_eval(model, data):
    model.eval()
    model.to('cpu')
    y_pred = model(data.X.to('cpu'))
    print(f'Baseline in sample: {F.mse_loss(y_pred, data.Y)}')
    y_pred_test = model(data.X_test.to('cpu'))
    print(f'Baseline on test: {F.mse_loss(y_pred_test, data.Y_test)}')
    model.train()

def get_covariance(n_feats):
    L = torch.rand(n_feats**2).reshape(n_feats,n_feats)
    M = L.t() @ L
    return M / torch.linalg.matrix_norm(M)