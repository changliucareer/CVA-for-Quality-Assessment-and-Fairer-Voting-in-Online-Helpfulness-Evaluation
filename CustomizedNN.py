import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix 
from tqdm import tqdm
import torch

# class LRNN_1layer(torch.nn.Module):
#     def __init__(self, input_dim, initial_tau):
#         super(LRNN_1layer, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, out_features=1, bias =False ).double()
#         self.tau = torch.nn.Parameter(torch.tensor(initial_tau,dtype=torch.float32),requires_grad=True)
        
#     def forward(self, x):
#         # print(f"self.tau is leaf? {self.tau.is_leaf}")
#         tau_column = torch.reshape(torch.pow(x[:,2],self.tau),(x.size(0),1))
#         x_tau = torch.cat((x[:,:2], tau_column, x[:,3:]), axis=1)
#         outputs = torch.sigmoid(self.linear(x_tau))
#         return outputs

# class LRNN_1layer_bias(torch.nn.Module):
#     def __init__(self, input_dim, initial_tau, tauColumnIndex):
#         super(LRNN_1layer_bias, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, out_features=1, bias =True ).double()
#         # self.tau = torch.nn.Parameter(torch.rand(1,dtype=torch.float32),requires_grad=True)
#         self.tau = torch.nn.Parameter(torch.tensor(initial_tau,dtype=torch.float32),requires_grad=True)
#         self.tauColIndex = tauColumnIndex

#     def forward(self, x):
#         # print(f"self.tau is leaf? {self.tau.is_leaf}")
#         # if self.tau < 0.01: # too small
#         #     # add eposilon to prevent pow() to be too small for backprop
#         #     tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],(self.tau + torch.finfo(torch.float32).eps)),(x.size(0),1)) 
#         # else:
#         #     tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],self.tau),(x.size(0),1))
#         tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],self.tau),(x.size(0),1))
#         x_tau = torch.cat((x[:,:self.tauColIndex], tau_column, x[:,self.tauColIndex+1:]), axis=1)
#         outputs = torch.sigmoid(self.linear(x_tau))
#         return outputs

class LRNN_1layer(torch.nn.Module):
    def __init__(self, input_dim, initial_tau, tauColumnIndex, positiveTau):
        super(LRNN_1layer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =False )
        self.tau = torch.nn.Parameter(torch.tensor(initial_tau,dtype=torch.float32),requires_grad=True)
        self.tauColIndex = tauColumnIndex
        self.positiveTau = positiveTau
        
    def forward(self, x):
        # print(f"self.tau is leaf? {self.tau.is_leaf}")
        if self.positiveTau: # constrain tau as positive by mapping tau to exp(tau)
            expTau = torch.exp(self.tau)
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],expTau),(x.size(0),1))
        else: # don't constrain tau
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],self.tau),(x.size(0),1))
        x_tau = torch.cat((x[:,:self.tauColIndex], tau_column, x[:,self.tauColIndex+1:]), axis=1)
        outputs = torch.sigmoid(self.linear(x_tau))
        return outputs

class LRNN_1layer_bias(torch.nn.Module):
    def __init__(self, input_dim, initial_tau, tauColumnIndex, positiveTau):
        super(LRNN_1layer_bias, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =True )
        # self.tau = torch.nn.Parameter(torch.rand(1,dtype=torch.float32),requires_grad=True)
        self.tau = torch.nn.Parameter(torch.tensor(initial_tau,dtype=torch.float32),requires_grad=True)
        self.tauColIndex = tauColumnIndex
        self.positiveTau = positiveTau

    def forward(self, x):
        # print(f"self.tau is leaf? {self.tau.is_leaf}")
        if self.positiveTau: # constrain tau as positive by mapping tau to exp(tau)
            expTau = torch.exp(self.tau)
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],expTau),(x.size(0),1))
        else: # don't constrain tau
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],self.tau),(x.size(0),1))
        x_tau = torch.cat((x[:,:self.tauColIndex], tau_column, x[:,self.tauColIndex+1:]), axis=1)
        outputs = torch.sigmoid(self.linear(x_tau))
        return outputs

class LRNN_1layer_bias_specify(torch.nn.Module):
    def __init__(self, coef_dim, nu_dim, q_dim, initial_tau):
        super(LRNN_1layer_bias_specify, self).__init__()
        self.coefs = torch.nn.Parameter(torch.rand(1,coef_dim,dtype=torch.float32),requires_grad=True) # coefficients of pos-vote-ratio, neg-vote-ratio, inverse-displayed-rank-term
        self.nus = torch.nn.Parameter(torch.rand(1,nu_dim,dtype=torch.float32),requires_grad=True) # sensitivities of relative-lenghths (question-level)
        self.qs = torch.nn.Parameter(torch.zeros(1,q_dim,dtype=torch.float32),requires_grad=True) # intrinsic qualities of answers
        self.tau = torch.nn.Parameter(torch.tensor(initial_tau,dtype=torch.float32),requires_grad=True) # the power of inverse-displayed-rank term
        self.bias = torch.nn.Parameter(torch.tensor(0,dtype=torch.float32),requires_grad=True) # universal bias 
        
    def forward(self, x):
        # print(f"self.tau is leaf? {self.tau.is_leaf}")
        tau_column = torch.reshape(torch.pow(x[:,2],self.tau),(x.size(0),1))
        x_tau = torch.cat((x[:,:2], tau_column, x[:,3:]), axis=1)
        try:
            combined_weights = torch.cat((self.coefs, self.nus, self.qs), axis=1)
            outputs = torch.sigmoid(torch.nn.functional.linear(x_tau, combined_weights, self.bias ))
            return outputs
        except Exception as e:
            print(e)

class LRNN_1layer_specify(torch.nn.Module):
    def __init__(self, coef_dim, nu_dim, q_dim, initial_tau):
        super(LRNN_1layer_specify, self).__init__()
        self.coefs = torch.nn.Parameter(torch.rand(1,coef_dim),requires_grad=True) # coefficients of pos-vote-ratio, neg-vote-ratio, inverse-displayed-rank-term
        self.nus = torch.nn.Parameter(torch.rand(1,nu_dim),requires_grad=True) # sensitivities of relative-lenghths (question-level)
        self.qs = torch.nn.Parameter(torch.zeros(1,q_dim),requires_grad=True) # intrinsic qualities of answers
        self.tau = torch.nn.Parameter(torch.tensor(initial_tau,dtype=torch.float32),requires_grad=True) # the power of inverse-displayed-rank term
        
    def forward(self, x):
        # print(f"self.tau is leaf? {self.tau.is_leaf}")
        tau_column = torch.reshape(torch.pow(x[:,2],self.tau),(x.size(0),1))
        x_tau = torch.cat((x[:,:2], tau_column, x[:,3:]), axis=1)
        try:
            combined_weights = torch.cat((self.coefs, self.nus, self.qs), axis=1)
            outputs = torch.sigmoid(torch.nn.functional.linear(x_tau, combined_weights))
            return outputs
        except Exception as e:
            print(e)
          
class LRNN_1layer_bias_specify_withoutRankTerm(torch.nn.Module):
    def __init__(self, coef_dim, nu_dim, q_dim):
        super(LRNN_1layer_bias_specify_withoutRankTerm, self).__init__()
        self.coefs = torch.nn.Parameter(torch.zeros(1,coef_dim),requires_grad=True) # coefficients of pos-vote-ratio, neg-vote-ratio, inverse-displayed-rank-term
        self.nus = torch.nn.Parameter(torch.zeros(1,nu_dim),requires_grad=True) # sensitivities of relative-lenghths (question-level)
        self.qs = torch.nn.Parameter(torch.zeros(1,q_dim),requires_grad=True) # intrinsic qualities of answers
        self.bias = torch.nn.Parameter(torch.tensor(0,dtype=torch.float32),requires_grad=True) # universal bias 
        
    def forward(self, x):
        try:
            combined_weights = torch.cat((self.coefs, self.nus, self.qs), axis=1)
            outputs = torch.sigmoid(torch.nn.functional.linear(x, combined_weights, self.bias ))
            return outputs
        except Exception as e:
            print(e)

class LRNN_1layer_bias_withoutRankTerm(torch.nn.Module):
    def __init__(self, input_dim):
        super(LRNN_1layer_bias_withoutRankTerm, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =True )
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class LRNN_1layer_withoutRankTerm(torch.nn.Module):
    def __init__(self, input_dim):
        super(LRNN_1layer_withoutRankTerm, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =False )
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class LRNN_1layer_withoutRankTerm_specifyLambda_forCVP(torch.nn.Module):
    def __init__(self, q_dim, lamb):
        super(LRNN_1layer_withoutRankTerm_specifyLambda_forCVP, self).__init__()
        self.coefs = torch.nn.Parameter(torch.rand(1,1),requires_grad=True) # coefficients of seen-pos-vote-ratio
        self.qs = torch.nn.Parameter(torch.zeros(1,q_dim),requires_grad=True) # intrinsic qualities of answers
        with torch.no_grad():
            w = torch.Tensor([lamb]).reshape(self.coefs.shape)
            self.coefs.copy_(w)
        
    def forward(self, x):
        combined_weights = torch.cat((self.coefs, self.qs), axis=1)
        outputs = torch.sigmoid(torch.nn.functional.linear(x, combined_weights ))
        return outputs
    
class LRNN_1layer_withoutRankTerm_specifyLambda_forNewModel(torch.nn.Module):
    def __init__(self, q_dim, newWeight):
        super(LRNN_1layer_withoutRankTerm_specifyLambda_forNewModel, self).__init__()
        self.lamb = torch.nn.Parameter(torch.rand(1,1),requires_grad=True) # coefficients of seen-pos-vote-ratio
        self.beta = torch.nn.Parameter(torch.rand(1,1),requires_grad=True) # coefficients of IPW displayed rank term
        self.qs = torch.nn.Parameter(torch.zeros(1,q_dim),requires_grad=True) # intrinsic qualities of answers
        with torch.no_grad():
            w = torch.Tensor([newWeight[0]]).reshape(self.lamb.shape)
            self.lamb.copy_(w)
            if newWeight[1]!=None:
                w = torch.Tensor([newWeight[1]]).reshape(self.beta.shape)
                self.beta.copy_(w)
        
    def forward(self, x):
        combined_weights = torch.cat((self.lamb, self.beta, self.qs), axis=1)
        outputs = torch.sigmoid(torch.nn.functional.linear(x, combined_weights ))
        return outputs

class LRNN_1layer_specify_withoutRankTerm(torch.nn.Module):
    def __init__(self, coef_dim, nu_dim, q_dim):
        super(LRNN_1layer_specify_withoutRankTerm, self).__init__()
        self.coefs = torch.nn.Parameter(torch.rand(1,coef_dim),requires_grad=True) # coefficients of pos-vote-ratio, neg-vote-ratio, inverse-displayed-rank-term
        self.nus = torch.nn.Parameter(torch.rand(1,nu_dim),requires_grad=True) # sensitivities of relative-lenghths (question-level)
        self.qs = torch.nn.Parameter(torch.zeros(1,q_dim),requires_grad=True) # intrinsic qualities of answers
        
    def forward(self, x):
        try:
            combined_weights = torch.cat((self.coefs, self.nus, self.qs), axis=1)
            outputs = torch.sigmoid(torch.nn.functional.linear(x, combined_weights ))
            return outputs
        except Exception as e:
            print(e)

class LRNN_1layer_interaction(torch.nn.Module):
    def __init__(self, input_dim, initial_tau, tauColumnIndex, positiveTau, interactionType):
        super(LRNN_1layer_interaction, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =False )
        self.tau = torch.nn.Parameter(torch.tensor(initial_tau,dtype=torch.float32),requires_grad=True)
        self.tauColIndex = tauColumnIndex
        self.positiveTau = positiveTau
        self.interactionType = interactionType
        
    def forward(self, x):
        # print(f"self.tau is leaf? {self.tau.is_leaf}")
        if self.positiveTau: # constrain tau as positive by mapping tau to exp(tau)
            expTau = torch.exp(self.tau)
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],expTau),(x.size(0),1))
        else: # don't constrain tau
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],self.tau),(x.size(0),1))
        
        if self.interactionType == 'D':
            interaction_column = torch.reciprocal(x[:,self.tauColIndex:self.tauColIndex+1]) * x[:,0:1]
        else: # interaction with reciprocal of D
            interaction_column = x[:,self.tauColIndex:self.tauColIndex+1] * x[:,0:1]
            
        x_interaction = torch.cat((x[:,:self.tauColIndex], tau_column, interaction_column, x[:,self.tauColIndex+1:]), axis=1)
        outputs = torch.sigmoid(self.linear(x_interaction))
        return outputs
    

class LRNN_1layer_interaction_bias(torch.nn.Module):
    def __init__(self, input_dim, initial_tau, tauColumnIndex, positiveTau, interactionType):
        super(LRNN_1layer_interaction_bias, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =True )
        self.tau = torch.nn.Parameter(torch.tensor(initial_tau,dtype=torch.float32),requires_grad=True)
        self.tauColIndex = tauColumnIndex
        self.positiveTau = positiveTau
        self.interactionType = interactionType
        
    def forward(self, x):
        # print(f"self.tau is leaf? {self.tau.is_leaf}")
        if self.positiveTau: # constrain tau as positive by mapping tau to exp(tau)
            expTau = torch.exp(self.tau)
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],expTau),(x.size(0),1))
        else: # don't constrain tau
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],self.tau),(x.size(0),1))
        
        if self.interactionType == 'D':
            interaction_column = torch.reciprocal(x[:,self.tauColIndex:self.tauColIndex+1]) * x[:,0:1]
        else: # interaction with reciprocal of D
            interaction_column = x[:,self.tauColIndex:self.tauColIndex+1] * x[:,0:1]
            
        x_interaction = torch.cat((x[:,:self.tauColIndex], tau_column, interaction_column, x[:,self.tauColIndex+1:]), axis=1)
        outputs = torch.sigmoid(self.linear(x_interaction))
        return outputs
    

class LRNN_1layer_interaction_withoutRankTerm(torch.nn.Module):
    def __init__(self, input_dim, tauColumnIndex, interactionType):
        super(LRNN_1layer_interaction_withoutRankTerm, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =False )
        self.tauColIndex = tauColumnIndex
        self.interactionType = interactionType
        
    def forward(self, x):
        if self.interactionType == 'D':
            interaction_column = torch.reciprocal(x[:,self.tauColIndex:self.tauColIndex+1]) * x[:,0:1]
        else: # interaction with reciprocal of D
            interaction_column = x[:,self.tauColIndex:self.tauColIndex+1] * x[:,0:1]
            
        x_interaction = torch.cat((x[:,:self.tauColIndex+1], interaction_column, x[:,self.tauColIndex+1:]), axis=1)
        outputs = torch.sigmoid(self.linear(x_interaction))
        return outputs
    

class LRNN_1layer_interaction_bias_withoutRankTerm(torch.nn.Module):
    def __init__(self, input_dim, tauColumnIndex, interactionType):
        super(LRNN_1layer_interaction_bias_withoutRankTerm, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =True )
        self.interactionType = interactionType
        self.tauColIndex = tauColumnIndex
        
    def forward(self, x):
        if self.interactionType == 'D':
            interaction_column = torch.reciprocal(x[:,self.tauColIndex:self.tauColIndex+1]) * x[:,0:1]
        else: # interaction with reciprocal of D
            interaction_column = x[:,self.tauColIndex:self.tauColIndex+1] * x[:,0:1]
            
        x_interaction = torch.cat((x[:,:self.tauColIndex+1], interaction_column, x[:,self.tauColIndex+1:]), axis=1)
        outputs = torch.sigmoid(self.linear(x_interaction))
        return outputs
    
class LRNN_1layer_quadratic(torch.nn.Module):
    def __init__(self, input_dim, initial_tau, tauColumnIndex, positiveTau):
        super(LRNN_1layer_quadratic, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =False )
        self.tau = torch.nn.Parameter(torch.tensor(initial_tau,dtype=torch.float32),requires_grad=True)
        self.tauColIndex = tauColumnIndex
        self.positiveTau = positiveTau
        
    def forward(self, x):
        # print(f"self.tau is leaf? {self.tau.is_leaf}")
        if self.positiveTau: # constrain tau as positive by mapping tau to exp(tau)
            expTau = torch.exp(self.tau)
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],expTau),(x.size(0),1))
        else: # don't constrain tau
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],self.tau),(x.size(0),1))
        
        quadratic_column = x[:,self.tauColIndex:self.tauColIndex+1] * x[:,self.tauColIndex:self.tauColIndex+1]
            
        x_quadratic = torch.cat((x[:,:self.tauColIndex], tau_column, quadratic_column, x[:,self.tauColIndex+1:]), axis=1)
        outputs = torch.sigmoid(self.linear(x_quadratic))
        return outputs
    

class LRNN_1layer_quadratic_bias(torch.nn.Module):
    def __init__(self, input_dim, initial_tau, tauColumnIndex, positiveTau):
        super(LRNN_1layer_quadratic_bias, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =True )
        self.tau = torch.nn.Parameter(torch.tensor(initial_tau,dtype=torch.float32),requires_grad=True)
        self.tauColIndex = tauColumnIndex
        self.positiveTau = positiveTau
        
    def forward(self, x):
        # print(f"self.tau is leaf? {self.tau.is_leaf}")
        if self.positiveTau: # constrain tau as positive by mapping tau to exp(tau)
            expTau = torch.exp(self.tau)
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],expTau),(x.size(0),1))
        else: # don't constrain tau
            tau_column = torch.reshape(torch.pow(x[:,self.tauColIndex],self.tau),(x.size(0),1))
        
        quadratic_column = x[:,self.tauColIndex:self.tauColIndex+1] * x[:,self.tauColIndex:self.tauColIndex+1]
            
        x_quadratic = torch.cat((x[:,:self.tauColIndex], tau_column, quadratic_column, x[:,self.tauColIndex+1:]), axis=1)
        outputs = torch.sigmoid(self.linear(x_quadratic))
        return outputs
    

class LRNN_1layer_quadratic_withoutRankTerm(torch.nn.Module):
    def __init__(self, input_dim, tauColumnIndex):
        super(LRNN_1layer_quadratic_withoutRankTerm, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =False )
        self.tauColIndex = tauColumnIndex
        
    def forward(self, x):
        quadratic_column = x[:,self.tauColIndex:self.tauColIndex+1] * x[:,self.tauColIndex:self.tauColIndex+1]
            
        x_quadratic = torch.cat((x[:,:self.tauColIndex+1], quadratic_column, x[:,self.tauColIndex+1:]), axis=1)
        outputs = torch.sigmoid(self.linear(x_quadratic))
        return outputs
    

class LRNN_1layer_quadratic_bias_withoutRankTerm(torch.nn.Module):
    def __init__(self, input_dim, tauColumnIndex):
        super(LRNN_1layer_quadratic_bias_withoutRankTerm, self).__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1, bias =True )
        self.tauColIndex = tauColumnIndex
        
    def forward(self, x):
        quadratic_column = x[:,self.tauColIndex:self.tauColIndex+1] * x[:,self.tauColIndex:self.tauColIndex+1]
            
        x_quadratic = torch.cat((x[:,:self.tauColIndex+1], quadratic_column, x[:,self.tauColIndex+1:]), axis=1)
        outputs = torch.sigmoid(self.linear(x_quadratic))
        return outputs