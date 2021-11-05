import torch


def put(args):
    args[0][args[1].item() if type(args[0]) == dict else args[1].long()] = args[2]
    return args[0]

def vector(args): #literally the wortst thing ever
    try:
        if torch.is_tensor(args[0]):
            return torch.stack(args)
        elif type(args[0]) in [int, float]:
            return torch.tensor(args)
        else:
            return args
    except:
        return args

class BaseDistribution():

    def __init__(self, dist, params):
        self.dist = dist(*params)
        self.params = params

    def sample(self):
        return self.dist.sample()
        
    def observe(self, x):
        return self.dist.log_prob(x)
        
    def get_params(self):
        return self.params
        

class Normal(BaseDistribution):
    def __init__(self, params):
        super().__init__(torch.distributions.Normal, params)
    
class Uniform(BaseDistribution):
    def __init__(self, params):
        super().__init__(torch.distributions.Uniform, params)
        
        
class Bernoulli(BaseDistribution):
    def __init__(self, params):
        super().__init__(torch.distributions.Bernoulli, params)
    
class Exponential(BaseDistribution):
    def __init__(self, params):
        super().__init__(torch.distributions.Exponential, params)
        
class Beta(BaseDistribution):
    def __init__(self, params):
        super().__init__(torch.distributions.Beta, params)

class Discrete(BaseDistribution):
    def __init__(self, params):
        super().__init__(torch.distributions.Categorical, params)
        
class Dirichlet(BaseDistribution):
    def __init__(self, params):
        super().__init__(torch.distributions.Dirichlet, params)

class Gamma(BaseDistribution):
    def __init__(self, params):
        super().__init__(torch.distributions.Gamma, params)

class Dirac():

    # If we are using a soft dirac, this will be modelled as a normal with epsilon variance centered at x_0, otherwise it's a point mass
    def __init__(self, x_0, soft=True):
        self.x_0 = x_0[0] # all the other distributions eat lists so this one will have to deal with it
        self.eps = torch.tensor(1e-2)
        self.soft = soft
    
    def sample(self):
        return self.x_0
        
    def observe(self, x):
        if self.soft:
#            return torch.distributions.Normal(self.x_0, self.eps).log_prob(torch.clip(x,self.x_0-3*self.eps,self.x_0+3*self.eps))
            return torch.distributions.Normal(self.x_0, self.eps).log_prob(x)
        else:
            return torch.tensor(float(x == self.x_0))
        
    def get_params(self):
        return self.x_0

    
primitives = {
    # Elementary Functions
    '+': lambda args: args[0] + args[1],
    '-': lambda args: args[0] - args[1],
    '*': lambda args: args[0] * args[1],
    '/': lambda args: args[0] / args[1],
    '**': lambda args: args[0] ** args[1],
    'sqrt': lambda args: args[0] ** 0.5,
    'log': lambda args: torch.log(args[0]),
    'exp': lambda args: torch.exp(args[0]),
    'abs': lambda args: torch.abs(args[0]),
    
    # Matrix Functions
    'mat-mul' : lambda args: torch.matmul(args[0],args[1]),
    'mat-add' : lambda args: torch.add(args[0],args[1]),
    'mat-tanh' : lambda args: torch.tanh(args[0]),
    'mat-relu' : lambda args: torch.ReLU(args[0]),
    'mat-transpose' : lambda args: args[0].T,
    'mat-repmat' : lambda args: args[0].repeat((args[1].long(),args[2].long())),
    
    # Logic Operators
    '=': lambda args: torch.tensor(float(args[0] == args[1])),
    '<=': lambda args: torch.tensor(float(args[0] <= args[1])),
    '>= ': lambda args: torch.tensor(float(args[0] >= args[1])),
    '<': lambda args: torch.tensor(float(args[0] < args[1])),
    '>': lambda args: torch.tensor(float(args[0] > args[1])),
    'and': lambda args: torch.tensor(float(args[0] and args[1])),
    'or': lambda args: torch.tensor(float(args[0] or args[1])),
    
    # Data Structure Operations
    'vector' : lambda args: vector(args),
    'first' : lambda args: args[0][0],
    'second' : lambda args: args[0][1],
    'rest' : lambda args: args[0][1:],
    'last' : lambda args: args[0][-1],
    'nth' : lambda args: args[0][args[1]],
    'append': lambda args: torch.hstack([args[0],args[1]]) if torch.is_tensor(args[0]) else (args[0] + args[1]),
    'conj' : lambda args: torch.hstack([args[0],args[1]]) if torch.is_tensor(args[0]) else (args[0] + [args[1]]),
    'hash-map' : lambda args: {args[i].item() :args[i+1] for i in range(0,len(args),2)},
    'get' : lambda args : args[0][args[1].item()] if type(args[0]) == dict else args[0][args[1].long()],
    'put': lambda args : put(args)
}

distributions = {
    'normal': Normal,
    'uniform': Uniform,
    'bernoulli': Bernoulli,
    'exponential': Exponential,
    'beta' : Beta,
    'discrete' : Discrete,
    'dirichlet' : Dirichlet,
    'gamma' : Gamma,
    'flip' : Bernoulli,
    'dirac' : Dirac
}


