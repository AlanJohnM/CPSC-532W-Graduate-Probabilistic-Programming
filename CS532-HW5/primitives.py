import torch
import torch.distributions as dist

class GradNormal(dist.Normal):
    
    def __init__(self, loc, scale, copy=False):
        
        if not copy:
            if scale > 20.:
                self.optim_scale = scale.clone().detach().requires_grad_()
            else:
                self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        else:
            self.optim_scale = scale

        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps, copy=True)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)
    
    def sample(self):
        return super().sample()

# citing Gaurav for this idea, was smart
class GradUniformContinuous(dist.Gamma):

    def __init__(self, low, high, copy=False):
        if not copy:
            # this is the case when you call from evaluate_program
            super().__init__(concentration=torch.tensor(float(2)),
                             rate=high.clone().detach())
        else:
            # this is the case when you go from make_copy_with_grads
            super().__init__(concentration=low,
                             rate=high)
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration, self.rate]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
        return GradUniformContinuous(*ps, copy=True)
        
    def log_prob(self, x):
        return super().log_prob(x)
    
class GradBernoulli(dist.Bernoulli):
    
    def __init__(self, probs=None, logits=None):
        if logits is None and probs is None:
            raise ValueError('set probs or logits')
        elif logits is None:
            if type(probs) is float:
                probs = torch.tensor(probs)
            logits = torch.log(probs/(1-probs)) ##will fail if probs = 0
        #
        super().__init__(logits = logits)
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.logits]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        logits = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Bernoulli(logits = logits)
    
    def sample(self):
        return super().sample()
    
    
class GradCategorical(dist.Categorical):
    
    def __init__(self, probs=None, logits=None, validate_args=None):
        
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            probs = probs / probs.sum(-1, keepdim=True)
            logits = dist.utils.probs_to_logits(probs)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        super().__init__(logits = logits)
        self.logits = logits.clone().detach().requires_grad_()
        self._param = self.logits
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.logits]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        logits = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Categorical(logits = logits)
    
    def sample(self):
        return super().sample()
    
class GradDirichlet(dist.Dirichlet):
    
    def __init__(self, concentration):
        #NOTE: logits automatically get added
        super().__init__(concentration)
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        concentration = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Dirichlet(concentration)
    
    def sample(self):
        return super().sample()
        
class GradGamma(dist.Gamma):
    
    def __init__(self, concentration, rate, copy=False):
        
        if not copy:
            if rate > 20.:
                self.optim_rate = rate.clone().detach().requires_grad_()
            else:
                self.optim_rate = torch.log(torch.exp(rate) - 1).clone().detach().requires_grad_()
        else:
            self.optim_rate = rate
        
        
        super().__init__(concentration, torch.nn.functional.softplus(self.optim_rate))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration, self.optim_rate]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        concentration,rate = [p.clone().detach().requires_grad_() for p in self.Parameters()]
        
        return Gamma(concentration, rate, copy=True)
    def log_prob(self, x):
        
        self.rate = torch.nn.functional.softplus(self.optim_rate)
        
        return super().log_prob(x)
        
    def sample(self):
        return super().sample()
    
    
class GradBeta(dist.Beta):
    
    def __init__(self, concentration1, concentration0, copy=False):
        
        if not copy:
            if concentration0 > 20.:
                self.optim_concentration0 = concentration0.clone().detach().requires_grad_()
            else:
                self.optim_concentration0 = torch.log(torch.exp(concentration0) - 1).clone().detach().requires_grad_()
        else:
            self.optim_concentration0 = concentration0
        
        super().__init__(concentration1, torch.nn.functional.softplus(self.optim_concentration0))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration1, self.optim_concentration0]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        concentration1,concentration0 = [p.clone().detach().requires_grad_() for p in self.Parameters()]
        
        return Beta(torch.FloatTensor([concentration1]), torch.FloatTensor([concentration0]), copy=True)
    def log_prob(self, x):
        self.concentration0 = torch.nn.functional.softplus(self.optim_concentration0)
        
        return super().log_prob(x)

    def sample(self):
        return super().sample()



grad_distributions = {
    'normal': GradNormal,
    'bernoulli': GradBernoulli,
    'flip': GradBernoulli,
    'categorical' : GradCategorical,
    'discrete' : GradCategorical,
    'dirichlet' : GradDirichlet,
    'gamma' : GradGamma,
    'uniform-continuous': GradUniformContinuous
}

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

def push_addr(alpha, value):
    return alpha + value


def put(args):
    try:
        args[0][args[1].item() if type(args[0]) == dict else args[1].long()] = args[2]
    except:
        args[0][args[1]] = args[2]
    return args[0]
    
def get(args):
    try:
        return args[0][args[1].item()] if type(args[0]) == dict else args[0][args[1].long()]
    except:
        return args[0][args[1]]

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

def hash_map(args):
    try:
        return {args[i].item() :args[i+1] for i in range(0,len(args),2)}
    except:
        return {args[i]:args[i+1] for i in range(0,len(args),2)}
        
def combine(format, args):
    if torch.is_tensor(args[0]) and torch.is_tensor(args[1]):
        if format in ['append','conj']:
            return torch.hstack([args[0],args[1]])
        else:
            return torch.hstack([args[1],args[0]])
    elif torch.is_tensor(args[0]) and not torch.is_tensor(args[1]):
        if format in ['append','conj']:
            return torch.hstack([args[0],torch.tensor(args[1])])
        else:
            return torch.hstack([torch.tensor(args[1]),args[0]])
    elif not torch.is_tensor(args[0]) and torch.is_tensor(args[1]):
        if format in ['append','conj']:
            try:
                return torch.hstack(args[0] + [args[1]])
            except:
                return args[0] + [args[1]]
        else:
            try:
                return torch.hstack([args[1]] + args[0])
            except:
                return [args[1]] + args[0]
        
        
non_grad_distributions = {
    'normal': Normal,
    'uniform': Uniform,
    'bernoulli': Bernoulli,
    'exponential': Exponential,
    'beta' : Beta,
    'discrete' : Discrete,
    'dirichlet' : Dirichlet,
    'gamma' : Gamma,
    'flip' : Bernoulli,
    'dirac' : Dirac,
    'uniform-continuous' : Uniform
}



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
    'peek' : lambda args: args[0][0],
    'second' : lambda args: args[0][1],
    'rest' : lambda args: args[0][1:],
    'last' : lambda args: args[0][-1],
    'nth' : lambda args: args[0][args[1]],
    'append': lambda args: combine('append', args),
    'conj' : lambda args: combine('conj', args),
    'cons' : lambda args: combine('cons', args),
    'hash-map' : lambda args: hash_map(args),
    'get' : lambda args : get(args),
    'put': lambda args : put(args),
    'empty?': lambda args : len(args[0]) == 0
}


env = {
    **primitives,
    'push-address' : push_addr,
    **non_grad_distributions
}






