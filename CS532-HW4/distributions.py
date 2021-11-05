import torch
import torch.distributions as dist

class Normal(dist.Normal):
    
    def __init__(self, loc, scale, copy=False):
        
        if not copy:
            if scale > 20.:
                self.optim_scale = scale.clone().detach().requires_grad_()
            else:
                self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        else:
            self.optim_scale = scale
        
        #print ('here', self.optim_scale)
        
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
class UniformContinuous(dist.Gamma):

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
        return UniformContinuous(*ps, copy=True)
        
    def log_prob(self, x):
        return super().log_prob(x)
    
class Bernoulli(dist.Bernoulli):
    
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
    
    
class Categorical(dist.Categorical):
    
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
    
class Dirichlet(dist.Dirichlet):
    
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
        
class Gamma(dist.Gamma):
    
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
    
    
class Beta(dist.Beta):
    
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
        
        #print (self.concentration0, self.concentration1)
        #self.concentration1 = 5
        #print ('here', torch.nn.functional.softplus(self.optim_concentration0))
        #print (se)
        
        #self.concentration0 = torch.nn.functional.softplus(self.optim_concentration0)
        #self.optim_concentration0 = torch.FloatTensor([torch.nn.functional.softplus(self.optim_concentration0)])
        
        self.concentration0 = torch.nn.functional.softplus(self.optim_concentration0)
        
        return super().log_prob(x)
    def sample(self):
        return super().sample()


vi_distributions = {
    'normal': Normal,
    'bernoulli': Bernoulli,
    'flip': Bernoulli,
    'categorical' : Categorical,
    'discrete' : Categorical,
    'dirichlet' : Dirichlet,
    'gamma' : Gamma,
    'uniform-continuous': UniformContinuous
}

if __name__ == '__main__':
    
    ##how to use this, 
    #given some input tensors that don't necessarily have gradients enables
    scale = torch.tensor(1.)
    loc = torch.tensor(0.)
    
    #and some data
    data = torch.tensor(2.)
    
    #construct a distribution
    d = Normal(loc, scale)
    
    #now you can make a copy, that has gradients enabled
    dg = d.make_copy_with_grads()
    
    #the function .Parameters() returns a list of parameters that you can pass to an optimizer
    optimizer = torch.optim.Adam(dg.Parameters(), lr=1e-2)
    
    #do the optimization. Here we're maximizing the log_prob of some data at 2.0
    #the scale should move to 2.0 as well,
    #furthermore, the scale should be constrained to the positive reals,
    #this last thing is taken care of by the new distributions defined above
    for i in range(1000):
        nlp = -dg.log_prob(data)
        nlp.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    #check the result is correct:
    print(dg.Parameters())
    
    
    #note: Parameters() returns a list of tensors that parametrize the distributions
    # gradients can be taken with respect to these parameters, and you can assume gradient updates are "safe"
    # i.e., under the hood, parameters constrained to the positive reals are transformed so that they can be optimized
    # over without worrying about the constraints
    
