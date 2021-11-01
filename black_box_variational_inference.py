from daphne import daphne
import torch
import numpy as np
from tests import is_tol, run_prob_test,load_truth
from primitives import primitives
from distributions import vi_distributions
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchviz import make_dot
import wandb



# Borrowed from Norvig
class Procedure():
    
    def __init__(self, params, sigma, body, env):
        self.params, self.sigma, self.body, self.env = params, sigma, body, env
        
    def __call__(self, args):
        parameters = {self.params[i]:args[i] for i in range(0,len(self.params))}
        return EVAL(self.body, self.sigma, parameters, self.env)[0]


def evaluate_program(ast, sigma = {}, l = {}, rho = {}):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    
    for e in ast:
        if type(e) == list and e[0] == 'defn':
            rho[e[1]] = Procedure(e[2], sigma, e[3], rho)
    ast = list(filter(lambda e : (type(e) != list or e[0] != 'defn'), ast))
    return EVAL(ast[0], sigma, l, rho)

def EVAL(e, sigma, l, rho):
#    print(e)
    if type(e) != list or len(e) == 1:
        if type(e) == list:
            e = e[0]
            if e == 'vector': # ugly sigh
                return torch.tensor([]), sigma
        if type(e) in [int, float, bool]:
            return torch.tensor(float(e)), sigma
        elif torch.is_tensor(e):
            return e, sigma
        elif e in primitives.keys():
            return primitives[e], sigma
        elif e in vi_distributions.keys():
            return vi_distributions[e], sigma
        elif e in l.keys():
            return l[e], sigma
        elif e in rho.keys():
            return rho[e], sigma
        
    elif e[0] == 'sample':
        v = id(e[1]) # highly questionable
        d, sigma = EVAL(e[1], sigma, l, rho)
        if v not in sigma['Q'].keys():
            sigma['Q'][v] = d.make_copy_with_grads()
            optimizer.param_groups[0]['params'] += sigma['Q'][v].Parameters()
        c = sigma['Q'][v].sample()
        q_log_prob = sigma['Q'][v].log_prob(c)
        q_log_prob.backward()
        sigma['G'][v] = [param.grad.clone().detach() for param in sigma['Q'][v].Parameters()]
        log_W_v = d.log_prob(c) - q_log_prob
        sigma['log_W'] = sigma['log_W'] + log_W_v
        return c, sigma

    elif e[0] == 'observe':
        d, sigma = EVAL(e[1], sigma, l, rho)
        c, sigma = EVAL(e[2], sigma, l, rho)
        sigma['log_W'] = sigma['log_W'] + d.log_prob(c).detach()
        return c, sigma
        
    elif e[0] == 'let':
        c_1, sigma = EVAL(e[1][1], sigma, l, rho)
        l[e[1][0]] = c_1
        return EVAL(e[2], sigma, l, rho)
        
    elif e[0] == 'if':
        e1_prime, sigma = EVAL(e[1],sigma,l, rho)
        if  e1_prime:
            return EVAL(e[2],sigma,l, rho)
        else:
            return EVAL(e[3],sigma,l, rho)

    elif type(e) == list:
        c = [0]*len(e)
        for i in range(0, len(e)):
            c[i], sigma = EVAL(e[i], sigma, l, rho)
        if c[0] in rho.keys():
            return rho[c[0]](c[1:]), sigma
        else:
            if c[0] in vi_distributions.values():
                return c[0](*c[1:]), sigma
            else:
                return c[0](c[1:]), sigma


def optimizer_step(Q, g_hat):
    for v  in g_hat.keys():
        for i, param in enumerate(Q[v].Parameters()):
            param.grad = -g_hat[v][i]
    optimizer.step()
    optimizer.zero_grad()
#    print(Q)
    return Q

def elbo_gradients(G, log_W): # this may be the worst code ive ever written
    G_union = {k:v for g in G for k,v in g.items()}
    param_counts = {k:len(v) for k,v in G_union.items()}
    param_shapes = {k:{id(p):p.shape for p in v} for k,v in G_union.items()}
    log_W = torch.tensor(log_W)
    g_hat = {v:[torch.zeros(param_shapes[v][p]) for p in param_shapes[v].keys()] for v in G_union}
    F = [{v:[torch.zeros(param_shapes[v][p]) for p in param_shapes[v].keys()] for v in G_union} for i in range(len(G))]
    for v in G_union.keys(): # for each random variable
        for l in range(len(G)): # for each gradient itteration
            if v in G[l].keys():
                for i, grad in enumerate(G[l][v]):
                    F[l][v][i] = grad*log_W[l]
        F_v, G_v, b_hat_v = [0.0]*param_counts[v],[0.0]*param_counts[v],[0.0]*param_counts[v]
        for i, p in enumerate(param_shapes[v].keys()):
            F_v[i], G_v[i] = torch.zeros((*param_shapes[v][p], len(G))), torch.zeros((*param_shapes[v][p], len(G)))
            F_v[i] = torch.stack([F[l][v][i] for l in range(len(G))])
            G_v[i] = torch.stack([G[l][v][i] for l in range(len(G))])
            if len(list(param_shapes[v][p])) != 0:
#                b_hat_v[i] = torch.zeros(param_shapes[v][p])
                b_hat_v[i] = torch.ones(param_shapes[v][p])
#                for j in range(len(list(param_shapes[v][p]))):
#                    b_hat_v[i][j] = torch.sum(torch.cov(torch.stack((F_v[i][j], G_v[i][j]))))/torch.sum(torch.var(G_v[i][j]))
            else:
#                b_hat_v[i] = torch.sum(torch.cov(torch.stack((F_v[i], G_v[i]))))/torch.sum(torch.var(G_v[i]))
                b_hat_v[i] = 1.0
            g_hat[v][i] = torch.sum(F_v[i] - b_hat_v[i]*G_v[i], dim = 0)/len(G)
    return g_hat
    
lr = 1e-2
T = 1000
optimizer = torch.optim.Adam([torch.tensor(0.0)], lr=lr)




def black_box_variational_inference(ast, L):
#    wandb.init(project="bbvi", entity="alanmilligan")
#    wandb.config = {
#      "learning_rate":lr,
#      "epochs": L,
#      "batch_size": T
#    }

    samples = []
    Q = {}
    bar = tqdm(range(L))
    for t in bar:
        log_W_t, G_t = [0.0]*T, [0.0]*T
        for l in range(T):
            sigma = {'log_W':0, 'G':{}, 'Q':Q}
            r_tl, sigma_tl = evaluate_program(ast,sigma=sigma)
            log_W_t[l], G_t[l] = sigma['log_W'], sigma['G']
#            print(log_W_t[l].item())
            samples.append((log_W_t[l], r_tl))
            optimizer.zero_grad()
        g_hat = elbo_gradients(G_t, log_W_t)
        Q = optimizer_step(Q, g_hat)
        print(torch.mean(torch.tensor(log_W_t)).item())
        
        bar.set_postfix({'ELBO':torch.mean(torch.tensor(log_W_t)).item()})
#        wandb.log({'ELBO':torch.mean(torch.tensor(log_W_t)).item()})
#        for k,q in Q.items():
#            wandb.log({'Mu':q.Parameters()[0]})
#            wandb.log({'Sigma':q.Parameters()[1]})
    return samples







def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)

def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        print(f'test_{i}.daphne desugars to ',ast)
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))

        ret, sig = evaluate_program(ast), {}

        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))

        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        print(f'test_{i}.daphne desugars to ',ast)
        
        stream = get_stream(ast)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')
    
    
def test_program(stream, program, style):
    n_samples = 1000
    samples = []
    for i in range(n_samples):
        samples.append(next(stream))
    for i in range(len(samples[0])):
        if samples[0][i].dim() == 0:
            vals = [s[i].item() for s in samples]
            plt.hist(vals, bins=50)
            plt.title(f'program_{program}_output_{i}_from_{style}')
            plt.savefig(f'figures/program_{program}_output_{i}_from_{style}.png', dpi=200)
            plt.clf()
            print(f'program_{program}_output_{i}_from_{style} has marginal expectation {torch.mean(torch.tensor(vals, dtype=torch.float))}')
        else:
            for j in range(len(samples[0][i])):
                for k in range(len(samples[0][i][j])):
                    vals = [s[i][j][k].item() for s in samples]
                    plt.hist(vals, bins=50)
                    plt.title(f'program_{program}_output_{i}_{j}_{k}_from_{style}')
                    plt.savefig(f'figures/program_{program}_output_{i}_{j}_{k}_from_{style}.png', dpi=200)
                    plt.clf()
                    print(f'program_{program}_output_{i}_{j}_{k}_from_{style} has marginal expectation {torch.mean(torch.tensor(vals, dtype=torch.float))}')



        
if __name__ == '__main__':

    
    run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)) # note there is still discrepency between return sigma and not
        test_program(get_stream(ast), i, 'evaluation')
        
