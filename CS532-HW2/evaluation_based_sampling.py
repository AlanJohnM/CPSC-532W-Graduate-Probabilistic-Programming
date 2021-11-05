from daphne import daphne
import torch
from tests import is_tol, run_prob_test,load_truth
from primitives import primitives, distributions
import matplotlib.pyplot as plt

# Borrowed from Norvig
class Procedure():
    
    def __init__(self, params, body, env):
        self.params, self.body, self.env = params, body, env
        
    def __call__(self, args):
        parameters = {self.params[i]:args[i] for i in range(0,len(self.params))}
        return EVAL(self.body, {}, parameters, self.env)[0]


def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    rho = {}
    l = {}
    sigma = {}
    
    for e in ast:
        if type(e) == list and e[0] == 'defn':
            rho[e[1]] = Procedure(e[2], e[3], rho)
    ast = list(filter(lambda e : (type(e) != list or e[0] != 'defn'), ast))
    return EVAL(ast[0], sigma, l, rho)[0]

def EVAL(e, sigma, l, rho):
    if type(e) != list or len(e) == 1:
        if type(e) == list:
            e = e[0]
            if e == 'vector': # ugly sigh
                return torch.tensor([]), sigma
        if type(e) in [int, float]:
            return torch.tensor(float(e)), sigma
        elif torch.is_tensor(e):
            return e, sigma
        elif e in primitives.keys():
            return primitives[e], sigma
        elif e in distributions.keys():
            return distributions[e], sigma
        elif e in l.keys():
            return l[e], sigma
        elif e in rho.keys():
            return rho[e], sigma
        
    elif e[0] == 'sample':
        d, sigma = EVAL(e[1], sigma, l, rho)
        return d.sample(), sigma

    elif e[0] == 'observe':
        fake_sample = e[2]
        return EVAL(fake_sample, sigma, l, rho)
        
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
            return c[0](c[1:]), sigma


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

    
#    run_deterministic_tests()
#    run_probabilistic_tests()

    for i in range(4,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)) # note there is still discrepency between return sigma and not
        test_program(get_stream(ast), i, 'evaluation')
        
