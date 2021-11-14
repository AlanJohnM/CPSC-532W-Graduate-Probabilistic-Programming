import torch
import sys
import matplotlib.pyplot as plt
from primitives import env as penv
from tqdm import tqdm
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist
from copy import deepcopy

#these are adapted from Peter Norvig's Lispy
class Env():
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.data = pmap(zip(parms, args))
        self.outer = outer
        if outer is None:
            self.level = 0
        else:
            self.level = outer.level+1

    def __getitem__(self, item):
        return self.data[item]

    def find(self, var):
        "Find the innermost Env where var appears."
        if (var in self.data):
            return self
        else:
            if self.outer is not None:
                return self.outer.find(var)
            else:
                raise RuntimeError('var "{}" not found in outermost scope'.format(var))

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, sigma, body, env):
        self.parms, self.sigma, self.body, self.env = parms, sigma, body, env
        
    def __call__(self, *args):
        return EVAL(self.body, sigma=self.sigma ,rho=Env(self.parms, args, deepcopy(self.env)))

def standard_env():
    "An environment with some Scheme standard procedures."
    penv['alpha'] = ''
    return Env(penv.keys(), penv.values())

def evaluate_program(ast, sigma = {}, l = {}, rho = {}):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    
    if rho == {}:
        rho = standard_env()

    address = ''
    fn = EVAL(ast, sigma, rho)[0]
    return fn("")[0]
    
def EVAL(e, sigma, rho):

    if type(e) == str:
        if e[0] == "\"":  # strings have double, double quotes
            return e[1:-1], sigma
        if e[0:4] == 'addr':
            return e[4:], sigma
        return rho.find(e)[e], sigma
    if type(e) != list:
        return torch.tensor(e), sigma
    
    op, params, *args = e

    if op == 'sample':
        d, sigma = EVAL(args[0], sigma, rho)
        return d.sample(), sigma

    elif op == 'observe':
        fake_sample = args[0]
        return EVAL(fake_sample, sigma, rho)

    elif op == 'if':
        e1_prime, sigma = EVAL(params, sigma, rho)
        if  e1_prime:
            return EVAL(args[0], sigma, rho)
        else:
            return EVAL(args[1], sigma, rho)

    elif op == 'fn':
        return Procedure(params, sigma, args[0], rho), sigma
    
    else:
        proc = EVAL(op, sigma, rho)[0]
        vals = [EVAL(arg, sigma, rho)[0] for arg in args]
        if type(proc) == Procedure:
            return proc(*([""] + vals))
        else:
            return proc(vals), sigma
            

#def EVAL(e, sigma, rho):
#    print("e: ", e)
#    if type(e) == list and e[0] != 'fn':
#        e, address = [e[0]] + e[2:], e[1]
##    print(e)
#    if type(e) != list or len(e) == 1: # basically fns with no parameters get eaten here and that is bad, second condition is a hac
#        print("inside base case: ", e)
#        if type(e) == list:
#            e = e[0]
#            if e == 'vector': # ugly sigh
#                return torch.tensor([]), sigma
#        if type(e) in [int, float, bool]:
#            return torch.tensor(float(e)), sigma
#        elif torch.is_tensor(e):
#            return e, sigma
#        elif type(e) is str:
#            if e[0] == "\"":  # strings have double, double quotes
#                return e[1:-1], sigma
#            if e[0:4] == 'addr':
#                return e[4:], sigma
#            lowest_env = rho.find(e)
##            print(e)
##            print(lowest_env[e])
##            if type(lowest_env[e]) == Procedure:
##                print(lowest_env[e].body)
##                print(lowest_env[e].parms)
##                print(lowest_env[e].env.data)
#            #its returning a procedure here and then not calling it?
#            if type(lowest_env[e]) == Procedure and len(lowest_env[e].parms) == 1: #hacky, probably should just rewrite the whole thing
#                return lowest_env[e]("")
#            else:
#                return lowest_env[e], sigma
#
#
#    elif e[0] == 'sample':
#        d, sigma = EVAL(e[1], sigma, rho)
#        return d.sample(), sigma
#
#    elif e[0] == 'observe':
#        fake_sample = e[2]
#        return EVAL(fake_sample, sigma, rho)
#
##    elif e[0] == 'let':
##        c_1, sigma = EVAL(e[1][1], sigma, rho)
##        l[e[1][0]] = c_1
##        return EVAL(e[2], sigma, rho)
#
#    elif e[0] == 'if':
#        e1_prime, sigma = EVAL(e[1], sigma, rho)
#        if  e1_prime:
#            return EVAL(e[2], sigma, rho)
#        else:
#            return EVAL(e[3], sigma, rho)
#
#    elif e[0] == 'fn':
#        params, body =  e[1:][0], e[1:][1] #fn is:  ['fn', ['arg1','arg2','arg3'], body_exp]
##        print("in fn: ", params)
##        print("in fn: ", body)
##        p = Procedure(params, sigma, body, rho)
##        print(p.env.data)
##        return p, sigma
#        return Procedure(params, sigma, body, rho), sigma
#
#
##    elif type(e) == list:
#    else:
#        c = [0]*len(e)
#        for i in range(0, len(e)):
##            print(e[i])
##            print(EVAL(e[i], address, sigma, l, rho))
#            c[i], sigma = EVAL(e[i], sigma,  rho)
#        if type(c[0]) == Procedure:
#            print("c:", c)
#            print("c_eval:", c[0](*([""] + c[1:])))
##            print("c_eval_2:", c[0](*([""] + c[1:]))[0](""))
#            return c[0](*([""] + c[1:]))
#        else:
#            print(c)
#            return c[0](c[1:]), sigma
        
        
def get_stream(exp):
    while True:
        yield evaluate_program(exp)


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate_program(exp)
        print(f'Returned: {ret} Truth: {truth}')
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))

        print('FOPPL Tests passed')

    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate_program(exp)
        print(f'Returned: {ret} Truth: {truth}')
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


def test_program(stream, program, style):
    n_samples = 20000
    samples = []
    for i in tqdm(range(n_samples)):
        samples.append(next(stream))
        
    if program == 1:
        vals = [s.item() for s in samples]
        plt.hist(vals, bins=50)
        plt.title(f'program_{program}_output_{0}_from_{style}')
        plt.savefig(f'figures/program_{program}_output_{0}_from_{style}.png', dpi=200)
        plt.clf()
        print(f'program_{program}_output_{0}_from_{style} has marginal expectation {torch.mean(torch.tensor(vals, dtype=torch.float))}')
        print(f'program_{program}_output_{0}_from_{style} has marginal variance {torch.var(torch.tensor(vals, dtype=torch.float))}')

    elif program == 2:
        vals = [s.item() for s in samples]
        plt.hist(vals, bins=50)
        plt.title(f'program_{program}_output_{0}_from_{style}')
        plt.savefig(f'figures/program_{program}_output_{0}_from_{style}.png', dpi=200)
        plt.clf()
        print(f'program_{program}_output_{0}_from_{style} has marginal expectation {torch.mean(torch.tensor(vals, dtype=torch.float))}')
        print(f'program_{program}_output_{0}_from_{style} has marginal variance {torch.var(torch.tensor(vals, dtype=torch.float))}')
    
    elif program == 3:
        for i in range(len(samples[0])):
            vals = [s[i].item() for s in samples]
            plt.hist(vals, bins=50)
            plt.title(f'program_{program}_output_{i}_from_{style}')
            plt.savefig(f'figures/program_{program}_output_{i}_from_{style}.png', dpi=200)
            plt.clf()
            print(f'program_{program}_output_{i}_from_{style} has marginal expectation {torch.mean(torch.tensor(vals, dtype=torch.float))}')
            print(f'program_{program}_output_{i}_from_{style} has marginal variance {torch.var(torch.tensor(vals, dtype=torch.float))}')


if __name__ == '__main__':
    
    sys.setrecursionlimit(10000)
#    run_deterministic_tests()
#    run_probabilistic_tests()
    

    for i in range(3,4):
        print(i)
        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(exp))
        test_program(get_stream(exp), i, 'evaluation')
