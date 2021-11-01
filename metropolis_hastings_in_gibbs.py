import torch
import torch.distributions as dist
import copy
from daphne import daphne

from tqdm import tqdm
from primitives import primitives, distributions
from evaluation_based_sampling import test_program
from tests import is_tol, run_prob_test,load_truth


# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
#env = {'normal': dist.Normal,
#       'sqrt': torch.sqrt}

def deterministic_eval(e, env = {}):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(e) != list or len(e) == 1:
        if type(e) == list:
            e = e[0]
            if e == 'vector': # ugly sigh
                return torch.tensor([])
        if type(e) in [int, float, bool]:
            return torch.tensor(float(e))
        elif torch.is_tensor(e):
            return e
        elif e in primitives.keys():
            return primitives[e]
        elif e in distributions.keys():
            return distributions[e]
        elif e in env.keys():
            return env[e]
    elif e[0] == 'let':
        env[e[1][0]] = deterministic_eval(e[1][1], env)
        return deterministic_eval(e[2], env)
    elif e[0] == 'if':
        return deterministic_eval(e[2],env) if deterministic_eval(e[1],env) else deterministic_eval(e[3],env)
    elif type(e) == list:
        c = [deterministic_eval(e[i],env) for i in range(len(e))]
        return c[0](c[1:])


# I am lazy and just paraphrased some generic topological sort algorithm from the internets (https://www.educative.io/collection/page/6151088528949248/4547996664463360/5657624383062016)
def helperFunction(V, A, v, seen, order) :
    seen[v] = True
    for i in A[v] :
        if seen[i] == False :
            helperFunction(V, A, i, seen, order)
    order.insert(0, v)

def topsort(V, A):
    order = []
    seen = {v:False for v in V}
    for v in V :
        if seen[v] == False :
            helperFunction(V, A, v, seen, order)
    return order
        
def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    V, A, P, Y = graph[1]['V'], graph[1]['A'], graph[1]['P'], graph[1]['Y']
    env = {}
    for v in V:
        if v not in A.keys():
            A[v] = []
    order = topsort(V,A)
    for v in order:
        if P[v][0] == 'sample*':
            sample = deterministic_eval(P[v][1], env).sample()
            env[v] = sample
        elif P[v][0] == 'observe*':
            observation = Y[v]
            env[v] = torch.tensor(observation)
    return deterministic_eval(graph[-1],env)
    
    
# this is terrible and inefficient, think of a better way
def markov_blankie(V, A):
    for v in V:
        if v not in A.keys():
            A[v] = []
    blanket = {v:[c for c in A[v]] for v in V} # deals with the children
    for v in V:
        for c in A[v]:
            if v not in blanket[c]:
                blanket[c].append(v) # deals with parents
    for v in V:
        for a in A[v]:
            for v_prime in V:
                if v != v_prime and a in A[v_prime] and v_prime not in blanket[v]:
                    blanket[v].append(v_prime) # parents of children (that are not you)
    return blanket
        
def accept(x, chi, chi_prime, Y_cal, P_cal, blanket):
    current_state = {**chi, **Y_cal}
    proposed_state = {**chi_prime, **Y_cal}
    d = deterministic_eval(P_cal[x][1], current_state)
    d_prime = deterministic_eval(P_cal[x][1], proposed_state)
    log_alpha = d_prime.observe(chi[x]) - d.observe(chi_prime[x])
    for v in (blanket[x] + [x]):
        log_alpha = log_alpha + deterministic_eval(P_cal[v][1], proposed_state).observe(proposed_state[v])
        log_alpha = log_alpha - deterministic_eval(P_cal[v][1], current_state).observe(current_state[v])
    return torch.exp(log_alpha)
    
        
burn = 0
    
def metropolis_hastings_in_gibbs(graph, L):
    V, A, P_cal, Y_cal = graph[1]['V'], graph[1]['A'], graph[1]['P'], graph[1]['Y']
    Y_cal = {k:torch.tensor(v).float() for k,v in Y_cal.items()}
    X = [v for v in V if P_cal[v][0] == 'sample*']
    Y = [v for v in V if P_cal[v][0] == 'observe*']
    blanket = markov_blankie(V, A)
    samples = []
    chi = {x:s for x,s in zip(X,sample_from_joint(graph[0:2] + [['vector'] + X]))}
    for i in tqdm(range(L)):
        for x in X:
            d = deterministic_eval(P_cal[x][1], {**chi, **Y_cal})
            chi_prime = copy.deepcopy(chi)
            chi_prime[x] = d.sample()
            alpha = accept(x, chi, chi_prime, Y_cal, P_cal, blanket)
            u = torch.distributions.Uniform(0,1).sample()
            if u < alpha:
                chi = chi_prime
        samples.append(({**chi,**Y_cal},deterministic_eval(graph[-1], {**chi, **Y_cal})))
    return samples

    

def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)











#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))

        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))

        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        next(stream)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1,6):
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))
        test_program(get_stream(graph), i, 'graphical')

    
