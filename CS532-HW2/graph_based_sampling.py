import torch
import torch.distributions as dist
import copy

from daphne import daphne

from primitives import primitives, distributions
from evaluation_based_sampling import evaluate_program, test_program
from tests import is_tol, run_prob_test,load_truth


# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
#env = {'normal': dist.Normal,
#       'sqrt': torch.sqrt}


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        return evaluate_program([exp])
    elif type(exp) in [int, float]:
        return torch.tensor(float(exp))
    elif torch.is_tensor(exp):
        return exp
    else:
        raise("Expression type unknown.", exp)

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
        
def propegate(e, name, sample):
    if type(e) == list:
        for i in range(0,len(e)):
            e[i] = propegate(e[i], name, sample)
    else:
        if e == name:
            return sample
    return e

def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    eval_graph = copy.deepcopy(graph) # otherwise propegation fixes the origional graph which is bad
        
    V, A, P, Y = eval_graph[1]['V'], eval_graph[1]['A'], eval_graph[1]['P'], eval_graph[1]['Y']
    for v in V:
        if v not in A.keys():
            A[v] = []
    
    order = topsort(V,A)
    for v in order:
        if P[v][0] == 'sample*':
            # we assume that all dependencies have been populated
            sample = deterministic_eval(P[v][1]).sample()
            
            eval_graph[-1] = propegate(eval_graph[-1], v, sample)
            # populate any dependencies
            for kid in A[v]:
                P[kid] = propegate(P[kid], v, sample)
        
        elif P[v][0] == 'observe*':
            # TODO: replace with real observe
            fake_sample = Y[v]
            eval_graph[-1] = propegate(eval_graph[-1], v, sample)
            # populate any dependencies
            for kid in A[v]:
                P[kid] = propegate(P[kid], v, sample)
        else:
            raise NotImplementedError()
    return deterministic_eval(eval_graph[-1])
    
    

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

    for i in range(1,5):
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))
        test_program(get_stream(graph), i, 'graphical')

    
