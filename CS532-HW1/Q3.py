import numpy as np
import random
from tqdm import tqdm
##Q3

random.seed(42)


##first define the probability distributions as defined in the excercise:

#define 0 as false, 1 as true
def p_C(c):
    p = np.array([0.5,0.5])
    return p[c]


def p_S_given_C(s,c):
    p = np.array([[0.5,0.9],[0.5,0.1]])
    return p[s,c]
    
def p_R_given_C(r,c):
    p = np.array([[0.8,0.2],[0.2,0.8]])
    return p[r,c]

def p_W_given_S_R(w,s,r):
    
    p = np.array([
            [[1.0,0.1],[0.1,0.001]],   #w = False
            [[0.0,0.9],[0.9,0.99]],   #w = True
            ])
    return p[w,s,r]


##1. enumeration and conditioning:
    
## compute joint:
p = np.zeros((2,2,2,2)) #c,s,r,w
for c in range(2):
    for s in range(2):
        for r in range(2):
            for w in range(2):
                p[c,s,r,w] = p_C(c)*p_S_given_C(s,c)*p_R_given_C(r,c)*p_W_given_S_R(w,s,r)
                
## condition and marginalize:

p_grass                  = np.sum(p,axis=(0,1,2))[1]
p_cloudy_given_grass     = np.sum(p,axis=(1,2))[1,1]/p_grass
p_not_cloudy_given_grass = np.sum(p,axis=(1,2))[0,1]/p_grass
p_C_given_W              = np.array([p_not_cloudy_given_grass,p_cloudy_given_grass])
print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(p_C_given_W[1]*100))



##2. ancestral sampling and rejection:

def rainy_ancestral():
    cloudy = 1 if (random.random() < 0.5) else 0
    if cloudy:
        sprinkler = 1 if (random.random() < 0.1) else 0
        rain = 1 if (random.random() < 0.8) else 0
    else:
        sprinkler = 1 if (random.random() < 0.5) else 0
        rain = 1 if (random.random() < 0.2) else 0
    if rain and sprinkler:
        return np.array([cloudy,sprinkler,rain, 1 if random.random() < 0.99 else 0])
    elif rain and not sprinkler:
        return np.array([cloudy,sprinkler,rain, 1 if random.random() < 0.90 else 0])
    elif not rain and sprinkler:
        return np.array([cloudy,sprinkler,rain, 1 if random.random() < 0.90 else 0])
    elif not rain and not sprinkler:
        return np.array([cloudy,sprinkler,rain, 1 if random.random() < 0.0 else 0])

num_samples = 10000000
samples = np.zeros(num_samples)
rejections = 0
i = 0
for i in tqdm(range(num_samples)):
    x = rainy_ancestral()
    u = random.uniform(0,p[x[0],x[1],x[2],x[3]])
    if u <= p[x[0],x[1],x[2],x[3]]*x[3]:
        samples[i] = x[0]
        i += 1
    else:
        rejections += 1

print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean()*100))
print('{:.2f}% of the total samples were rejected'.format(100*rejections/(samples.shape[0]+rejections)))


#3: Gibbs
# we can use the joint above to condition on the variables, to create the needed
# conditional distributions:


#we can calculate p(R|C,S,W) and p(S|C,R,W) from the joint, dividing by the right marginal distribution
#indexing is [c,s,r,w]
p_R_given_C_S_W = p/p.sum(axis=2, keepdims=True)
p_S_given_C_R_W = p/p.sum(axis=1, keepdims=True)


# but for C given R,S,W, there is a 0 in the joint (0/0), arising from p(W|S,R)
# but since p(W|S,R) does not depend on C, we can factor it out:
#p(C | R, S) = p(R,S,C)/(int_C (p(R,S,C)))

#first create p(R,S,C):
p_C_S_R = np.zeros((2,2,2)) #c,s,r
for c in range(2):
    for s in range(2):
        for r in range(2):
            p_C_S_R[c,s,r] = p_C(c)*p_S_given_C(s,c)*p_R_given_C(r,c)
            
#then create the conditional distribution:
p_C_given_S_R = p_C_S_R[:,:,:]/p_C_S_R[:,:,:].sum(axis=(0),keepdims=True)


##gibbs sampling
num_samples = 1000000
samples = np.zeros(num_samples)
state = np.zeros(4,dtype='int')
#c,s,r,w, set w = True

state[3] = 1
i = 0
for i in tqdm(range(num_samples)):
    state[0] = np.random.choice([0,1], p=p_C_given_S_R[:, state[1],state[2]])
    state[1] = np.random.choice([0,1], p=p_S_given_C_R_W[state[0],:,state[2],state[3]])
    state[2] = np.random.choice([0,1], p=p_R_given_C_S_W[state[0],state[1],:,state[3]])
    samples[i] = state[0]
    i += 1

print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean()*100))



### pure monte carlo
#
#n = 100000
#clouds = 0
#grass = 0
#for i in range(0,n):
#    samp = rainy_ancestral()
#    if samp[3] and samp[0]:
#        clouds += 1
#    if samp[3]:
#        grass += 1
#print(float(clouds/grass))
