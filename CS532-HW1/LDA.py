from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from joint_log_lik import joint_log_lik
from sample_topic_assignment import sample_topic_assignment


bagofwords = loadmat('bagofwords_nips.mat')
WS = bagofwords['WS'][0] - 1  #go to 0 indexed
DS = bagofwords['DS'][0] - 1

WO = loadmat('words_nips.mat')['WO'][:,0]
titles = loadmat('titles_nips.mat')['titles'][:,0]

#print("\n".join([w[0] for w in WO]))

#This script outlines how you might create a MCMC sampler for the LDA model

alphabet_size = WO.size

document_assignment = DS
words = WS

#subset data, EDIT THIS PART ONCE YOU ARE CONFIDENT THE MODEL IS WORKING
#PROPERLY IN ORDER TO USE THE ENTIRE DATA SET
#words = words[document_assignment < 100]
#document_assignment  = document_assignment[document_assignment < 100]

n_docs = document_assignment.max() + 1

#number of topics
n_topics = 20

#initial topic assigments
topic_assignment = np.random.randint(n_topics, size=words.size)


#within document count of topics
doc_counts = np.zeros((n_docs,n_topics))

for d in range(n_docs):
    #histogram counts the number of occurences in a certain defined bin
    doc_counts[d] = np.histogram(topic_assignment[document_assignment == d], bins=n_topics, range=(-0.5,n_topics-0.5))[0]

#doc_N: array of size n_docs count of total words in each document, minus 1
doc_N = doc_counts.sum(axis=1) - 1

#within topic count of words
topic_counts = np.zeros((n_topics,alphabet_size))

for k in range(n_topics):
    w_k = words[topic_assignment == k]
    topic_counts[k] = np.histogram(w_k, bins=alphabet_size, range=(-0.5,alphabet_size-0.5))[0]


#topic_N: array of size n_topics count of total words assigned to each topic
topic_N = topic_counts.sum(axis=1)

#prior parameters, alpha parameterizes the dirichlet to regularize the
#document specific distributions over topics and gamma parameterizes the 
#dirichlet to regularize the topic specific distributions over words.
#These parameters are both scalars and really we use alpha * ones() to
#parameterize each dirichlet distribution. Iters will set the number of
#times your sampler will iterate.
alpha = 0.1*np.ones(n_topics)
gamma = 0.001*np.ones(alphabet_size)
iters = 150


jll = []
bar = tqdm(range(iters), total=iters)
best_assignments, best_topic_counts, best_doc_counts, best_topic_N = None,None,None,None
best_jll = -1*np.inf
for i in bar:
    jll.append(joint_log_lik(doc_counts,topic_counts,alpha,gamma))
    if jll[i] > best_jll:
        best_jll = jll[i]
        best_assignments = topic_assignment
        best_topic_counts = topic_counts
        best_doc_counts = doc_counts
        best_topic_N = topic_N
    bar.set_postfix({'jll':jll[i]})
    prm = np.random.permutation(words.shape[0])
    
    words = words[prm]
    document_assignment = document_assignment[prm]
    topic_assignment = topic_assignment[prm]
    
    topic_assignment, topic_counts, doc_counts, topic_N = sample_topic_assignment(
                                topic_assignment,
                                topic_counts,
                                doc_counts,
                                topic_N,
                                doc_N,
                                alpha,
                                gamma,
                                words,
                                document_assignment)
#    fstr = ''
#    for i in range(n_topics):
#        print(topic_counts[i,:].shape)
#        t = WO[np.argsort(-1*topic_counts[i,:])[0:20]]
#        fstr += f'Topic {i}: {" ".join([w[0] for w in t])}\n'
#    print(fstr)


jll.append(joint_log_lik(doc_counts,topic_counts,alpha,gamma))

plt.plot(np.exp(jll)[-50:])
plt.savefig('jll.png')
#plt.show()

### find the 10 most probable words of the 20 topics:

fstr = ''
for i in range(n_topics):
    words = WO[np.argsort(-1*best_topic_counts[i,:])[0:20]]
    fstr += f'Topic {i}: {" ".join([w[0] for w in words])}\n'

with open('most_probable_words_per_topic','w') as f:
    f.write(fstr)
        
#most similar documents to document 0 by cosine similarity over topic distribution:
#normalize topics per document and dot product:
n_sim = 10

fstr = f'{n_sim} most similar documents to document 0: {titles[0][0]} are:\n'
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

sim_docs = np.argsort(-1*np.dot(best_doc_counts[:,0].transpose()/np.linalg.norm(best_doc_counts[:,0]), doc_counts/np.linalg.norm(best_doc_counts, axis=0)))[1:n_sim+1]
for i in range(n_sim):
    fstr += f'{ordinal(i+1)} most similar is document {sim_docs[i]}: {titles[sim_docs[i]][0]}\n'

with open('most_similar_titles_to_0','w') as f:
    f.write(fstr)
    


