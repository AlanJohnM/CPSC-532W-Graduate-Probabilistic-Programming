import numpy as np
import scipy.special as sp

def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
    """
    Calculate the joint log likelihood of the model
    
    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.
    Returns:
        ll: the joint log likelihood of the model
    """
    n_docs, n_topics = np.shape(doc_counts)
    _, alphabet_size = np.shape(topic_counts)
    
    add = (n_docs*sp.gammaln(np.sum(alpha)) +
           np.sum(sp.gammaln(doc_counts + alpha.reshape((1,n_topics)))) +
           n_topics*sp.gammaln(np.sum(gamma)) +
           np.sum(sp.gammaln(topic_counts + gamma.reshape((1,alphabet_size)))))
    sub = (n_docs*np.sum(sp.gammaln(alpha)) +
          np.sum(sp.gammaln(np.sum(doc_counts + alpha.reshape((1,n_topics)), axis=1))) +
          n_topics*np.sum(sp.gammaln(gamma)) +
          np.sum(sp.gammaln(np.sum(topic_counts + gamma.reshape((1,alphabet_size)), axis=1))))
    return add - sub
    
    
    
    
    
    #add all terms and do subtraction last to minimize cancellation error
    
#    n_docs*sp.gammaln(np.sum(alpha)) # +
#    n_docs*np.sum(sp.gammaln(alpha)) # -
#    np.sum(sp.gammaln(doc_counts + alpha.reshape((1,n_topics)))) # +
#    np.sum(sp.gammaln(np.sum(doc_counts + alpha.reshape((1,n_topics)), axis=1))) # -
#
#    n_topics*sp.gammaln(np.sum(gamma)) # +
#    n_topics*np.sum(sp.gammaln(gamma))) # -
#    np.sum(sp.gammaln(topic_counts + gamma.reshape((1,alphabet_size)))) # +
#    np.sum(sp.gammaln(np.sum(topic_counts + gamma.reshape((1,alphabet_size)), axis=1))) # -
#
