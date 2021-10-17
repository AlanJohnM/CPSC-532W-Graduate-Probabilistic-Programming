import numpy as np
from tqdm import tqdm

def sample_topic_assignment(topic_assignment,
                            topic_counts,
                            doc_counts,
                            topic_N,
                            doc_N,
                            alpha,
                            gamma,
                            words,
                            document_assignment):
    """
    Sample the topic assignment for each word in the corpus, one at a time.
    
    Args:
        topic_assignment: size n array of topic assignments
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words        
        doc_counts: n_docs x n_topics array of counts per document of unique topics

        topic_N: array of size n_topics count of total words assigned to each topic
        doc_N: array of size n_docs count of total words in each document, minus 1
        
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.

        words: size n array of words
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """
    for word, doc, i in tqdm(zip(words,document_assignment, range(words.size)), total=words.size):
        
        topic = topic_assignment[i]
        topic_counts[topic, word] -= 1
        doc_counts[doc, topic] -= 1
        topic_N[topic] -= 1
        
        topic_dist = (doc_counts[doc,:] + alpha)*(topic_counts[:,word] + gamma[word])/(topic_N + np.sum(gamma))
        topic_dist /= np.sum(topic_dist)
        topic = np.random.choice(topic_N.size, p=topic_dist)

        topic_assignment[i] = topic
        topic_counts[topic, word] += 1
        doc_counts[doc, topic] += 1
        topic_N[topic] += 1
        
    return (topic_assignment, topic_counts, doc_counts, topic_N)

