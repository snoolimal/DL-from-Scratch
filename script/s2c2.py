from data import ptb
from utils.nlp_util import *


def check_utils():
    text = 'You say goodbye and I say hello.'
    tokenized_corpus, word_to_id, id_to_word = tokenize_corpus(text)
    C = create_co_matrix(tokenized_corpus)

    c0 = C[word_to_id['you']]
    c1 = C[word_to_id['i']]
    print(cos_similarity(c0, c1))

    query = 'you'
    most_similar(query, word_to_id, id_to_word, C)

    W = ppmi(C)

    print('Co-occurence Matrix')
    print(C)
    print('-' * 50)
    print('PPMI')
    print(W.round(3))

    # breakpoint()


def check_ptb():
    window_size = 2
    wordvec_size = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    C = create_co_matrix(corpus, window_size)
    W = ppmi(C)

    print('Calculating SVD ...')
    try:
        # truncated SVD (빠르다!)
        from sklearn.utils.extmath import randomized_svd
    except ImportError:
        # 그냥 SVD (느리다)
        def randomized_svd(W, **kwargs):
            U, S, V = np.linalg.svd(W)

            return U, S, V
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                             random_state=None)

    word_vecs = U[:, :wordvec_size]

    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

    # breakpoint()


if __name__ == "__main__":
    # check_utils()
    check_ptb()
