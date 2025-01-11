from tqdm import tqdm
import numpy as np
# from config import np


def tokenize_corpus(corpus):
    """
    말뭉치를 tokenize한다.
    ---
    e.g. 'You say goodbye and I say hello.' -> [0 1 2 3 4 1 5 6]
    ---
    Returns:
        tokenized_corpus: 단어 ID array
        word_to_id: 단어-ID dict
        id_to_word: ID-단어 dict
    """
    corpus = corpus.lower()
    corpus = corpus.replace('.', ' .')
    words = corpus.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    tokenized_corpus = np.array([word_to_id[w] for w in words])

    return tokenized_corpus, word_to_id, id_to_word


def create_one_hot_context_target(corpus, window_size=1):
    """
    cf. fig 3-18 (p.134(135))
    """
    tokenized_corpus, _, _ = tokenize_corpus(corpus)
    contexts, target = create_context_target(tokenized_corpus, window_size)
    vocab_size = len(np.unique(tokenized_corpus))

    contexts = convert_one_hot(tokenized=contexts, vocab_size=vocab_size)
    target = convert_one_hot(tokenized=target, vocab_size=vocab_size)

    return contexts, target, vocab_size


def create_context_target(tokenized_corpus, window_size=1):
    """
    e.g. 'You say goodbye and I say hello.'
    ---
    Args:
        tokenized_corpus: [0, 1, 2, 3, 4, 1, 5, 6] | [N,]
    ---
    Returns:
        contexts: [[0, 2], [1, 3, ..., [1, 6]] | [N-context_window_size,context_window_size]
            cf. context_window_size = 2 * window_size
        target: [1, 2, 3, 4, 1, 5] | [N-context_window_size,]
    """
    target = tokenized_corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(tokenized_corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(tokenized_corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(tokenized, vocab_size):
    """
    Args:
        tokenized: tokenized corpus or tokenized target
        vocab_size V: tokenized의 corpus의 단어 개수
    ---
    Returns:
        one_hot: [tokenized.shape,vocab_size]
    """
    N = tokenized.shape[0]

    # target
    if tokenized.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(tokenized):
            one_hot[idx, word_id] = 1
    # contexts
    elif tokenized.ndim == 2:
        C = tokenized.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(tokenized):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def create_co_matrix(tokenized_corpus, window_size=1):
    """
    Tokenized 말뭉치의 동시발생행렬:
        분포 가설에 기반해 나타낸 단어의 벡터 표현을 행으로 쌓은 행렬로,
        (x,y)th 원소 C(x,y)는 단어 x와 y의 동시발생횟수를 나타낸다.
        cf. 단어 ID가 행 번호가 된다.
    을 생성한다.
    """
    vocab_size = len(np.unique(tokenized_corpus))   # 단어 벡터의 차원은 vocab_size지
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in tqdm(enumerate(tokenized_corpus), total=len(tokenized_corpus),
                             desc='Creating co-occurence matrix...'):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = tokenized_corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < len(tokenized_corpus):
                right_word_id = tokenized_corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def ppmi(C, eps=1e-8):
    """
    동시발생행렬을 기반으로 ppmi를 근사적으로 계산:
        1. S = C(x) ≈ sum_i{C(i,x)}
        2. N ≈ sum_{i,j}C(i,j)
    한다.
    """
    W = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)

    for i in tqdm(range(C.shape[0]), desc='Creating ppmi matrix...', position=0):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            W[i, j] = max(0, pmi)
    # 요 이중 for loop는 cupy로 작동시키면 numpy보다 훨씬 느리네. 왜지?

    return W


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)    # 벡터를 L2-norm으로 정규화
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)                       # 한 후 내적


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 검색어를 말뭉치에서 꺼내
    if query not in word_to_id:
        print('Cannot find %s in corpus.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 꺼낸 단어 벡터와 말뭉치 내 모든 다른 단어 벡터들과의 cos 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # cos 유사도를 기준으로 내림차순 출력
    count = 0
    for i in (-1 * similarity).argsort():
        i = i.item()
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return
