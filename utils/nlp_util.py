import numpy as np
import collections
from tqdm import tqdm
# from config import np
from config import GPU
from utils import normalize
"""
cupy를 쓰면
    ppmi()의 이중 for loop가 ptb dataset처럼 큰 말뭉치에 대해서 매우 느려진다.
    analogy()의 np.dot() 연산에서 .get()으로 값을 꺼내오라며 에러가 발생한다.
"""


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


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    """
    king - man + woman = queen
    """
    for word in (a, b, c):
        if word not in word_to_id:
            print('Cannot find %s in corpus.' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)     # query 벡터와 word matrix의 각 행(단어 벡터)들과의 유사도를 내적으로 get

    # 정답도 제공했다면 정답과의 유사도 get
    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    # 유사도를 기준으로 정렬
    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


class UnigramSampler:
    """
    다중 분류를 이진 분류로 전환하려면 정답을 정답으로 분류할 뿐 아니라 오답을 오답으로 분류하도록 param을 학습시켜야 한다.
    따라서 말뭉치의 개별(uni) 단어의 확률 분포에 따라 오답을 sampling하는 negative sampling이 필요하다.
    여기에는 희소한 단어보다 흔하게 등장하는 단어를 잘 처리하는 방향으로 model을 학습시키겠다는 의도가 전제된다.
    이때 희소한 단어의 출현 확률을 조금이나마 높여주기 위해 특정한 값을 제곱해 확률 분포를 약간 수정한다.
    """
    def __init__(self, tokenized_corpus, sample_size, power=0.75):
        """
        Args:
            tokenized_corpus: [NN,]
        """
        self.sample_size = sample_size  # negative sampling size
        self.vocab_size = None
        self.word_dist = None              # 단어 확률 분포

        counts = collections.Counter()
        for word_id in tokenized_corpus:
            counts[word_id] += 1
        # alternative 1.
        # counts = collections.Counter(tokenized_corpus)
        # alternative 2. without Counter class
        # counts = {}
        # for word_id in corpus:
        #     if word_id not in counts:
        #         counts[word_id] = 0
        #     counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_dist = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_dist[i] = counts[i]

        self.word_dist = np.power(self.word_dist, power)
        self.word_dist /= np.sum(self.word_dist)

    def negative_sample(self, target):
        """
        Args:
            target: target의 단어 IDs (indices) | [N,]
        ---
        Returns:
            negative_sample: 각 target에 대한 negative 단어 IDs (indices) | [N,sample_size]
        """
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
            for i in range(batch_size):
                word_dist = self.word_dist.copy()
                target_idx = target[i]
                word_dist[target_idx] = 0           # 정답은 빼고
                word_dist /= word_dist.sum()        # 다시 합이 1이 되도록 dist 조정
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size,
                                                         replace=False, p=word_dist)
                """cf. ndarray의 copy method
                ndarray의 copy()는 shallow copy지만 데이터 자체를 새로운 메모리 공간에 복사한다.
                ndarray가 연속된 메모리 블록에 데이터를 저장하는 (특별한) 구조이기 때문이다.
                따라서 word_dist를 변경해도 self.word_dist는 바뀌지 않는다.
                """
        else:
            # cupy를 사용한다면 속도를 우선한다.
            # negative sample에 true target이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_dist)

        return negative_sample
