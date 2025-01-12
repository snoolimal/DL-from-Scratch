"""
Chapter 4. word2vec 속도 개선
"""

import pickle
from pathlib import Path
from config import np, GPU
from data import ptb
from utils import cupy_util as cuu
from utils.nlp_util import create_context_target, most_similar, analogy
from model import CBOW, SkipGram
from common.trainer import Trainer
from common.optimizer import Adam


def main(phase='eval', model_type=1):
    assert phase in ['train', 'eval'], "Pass either 'train' or 'eval' to phase argument."
    assert model_type in [1, 2], "Select either 1 for CBOW or 2 for SkipGram as model_type."

    root_dir = Path(__file__).resolve().parents[1]
    model_name = 'cbow' if model_type == 1 else 'skipgram'
    file_name = str(root_dir / 'model' / f'_{model_name}_params.pkl')
    # file_name = str(root_dir / 'model' / f'_{model_name}_params_repo.pkl')

    if phase == 'train':
        # hyperparameters
        window_size = 5
        hidden_size = 100   # embedding dimension (단어의 분산 표현 밀집 벡터의 차원)
        batch_size = 128
        max_epoch = 10

        tokenized_corpus, word_to_id, id_to_word = ptb.load_data('train')
        vocab_size = len(word_to_id)

        contexts, target = create_context_target(tokenized_corpus, window_size)
        if GPU:
            contexts, target = cuu.to_gpu(contexts), cuu.to_gpu(target)

        if model_type == 1:
            model = CBOW(tokenized_corpus, vocab_size, hidden_size, window_size)
        elif model_type == 2:
            model = SkipGram(tokenized_corpus, vocab_size, hidden_size, window_size)
        optimizer = Adam()
        trainer = Trainer(model, optimizer)

        trainer.fit(
            x=contexts, t=target,
            max_epoch=max_epoch, batch_size=batch_size
        )
        trainer.plot()

        word_vecs = model.word_vecs
        if GPU:
            word_vecs = cuu.to_cpu(word_vecs)

        # save
        params = dict()
        params['word_vecs'] = word_vecs.astype(np.float16)
        params['word_to_id'] = word_to_id
        params['id_to_word'] = id_to_word
        with open(file_name, 'wb') as f:
            pickle.dump(params, f, -1)

    elif phase == 'eval':
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        word_vecs = params['word_vecs']
        word_to_id = params['word_to_id']
        id_to_word = params['id_to_word']

        queries = ['you', 'year', 'car', 'toyota']
        for query in queries:
            most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

        print('-' * 50)
        analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)
        analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
        analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
        analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)


if __name__ == '__main__':
    # main('train')
    main('eval')