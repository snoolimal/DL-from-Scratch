"""
Chapter 3. word2vec
"""

from model import SimpleCBOW, SimpleSkipGram
from common.optimizer import Adam
from common.trainer import Trainer
from utils.nlp_util import create_one_hot_context_target, tokenize_corpus


def main(model_type=1):
    assert model_type in [1, 2], "Select among 1 for SimpleCBOW or 2 for SimpleSkipGram as model_type."
    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    corpus = 'You say goodbye and I say hello.'
    contexts, target, vocab_size = create_one_hot_context_target(corpus, window_size)

    if model_type == 1:
        print('Model: SimpleCBOW')
        # model = SimpleCBOW(vocab_size, hidden_size)
        model = SimpleCBOW(vocab_size, hidden_size, window_size)
    elif model_type == 2:
        print('Model: SimpleSkipGram')
        model = SimpleSkipGram(vocab_size, hidden_size, window_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)
    trainer.fit(
        x=contexts, t=target,
        max_epoch=max_epoch, batch_size=batch_size
    )
    trainer.plot()

    _, word_to_id, id_to_word = tokenize_corpus(corpus)
    word_vecs = model.word_vecs
    for word_id, word in id_to_word.items():
        # print(word, word_vecs[word_id])
        vec = word_vecs[word_id]
        print(f"{word:8} [{', '.join(f'{x:8.4f}' for x in vec)}]")


def preprocess():
    corpus = 'You say goodbye and I say hello.'
    window_size = 1
    contexts, target, vocab_size = create_one_hot_context_target(corpus, window_size)
    print(contexts.shape, target.shape)

    breakpoint()


if __name__ == '__main__':
    # preprocess()
    main(1)
    main(2)