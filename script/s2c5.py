import numpy as np

from data import ptb
from model import SimpelRnnlm
from common.trainer import TimeTrainer
from common.optimizer import SGD


def main():
    # dataset
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    x, t = corpus[:-1], corpus[1:]
    vocab_size = int(max(corpus) + 1)   # 단어 ID는 0번부터니까

    # model hyperparams
    wordvec_size = 100
    hidden_size = 100
    seq_len = 5

    # model
    model = SimpelRnnlm(vocab_size, wordvec_size, hidden_size)

    # training hyperparams
    max_epoch = 100
    batch_size = 10
    lr = 1e-1

    # trainer
    optimizer = SGD(lr)
    trainer = TimeTrainer(model, optimizer)

    # training
    trainer.fit(x, t, seq_len, max_epoch, batch_size)
    trainer.plot()


if __name__ == '__main__':
    main()