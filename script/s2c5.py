import numpy as np

from data import ptb
from model import SimpelRnnlm
from common.trainer import TimeTrainer
from common.optimizer import SGD


def main():
    # dataset
    corpus_size = 3000
    corpus, _, _ = ptb.load_data('train')
    corpus = corpus[:corpus_size]
    # corpus = np.arange(17)
    x, t = corpus[:-1], corpus[1:]
    vocab_size = int(max(corpus) + 1)   # 단어 ID는 0번부터니까

    # model hyperparams
    wordvec_size = 100
    hidden_size = 100
    seq_len = 5
    # seq_len = 4

    # model
    model = SimpelRnnlm(vocab_size, wordvec_size, hidden_size)

    # training hyperparams
    batch_size = 10
    # batch_size = 2
    max_epoch = 100
    lr = 1e-1

    # trainer
    optimizer = SGD(lr)
    trainer = TimeTrainer(model, optimizer)

    # training
    trainer.fit(x, t, seq_len, max_epoch, batch_size)
    trainer.plot()
    trainer.plot_grad()


def debug():
    pass


if __name__ == '__main__':
    main()
    # debug()