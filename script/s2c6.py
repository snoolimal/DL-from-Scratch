from data import ptb
from model import Rnnlm
from common.trainer import TimeTrainer
from common.optimizer import SGD
from utils.eval_util import eval_perplexity


def main():
    # dataset
    corpus, word_to_id, _ = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    # model hyperparameters
    wordvec_size = 100
    hidden_size = 100
    seq_len = 35

    # model
    model = Rnnlm(vocab_size, wordvec_size, hidden_size)

    # training hyperparams
    batch_size = 20
    max_epoch = 4
    lr = 20.0
    max_grad = 0.25

    # trainer
    optimizer = SGD(lr)
    trainer = TimeTrainer(model, optimizer)

    # training
    trainer.fit(
        x=xs, t=ts,
        seq_len=seq_len, max_epoch=max_epoch, batch_size=batch_size, max_grad=max_grad
    )
    trainer.plot(ylim=(0, 500))

    # evaluation
    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test, batch_size, seq_len)
    print('Test perplexity:', ppl_test)

    model.save_params()


if __name__ == '__main__':
    main()