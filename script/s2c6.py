from config import *
from data import ptb
from model import RnnLM, BetterRnnLM
from common.trainer import TimeTrainer
from common.optimizer import SGD
from utils import cupy_util as cuu
from utils.eval_util import eval_perplexity


def main(vanila=False):
    if vanila:
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
        model = RnnLM(vocab_size, wordvec_size, hidden_size)

        # training hyperparameters
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
        print('Test Perplexity:', ppl_test)

        model.save_params()
    else:
        # dataset
        corpus, word_to_id, _ = ptb.load_data('train')
        corpus_val, _, _ = ptb.load_data('val')
        corpus_test, _, _ = ptb.load_data('test')
        if GPU:
            corpus = cuu.to_gpu(corpus)
            corpus_val = cuu.to_gpu(corpus_val)
            corpus_test = cuu.to_gpu(corpus_test)
        vocab_size = len(word_to_id)
        xs, ts = corpus[:-1], corpus[1:]

        # model hyperparameters
        wordvec_size = 650
        hidden_size = 650
        seq_len = 35
        dropout_ratio = 0.5

        # model
        model = BetterRnnLM(vocab_size, wordvec_size, hidden_size, dropout_ratio)

        # train hyperparameters
        batch_size = 20
        max_epoch = 40
        lr = 20.0
        max_grad = 0.25

        # trainer
        optimizer = SGD(lr)
        trainer = TimeTrainer(model, optimizer)

        # train and eval
        best_ppl = float('inf')
        for epoch in range(max_epoch):
            trainer.fit(
                x=xs, t=ts,
                seq_len=seq_len, max_epoch=1, batch_size=batch_size, max_grad=max_grad
            )
            model.reset_state()     # reset for validation
            val_ppl = eval_perplexity(model, corpus_val, batch_size, seq_len)
            print(f'Epoch {epoch + 1} Evaluation Perplexity:', val_ppl)

            if best_ppl > val_ppl:
                best_ppl = val_ppl
                model.save_params()
            else:
                lr /= 4.0
                optimizer.lr = lr

            model.reset_state()     # reset per train batch
            print('-' * 50)

        # test
        model.reset_state()
        test_ppl = eval_perplexity(model, corpus_test, batch_size, seq_len)
        print('Test Perplexity:', test_ppl)


if __name__ == '__main__':
    main()