from config import np
import matplotlib.pyplot as plt
from data import ptb, sequence
from model import RNNLMGen, BetterRnnlmGen, Seq2Seq, PeekySeq2Seq
from common.trainer import Trainer
from common.optimizer import Adam
from utils.eval_util import eval_seq2seq


def main(operation):
    assert operation in [1, 2, 3], ("\noperation\n"
                                    "\t1: check addition data\n"
                                    "\t2: generate sentence"
                                    "\t3: seq2seq")

    if operation == 1: check()
    elif operation == 2: generate()
    elif operation == 3: seq2seq()


def seq2seq(vanila=False):
    # dataset
    (x_train, t_train), (x_val, t_val) = sequence.load_data('addition.txt', seed=1984)
    char_to_id, id_to_char = sequence.get_vocab()
    vocab_size = len(char_to_id)
    is_reverse = False
    if not vanila:
        is_reverse = True
        x_train, x_val = x_train[:, ::-1], x_val[:, ::-1]

    # model hyperparameters
    wordvec_size = 16
    hidden_size = 128

    # model
    if vanila:
        model = Seq2Seq(vocab_size, wordvec_size, hidden_size)
    else:
        model = PeekySeq2Seq(vocab_size, wordvec_size, hidden_size)

    # training hyperparameters
    batch_size = 128
    max_epoch = 25
    lr = 1e-3
    max_grad = 5.0

    # trainer
    optimizer = Adam(lr)
    trainer = Trainer(model, optimizer)
    """
    문장(짧은 시계열)이 data points이며 요 단위로 처리된다.
    따라서 이러한 data points를 담은 batch를 만들면 되니까 일반적인 Trainer의 작동이다. (TimeTrainer 불필요)
    """

    acc_list = []
    for epoch in range(max_epoch):
        trainer.fit(
            x=x_train, t=t_train,
            max_epoch=1, batch_size=batch_size, max_grad=max_grad
        )

        # epoch마다 10개씩 validation data를 뽑아 올바르게 답했는지를 채점
        correct_num = 0
        for i in range(len(x_val)):
            question, correct = x_val[[i]], t_val[[i]]      # []가 아닌 [[]]로 뽑아 batch form N=1 유지, np.array([[IDs]])
            # verbose = i < 10
            verbose = False
            correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose, is_reverse)

        acc = float(correct_num) / len(x_val)
        acc_list.append(acc)
        print(f'Evaluation Accuracy: {(acc * 100):.3f}%')

    x = np.arange(len(acc_list))
    plt.plot(x, acc_list, marker='o')
    plt.xlabel('에폭')
    plt.ylabel('정확도')
    plt.ylim(0, 1.0)
    plt.show()


def generate(model_type=2):
    assert model_type in [1, 2], ("\nmodel type\n"
                                  "\t1: Vanilla Rnnlm \n"
                                  "\t2: BetterRnnlm")

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    # corpus_size = len(corpus)

    if model_type == 1:
        model = RNNLMGen(vocab_size=vocab_size, wordvec_size=100, hidden_size=100)
        model.load_params(file_name='_Rnnlm')
    elif model_type == 2:
        model = BetterRnnlmGen(vocab_size=vocab_size, wordvec_size=650, hidden_size=650, dropout_ratio=0.5)
        model.load_params(file_name='_BetterRnnlm')

    start_word = 'you'
    start_id = word_to_id[start_word]
    skip_words = ['N', '<unk>', '$']
    skip_ids = [word_to_id[w] for w in skip_words]

    # generate sentence
    sample_size = 100
    word_ids = model.generate(start_id, skip_ids, sample_size)
    txt = (' '.join([id_to_word[i] for i in word_ids])).replace(' <eos>', '.\n')
    txt = '\n'.join([line.strip().capitalize() for line in txt.split('\n')])
    """
    줄바꿈 문자를 기준으로 문자열을 나누고
    strip()으로 줄 앞뒤의 공백 제거 후 capitalize()로 문자열의 첫 글자를 대문자로 바꾸고
    나눴던 문자열을 다시 결합한다.
    """
    print(txt)


def check():
    (x_train, t_train), (x_val, t_val) = sequence.load_data('addition.txt', seed=1984)
    char_to_id, id_to_char = sequence.get_vocab()
    breakpoint()

    print('x_train shape:', x_train.shape, 't_train shape:', t_train.shape)
    print('x_val shape:', x_val.shape, 't_val shape:', t_val.shape)
    breakpoint()

    print('x_train[0]:', x_train[0])
    print('t_train[0]', t_train[0])
    breakpoint()

    print(''.join([id_to_char[c] for c in x_train[0]]))
    print(''.join([id_to_char[c] for c in t_train[0]]))
    breakpoint()


if __name__ == "__main__":
    main(3)