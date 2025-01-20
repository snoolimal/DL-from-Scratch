from data import ptb
from model import RNNLMGen, BetterRnnlmGen


def main(model_type):
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


if __name__ == "__main__":
    main(1)