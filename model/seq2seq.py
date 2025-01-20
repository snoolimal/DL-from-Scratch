from config import np
from common.base import Model
from common.layers.time import TimeEmbedding, TimeAffine
from common.layers.timegate import TimeLSTM
from common.layers.criterion import TimeSoftmaxWithLoss


class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # weight initialization (Xavier)
        w_embed = (rn(V, D) / 100).astype('f')
        wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        b = np.zeros(4*H).astype('f')

        # model architecture
        self.embed = TimeEmbedding(w_embed)
        self.lstm = TimeLSTM(wx, wh, b)

        # computation node
        self.params = self.embed.params + self.lstm.params  # [[], []]
        self.grads = self.embed.grads + self.lstm.grads     # [[], []]

        # cache
        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)      # stateful=False
        """stateful=False
        여기선 이전 출발어로 만든 TimeLSTM은 hidden state를 유지할 필요가 없으니까.
        전에는 이전 chunk로 만든 hidden state를 유지해야 batch의 time dependency를 유지할 수 있었지.
        즉, 전에는 긴 시계열 데이터가 하나뿐인 -- 정확히는 batch의 개수만큼의 긴 시계열 데이터들을 동시에 처리하는, time dependency는 batch별로
        유지되는 -- 문제를 다뤘기에 그러한 time dependency를 유지하기 위해 chunk 사이의 LSTM layer에서 hidden state를 유지했으나,
        지금은 짧은 시계열 데이터가 여럿인 문제이므로 각 시계열 데이터를 처리할 때마다 LSTM의 hidden state를 0 벡터로 초기화해야 한다.
        """
        self.hs = hs

        return hs[:, -1, :]     # h

    def backward(self, dh):
        """
        Args:
            dh: Decoder가 전해 주는 gradient
        """
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dy = self.lstm.backward(dhs)    # reversed timestep으로 gradient 흘려 나가며 param의 gradient 누적
        dy = self.embed.backward(dy)

        return dy


class Decoder:
    """
    Deterministic한 생성을 사용하자. 따라서 fpass에서 argmax를 사용하므로 softmax는 불필요하다.
    Argmax는 element-wise 연산이다.
    상술했듯 decoder는 학습과 생성에서 Softmax layer를 다르게 취급한다.
    따라서 bpass의 SoftmaxwithLoss layer는 이후의 Seq2Seq class에서 처리하고 여기서는 그 앞까지만 구현하자.
    """

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        w_embed = (rn(V, D) / 100).astype('f')
        wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        b = np.zeros(4*H).astype('f')
        wa = (rn(H, V) / np.sqrt(H)).astype('f')
        ba = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(w_embed)
        self.lstm = TimeLSTM(wx, wh, b, stateful=True)
        self.affine = TimeAffine(wa, ba)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        """
        학습과 생성에서의 fpass가 다르다. 요건 학습에서의 fpass를 수행한다.
        """
        self.lstm.set_state(h)  # encoder의 h 전달 받은 decoder

        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        ys = self.affine.forward(hs)

        return ys

    def backward(self, dy):     # dy: dys
        dy = self.affine.backward(dy)
        dy = self.lstm.backward(dy)
        dh = self.lstm.dh
        _ = self.embed.backward(dy)

        return dh

    def generate(self, h, start_id, sample_size):
        """
        학습과 생성에서의 fpass가 다르다. 요건 생성에서의 fpass를 수행한다.
        """
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)  # encoder의 h 전달받은 decoder

        # 문자를 1개씩 주고 최대 확률을 갖는 문자 ID를 선택하는 작업을 반복한다.
        for _ in range(sample_size):
            x = np.array(sample_id).reshape(1, 1)

            x = self.embed.forward(x)
            h = self.lstm.forward(x)
            y = self.affine.forward(h)

            sample_id = np.argmax(y.flatten())
            sampled.append(int(sample_id))

        return sampled


class Seq2Seq(Model):
    """
    Encoder class와 Decoder class를 연결하고 TimeSoftmaxwithLoss layer로 loss를 계산한다.
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__()

        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.criterion = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]  # decoder_xs의 첫 입력은 구분자

        h = self.encoder.forward(xs)
        ys = self.decoder.forward(decoder_xs, h)
        loss = self.criterion.forward(ys, decoder_ts)

        return loss

    def backward(self, dy=1):
        dy = self.criterion.backward(dy)
        dh = self.decoder.backward(dy)
        dy = self.encoder.backward(dh)

        return dy

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled


class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        w_embed = (rn(V, D) / 100).astype('f')
        wx = (rn(H+D, 4*H) / np.sqrt(D)).astype('f')    # LSTM의 node의 입력에 encoder가 뱉은 h롤 concat해 입력
        wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        b = np.zeros(4*H).astype('f')
        wa = (rn(H+H, V) / np.sqrt(H)).astype('f')      # Affine node의 입력에 encoder가 뱉은 h를 concat해 입력
        ba = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(w_embed)
        self.lstm = TimeLSTM(wx, wh, b, stateful=True)
        self.affine = TimeAffine(wa, ba)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

        self.cache = None

    def forward(self, xs, h):
        """
        0. Input
           xs[N,T], h[N,H]
        1. Embedding
           out[N,T,D]
        2. Repeat
            밑으로(axis=0 방향)으로 repeat
           hs = [n1[H,]
                 n1[H,]
                 ...
                 n1[H,]     # T번
                 n2[H,]
                 ...
                 n2[H,]
                 ...
                 nN[H,]
                 ...
                 nN[H,]] | [N*T,H]
        3. Reshape
           N*T를 T개씩 묶어 N개로 분할해 axis=1을 만든다.
           hs = [[n1,
                  ...
                  n1],
                 [n2,
                  ...
                  n2],
                 ...
                 [nN,
                  ...
                  nN]] | [N,T,H]
        4. Concatenate
           그대로 옆에(axis=2 방향으로) 붙인다. | [N,T,H+D]

        cf. fig 7-26
        """
        N, T = xs.shape
        N, H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs)                    # [N,T,D]
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)   # concat을 위해서 copy gate 통과, T개로 copy | [N,T,H]
        out = np.concatenate((hs, out), axis=2)         # hs와 Embedding의 출력을 연결해 lstm node로 | [N,T,H+D]
        out = self.lstm.forward(out)                    # [N,T,H] ([N,T,H+D]와 [H+D,4H]의 행렬곱)
        out = np.concatenate((hs, out), axis=2)         # [N,T,H+H]
        ys = self.affine.forward(out)                   # [N,T,V]  ([N,T,H+H]와 [H+H,V]의 행렬곱)

        self.cache = H

        return ys

    def backward(self, dscore):
        H = self.cache
        """
        dscore: TimeSoftmaxWithLoss로부터의 dns grad (dys) | [N,T,V]
        dout = self.affine.backward(dscore): TimeAffine의 dns grad로 peeky h와 LSTM 출력의 concat의 grad | [N,T,H+H]
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]: concat 분리, 각각은 LSTM의 ups grad와 TimeAffine에 뿌린 peeky h의 grad
                                                     fpass의 concat은 peeky h에 LSTM의 출력을 붙였으니 요 순서가 맞지.
                                                     | dout[N,T,H], dhs0[N,T,H]
        dout = self.lstm.backward(dout): LSTM이 ups grad dout[N,T,H]를 받아 peeky h와 Embedding 출력의 concat의 grad 뱉음
                                         | dout[N,T,H+D]
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]: concat 분리, 각각은 Embed의 ups grad와 TimeLSTM에 뿌린 peeky h의 grad
                                                       | dembed[N,T,D], dhs1[N,T,H]
        dhs = dhs0 + dhs1: TimeAffine에 뿌린 peeky h의 grad + TimeLSTM에 뿌린 peeky h의 grad
        dh = self.lstm.dh + np.sum(dhs, axis=1): Peeky 없을 때의 encoder의 출력 h의 grad, 그것의 copy의 grad
            lstm.dh[N,H]
            np.sum(dhs, axis=1)을 하면 (axis 0, 2의 분리를 유지한 채) axis=1을 따라 합하니까 [N,H]
                -> 각 Time으로 들어간 copy된 peeky h인 hs0, hs1는 (copy된) h[N,H]가 axis=1을 추가하며 T개로 copy되었으니
                    np.sum(axis=1)로 합.
                -> (3군데로의: 원래 TimeLSTM, peeky TimeAffine, TimeLSTM) peeky copy는 +가 처리.
        """
        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)

        return dh

    def generate(self, h, start_id, sample_size):
        """
        생성의 forward pass (학습의 fpass와 다름)
        ---
        Args:
            h: encoder가 보내 준 hidden state
        """
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)  # encoder의 h를 전달받은 decoder

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)    # batch dim뿐 아니라 copy dim도, [N=1,T=1,H]
        for _ in range(sample_size):
            x = np.array(char_id).reshape(1, 1)     # [N=1,T=1]
            out = self.embed.forward(x)     # [N,T]를 받으므로 위에서 reshape

            # forward()랑 같이 보면 쉽게 이해된다! (똑같은 구조, 마지막만 argmax)
            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            ys = self.affine.forward(out)   # [N=1,T=1,V]

            char_id = np.argmax(ys.flatten())
            sampled.append(char_id)

        return sampled


class PeekySeq2Seq(Seq2Seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__(vocab_size, wordvec_size, hidden_size)

        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads