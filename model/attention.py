from config import np
from common.layers.time import TimeEmbedding, TimeAffine
from common.layers.timegate import TimeLSTM
from common.layers.attention import TimeAttention
from common.layers.criterion import TimeSoftmaxWithLoss
from model import Encoder, Seq2Seq


class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        w_embed = (rn(V, D) / 100).astype('f')
        wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        b = np.zeros(4*H).astype('f')
        wa = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
        ba = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(w_embed)
        self.lstm = TimeLSTM(wx, wh, b, stateful=True)
        self.attention = TimeAttention()    # !
        self.affine = TimeAffine(wa, ba)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, hs_enc):
        h = hs_enc[:, -1]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs_dec = self.lstm.forward(out)
        c = self.attention.forward(hs_enc, hs_dec)  # !
        out = np.concatenate((c, hs_dec), axis=2)   # !
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        ddec_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs

    def generate(self, hs_enc, start_id, sample_size):
        sampled = []
        sample_id = start_id
        h = hs_enc[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            hs_dec = self.lstm.forward(out)
            c = self.attention.forward(hs_enc, hs_dec)
            out = np.concatenate((c, hs_dec), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled


class AttentionSeq2Seq(Seq2Seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        super().__init__(*args)

        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
