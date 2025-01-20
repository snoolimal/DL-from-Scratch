from config import np
from common.functions import matmul, sigmoid, tanh


class LSTM:
    """LSTM cell for Single Timestep t"""
    def __init__(self, wx, wh, b):
        """
        Args:
            wx: [wxf, wxg, wxi, wxo]    | [D,4H]
                x_t와 곱해지는 w[D,H]들이 hstack되어 있다.
            wh: [whf, whg, whi, who]    | [H,4H]
                h_{t-1}과 곱해지는 w[H,H]들이 hstack되어 있다.
            b: [bf, bg, bi, bo]         | [4H,] -> [N,4H]
               각각이 [H,] -> [N,H]로 bdcast되며 이들이 hastck되어 있다.
        """
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(param) for param in self.params]

        self.cache = None   # bpass에 필요한 fpass의 결과롤 보관해 instance var

    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x: x_t              | [N,D]
            h_prev: h_{t-1}     | [N,H]
            c_prev: c_{t-1}     | [N,H]
        ---
        Returns:
            h_next: h_t         | [N,H]
                hidden state 출력은 copy되어
                    하나는 다음 tstep의 LSTM cell로 입력되고 (LSTM cell 내부에서 돌고 도는 작동)
                    다른 하나는 loss 계산을 위해 LSTM cell 외부로 출력
                된다.
            c_next: c_t         | [N,H]
        """
        wx, wh, b = self.params
        N, H = h_prev.shape

        A = matmul(x, wx) + matmul(h_prev, wh) + b  # [N,4H]+[N,4H]+[N,4H]=[N,4H]

        # slice
        f = A[:, :H]        # [N,H]
        g = A[:, H:2*H]     # [N,H]
        i = A[:, 2*H:3*H]   # [N,H]
        o = A[:, 3*H:]      # [N,H]

        # nonlinearity
        f = sigmoid(f)  # [N,H]
        g = tanh(g)     # [N,H]
        i = sigmoid(i)  # [N,H]
        o = sigmoid(o)  # [N,H]

        # output
        c_next = c_prev * f + (g * i)   # ⊙ | [N,H]
        h_next = o * tanh(c_next)       # ⊙ | [N,H]

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)

        return h_next, c_next

    def backward(self, dh_next, dc_next):
        """
        Args:
            dh_next: dh_t       | [N,H]
                다음 tstep의 LSTM cell로 입력된 h_next의 dns grad _dh_next[N,H]와 (LSTM cell 내부에서 돌고 도는 작동)
                loss 계산을 위해 LSTN cell 외부로 출력된 h_next의 dns grad _dh_t[N,H]의 합이다.
                cf. Fpass에서는 copy되므로 _h_next와 _h_t는 같으나 bpass에서 _dh_next와 _dh_t는 당빠 다르다.
            dc_next: dc_t       | [N,H]
        ---
        Returns:
            dx: dx_t            | [N,H]
            dh_prev: dh_{t-1}   | [N,H]
            dc_prev: dc_{t-1}   | [N,H]
        """
        wx, wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        # node: o ⊙ tanh(c_next)
        th = tanh(c_next)  # [N,H]
        do = th * dh_next  # [N,H]⊙[N,H]=[N,H]
        dth = o * dh_next  # [N,H]⊙[N,H]=[N,H]

        # node: tanh(c_next)
        _dc_next = (1 - th ** 2) * dth    # [N,H]

        # node: + of memory cell
        ds = dc_next + _dc_next     # [N,H]+[N,H]=[N,H]

        # node: c_prev ⊙ f
        dc_prev = f * ds            # [N,H]⊙[N,H]=[N,H]
        df = c_prev * ds            # [N,H]⊙[N,H]=[N,H]

        # node: g ⊙ i
        dg = i * ds                 # [N,H]⊙[N,H]=[N,H]
        di = g * ds                 # [N,H]⊙[N,H]=[N,H]

        # activations
        df *= f * (1 - f)           # sigmoid | [N,H]
        dg *= (1 - g ** 2)          # tanh | [N,H]
        di *= i * (1 - i)           # sigmoid | [N,H]
        do *= o * (1 - o)           # sigmoid | [N,H]

        # slice node
        # 대응되는 위치에 고대로 다시 넣어 준다.
        dA = np.hstack((df, dg, di, do))    # [N,H] 4개의 hstack -> [N,4H]

        # node: x * w_x + h_prev * wh + b = A
        dwx = np.dot(x.T, dA)       # [D,N]*[N,4H]=[D,4H]
        dwh = np.dot(h_prev.T, dA)  # [H,N]*[N,4H]=[H,4H]
        db = dA.sum(axis=0)         # [N,4H] -> [4H,]

        # downstream gradient towards optimizer
        self.grads[0][...] = dwx
        self.grads[1][...] = dwh
        self.grads[2][...] = db

        # downstream gradient as upstream gradient
        dx = np.dot(dA, wx.T)       # [N,4H]*[4H,D]=[N,D]
        dh_prev = np.dot(dA, wh.T)  # [N,4H]*[4H,H]=[N,H]
        # dc_prev                   # [N,H]

        return dx, dh_prev, dc_prev


class TimeLSTM:
    """LSTM cell for sequence of length T
    T개의 LSTM cell의 묶음이다.
    """
    def __init__(self, wx, wh, b, stateful=False):
        """
        Args:
            T개의 LSTM cell의 처리를 묶은 것일 뿐이므로 얘네들은 바뀌지 않지.
            wx: [wxf, wxg, wxi, wxo]    | [D,4H]
                x_t와 곱해지는 w[D,H]들이 hstack되어 있다.
            wh: [whf, whg, whi, who]    | [H,4H]
                h_{t-1}과 곱해지는 w[H,H]들이 hstack되어 있다.
            b: [bf, bg, bi, bo]         | [4H,] -> [N,4H]
               각각이 [H,] -> [N,H]로 bdcast되며 이들이 hastck되어 있다.
            stateful:
                Truncated BPTT를 수행해도 fpass에서의 (h와 c의) time dependency는 유지되어야 한다.
                Truncation이 없다면 이 option은 불필요하다. 고로 default는 False로 하고, True라면 truncation 수행을 나타낸다.
        """
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(param) for param in self.params]

        self.layers = None
        self.h, self.c = None, None     # (truncation 속) each tstep의 hidden state와 cell state | h[N,H], C[N,H]
        self.dh = None                  # (truncation 속) each tstep의 LSTM cell의 ups grad | [N,H]

        self.stateful = stateful

    def forward(self, xs):
        """
        Args:
            xs: [N,T,D]
        ---
        Returns:
            hs: [N,T,H]
                Copy되어 하나는 다음 LSTM block으로 입력되고 하나는 loss 계산을 위해 LSTM block 외부로 출력된다.
        """
        wx, wh, b = self.params
        N, T, D = xs.shape
        H = wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        # LSTM blocks의 time dependency를 유지하거나 처음으로 LSTM block을 생성했을 때
        # cf. LSTM blocks의 time dep을 유지하지 않을 것이라면 dataloder에서 xs를 꺼내올 때마다 model.reset_state()를 호출하겠지.
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            x_t = xs[:, t, :]   # [N,D]
            self.h, self.c = layer.forward(x_t, self.h, self.c)     # h[N,H], c[N,H]
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        """
        Args:
            dhs: LSTM cell 외부로 출력된 hs의 dns grad    | [N,T,H]
                LSTM cell 내부의 작동인 또 하나의 (copied) hs와 c는 출력되지 않고 내부적으로 계산되므로 외부로부터 관련된 값을 받지 않는다.
        ---
        Returns:
            dxs: [N,T,D]
        """
        wx, wh, b = self.params
        N, T, H = dhs.shape
        D = wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0   # LSTM 내부의 작동   | _dh_next[N,H] dc[N,H]

        grads = [0, 0, 0]   # dwx, dwh, db
        for t in reversed(range(T)):
            layer = self.layers[t]
            dh_next = dhs[:, t, :] + dh
            dc_next = dc
            dx, dh, dc = layer.backward(dh_next, dc_next)   # dx[N,D], dh[N,H], dc[N,H]
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad    # RNN과 마찬가지로 하나의 LSTM cell의 param이 공유되므로 grad는 누적

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None


class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flag = True

    def forward(self, xs):
        if self.train_flag:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            # self.mask = flg.astype(np.float32)
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale  # 이렇게 mask를 지정해야 backward에서 self.mask가 scaling 포함 (성능 차이 많이 나네...)
            # return xs * self.mask * scale
            return xs * self.mask
        else:
            return xs

    def backward(self, dy):
        return self.mask * dy