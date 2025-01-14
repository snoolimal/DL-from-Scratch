from typing import Optional, List
from config import np
from common.layers.recurrent._time import RNN


class TimeRNN:
    """
    Unit data가 길이 T의 sequence인 batch를 처리한다.
    Truncated BPTT의 방식으로 학습하려면 일정한 개수의 RNN cell을 모아야 한다.
    RNN node T개를 연결하여 T개 timesteps에서의 작동을 처리한다.
    ---
    B: raw batch size
    T: sequence length
        T개의 RNN cell이 모여 RNN block을 구성한다.
    N: batch size = B / T
    H: hidden state h의 dimension
        h: [N,H]
    D: single (timestep의) input vector x의 dimension
        x: [N,D]
    ---
    e.g. sequence = (x0, ..., x15)
    B = 8
        raw batch 1: (x0, ..., x7)
        raw batch 2: (x8, ..., x15)
    T = 4 -> N = 2
        batch 1: ((x0, x1, x2, x3),
                  (x8, x9, x10, x11))
        batch 2: ((x4, x5, x6, x7),
                  (x12, x13, x14, x15))
        cf. 길이 T의 sequence가 unit data이므로 N은 batch size가 맞다.
    (x0, -> (x1, -> ... (x3, -> | (x4, -> ... (x7,
     x8)     x9)         x11)      x12)        x15)
    순서로 처리된다. (각각을 chunk라 부르자)
    Batch의 unit data는 길이 T의 sequence이며 이들은 "순서대로" 처리되어야 한다.
    따라서 batch 하나의 처리는
        N*T를 batch size로, single input vector를 unit data로 하여 그들을 순서 없이 처리하는 것이 아닌
        0부터 T-1까지의 timestep 각각에 걸리는 input vectors를 unit data로 하여 N번 처리해
    실현한다.
    """
    def __init__(self, wx, wh, b, stateful=False):
        """
        Instance Variables:
            self.layers: RNN block 속 RNN nodes 그릇
            self.h: (RNN block의) hidden state
                RNN block의 최종 출력인 hidden state h를 instance variable로 관리하여
                block 속 RNN node들을 거칠 때마다 그 값을 (저장도 하고) 덮어 쓰면
                RNN block이 끝났을 때 다음 RNN block에 hidden state를 연계하는 작업을 자연스럽게 처리할 수 있다.
            self.dh: 마지막 dh_prev
        ---
        Args:
            wx: [D,H]
            wh: [H,H]
            b: [H,] -> [N,H] (broadcasting, N개 copy)
            stateful: hidden state를 인계 받을 것인가?
                이전 timestep -- 여기서는 timeblock -- 의 hidden state를 유지할 것인지 결정한다.
                    True: hidden state를 유지하므로 아무리 긴 sequence data라도 fpass는 끊어지지 않는다.
                    False: hidden state를 zero matrix로 initialize한다.
        """
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(param) for param in self.params]
        self.layers: Optional[List] = None

        self.stateful = stateful
        self.h = None
        self.dh = None

    def forward(self, xs):
        """
        Args:
            xs: batch sequence (default batch form) | [N,T,D]
                xs[:, t, :]: chunk x_t | [N,D]
                cf. single sequence form: [T,D]
        ---
        Returns:
            hs: batch hidden state (keep time dimension)    | [N,T,H]
                hs[:, t, :]: chunk hidden state h_t         | [N,H]
                hs는 각 timestep의 hidden state h_t를 담고 있다.
        """
        wx, wh, b = self.params
        N, T, D = xs.shape
        D, H = wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        # hidden state를 유지하지 않거나 RNN block을 생성 시
        if not self.stateful or self.h is None:
            # hidden state를 zero matrix로 initialize
            self.h = np.zeros((N, H), dtype='f')

        # across the sequence로 timestep 하나씩 처리
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)     # xs[:, t, :]: x_t  | [N,D]

            hs[:, t, :] = self.h                            # hs[:, t, :]: h_t  | [N,H]
            self.layers.append(layer)
            """self.h의 연계
            위 for loop를 다 돌면 -- i.e., sequence를 timestep 순으로 전부 처리하면 -- 
            self.h에는 마지막 timestep에서의 RNN node의 hidden state가 저장된다.
            따라서 다음 번 forward() 호출에서는 stateful이
                True면 그러한 self.h가 사용
                False면 다시 self.h가 zero matrix로 초기화
            된다. 
            """

        return hs

    def backward(self, dhs):
        """
        Args:
            dhs: [N,T,H]
        ---
        Returns:

        """
        wx, wh, b = self.params
        N, T, H = dhs.shape
        D, H = wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0              # truncated BPTT라면 첫 dh_next는 0
        grads = [0, 0, 0]   # dwx, dwh, db
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx_t, dh = layer.backward(dhs[:, t, :] + dh)    # dx_t, dh_prev = layer.backward(dh_t + dh_next)
            dxs[:, t, :] = dx_t
            """
            Fpass에서 h_t는
                h_t: affine node로 들어 가는 hidden state
                h_next: 다음 RNN node로 들어 가는 hidden state와
            로 분기(copy)된다.
            따라서 bpass에서는 -- i.e., backward()에서는 -- 둘의 gradient:
                dh_t = dhs[:, t, :]
                dh_next = dh
            을 합산하여 downstream gradient:
                dh_prev = dh
            를 얻는다.
            """

            # RNN block의 downstream gradient를 update
            for i, grad in enumerate(layer.grads):  # timestep t의 RNN node에서 얻은 dwx, dwh, db
                grads[i] += grad                    # 모든 RNN node는 동일한 하나의 RNN node, param은 재사용
                """
                grads는 RNN block의 downstream gradient, layer.grads는 RNN node의 downstream gradient이다.
                RNN node에서
                ```
                self.grads[0][...] = dwx
                self.grads[1][...] = dwh
                self.grads[2][...] = db
                ```
                로 저장한 downstream gradient를 꺼내온다. (self = layer = RNN)
                """

        # RNN block의 최종 downstream gradient towards optimizer를 저장
        for i, grad in enumerate(grads):    # RNN block의 dwx, dwh, db
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None
