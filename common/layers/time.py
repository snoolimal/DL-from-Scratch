from typing import Optional, List
from config import np
from common.layers import Affine
from common.layers.embedding import Embedding


class RNN:
    """
    Unit data가 single timestep t의 vector인 chunk x_t를 처리한다.
        cf. Batch form이 default이므로 x_t를 single timestep의 input vector가 아닌 그들이 모인 chunk로 한다.
    ---
    N: batch size
    H: hidden state h의 dimension
        h: [N,H]
    D: (single timestep의) input vector x의 dimension
        x: [N,D]
    ---
    e.g. sequence = batch = (x0, ..., x15)
    (x0,
     x_1,
     ...
     x_15)
     를 처리한다.
    Batch form으로 일반화하면 각 chunk:
        (x0, -> (x1, -> ... (x3, -> | (x4, -> ... (x7,
         x8)     x9)         x11)      x12)        x15)
    를 처리한다.
    """
    def __init__(self, wx, wh, b):
        """
        Args:
            wx: [D,H]
                D: x_t와의 행렬곱을 위해 D여야 한다.
                H: h_{t-1}w_h와의 행렬합을 통해 h_t를 만들기 위해 H여아 한다.
            wh: [H,H]
                H: h_{t-1}와의 행렬곱을 위해 H여야 한다.
                H: setting
            b: [H,] -> [N,H] (broadcasting, N개 copy)
        """
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(param) for param in self.params]
        self.cache = None  # for bpass

    def forward(self, x_t, h_prev):
        """
        Args:
            x_t: timestep t의 chunk  | [N,D]
            h_prev: = h_{t-1}       | [N,H]
        ---
        Returns:
            h_next: = h_t           | [N,H]
                h_next는 다음 RNN node로 들어갈 뿐 아니라 copy(분기)되어 affine node로 들어간다.
                여기선 affine node로 들어 가는 h_next는 h_t로 부르자.
        """
        wx, wh, b = self.params
        t = np.matmul(h_prev, wh) + np.matmul(x_t, wx) + b
        h_next = np.tanh(t)

        self.cache = (x_t, h_prev, h_next)

        return h_next

    def backward(self, dh_next):
        """
        RNN node의 입력 wx, wh, b, x, h_prev에 대한 downstream gradient를 계산해 흘리고 보낸다.
        ---
        Args:
            dh_next: dh_t + dh_next (h_t는 copy gate 통과) | [N,H]
                cf. Truncated BPTT라면 bpass의 시작점에서는 dh_t만 있겠지.
        ---
        Returns:
            dx: [N,D]
            dh_prev: [N,H]
        """
        wx, wh, _ = self.params
        x, h_prev, h = self.cache

        # tanh node
        dl = 1 - h ** 2     # element-wise (activation)
        dt = dl * dh_next   # hadmard product

        # bias node
        db = np.sum(dt, axis=0)

        # matmul node 1 (hw_h)
        dwh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, wh.T)

        # matmul node 2 (xw_x)
        dwx = np.matmul(x.T, dt)
        dx = np.matmul(dt, wx.T)

        # downstream gradient towards optimizer
        self.grads[0][...] = dwx
        self.grads[1][...] = dwh
        self.grads[2][...] = db

        # downstream gradient towards previous node
        return dx, dh_prev


class TimeRNN:
    """
    Unit data가 길이 T의 sequence인 batch를 처리한다.
    Truncated BPTT의 방식으로 학습하려면 일정한 개수의 RNN cell을 모아야 한다.
    RNN node T개를 연결하여 T개 timesteps에서의 작동을 처리한다.
    ---
    B: raw batch size
    T: sequence length
        T개의 RNN cell이 모여 RNN block을 구성한다.
    N: batch size = (B + T - 1) // T (B // T + (B % T))
    H: hidden state h의 dimension
        h: [N,H]
    D: single (timestep의) input vector x의 dimension
        x: [N,D]
    ---
    e.g. sequence = (x0, ..., x15)
    B = 8
        raw batch 1: ((x0, ..., x3,
                       x4, ..., x7))
        raw batch 2: ((x8, ..., x11),
                      (x12, ..., x15))
    T = 4 -> N = 2
        batch 1: ((x0, x1, x2, x3),
                  (x8, x9, x10, x11))
        batch 2: ((x4, x5, x6, x7),
                  (x12, x13, x14, x15))
        cf. 길이 T의 sequence가 unit data이므로 N은 batch size가 맞다.
    (x0, -> (x1, -> ... (x3, -> | (x4, -> ... (x7,
     x8)     x9)         x11)      x12)        x15)
    순서로 처리된다. Batch 단위로 한 번에 입력되며, batch 안에서의 처리 순서인 각각을 chunk라 부르자.
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
            dxs: [N,T,D]
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


class _TimeAffine:
    """
    TimeRNN에서 Time류 node를 만든 것과 마찬가지로
    Affine node를 T개 준비하여 (timestep 순서대로) 각 chunk를 처리하면 된다.
    역시나 1개의 Affine node가 재사용된다.
    """
    def __init__(self, w, b):
        """
        Args:
            w: [H,V]
                V: vocab size
            b: [V,] -> [N,V] (broadcasting, N개 copy)
        """
        self.params = [w, b]
        self.grads = [np.zeros_like(param) for param in self.params]

        self.layers: Optional[List] = None

    def forward(self, hs):
        """
        Args:
            hs: [N,T,H]
                cf. h_t: hs[:, t, :] | [N,H]
        ---
        Returns:
            ls: [N,T,V]
                l_t = h_t * w + b | [N,V]
                ls[:, t, :] = l_t
        """
        w, b = self.params
        N, T, H = hs.shape
        H, V = w.shape

        ls = np.empty((N, T, V), dtype='f')
        for t in range(T):
            layer = Affine(w, b)
            ls[:, t, :] = layer.forward(hs[:, t, :])
            self.layers.append(layer)

        return ls

    def backward(self, dls):
        """
        Args:
            dls: [N,T,V]
        ---
        Returns:
            dhs: [N,T,H]
        """
        w, _ = self.params
        N, T, V = dls.shape
        H, V = w.shape

        dhs = np.empty((N, T, H), dtype='f')
        grads = [0, 0]  # dw, db
        for t in range(T):
            layer = self.layers[t]
            dhs[:, t, :] = layer.backward(dls[:, t, :])
            """reversed(range(T))
            Affine node의 적용부터는 time dependency가 사라진 일반적인(병럴적) copy gate를 통과한 w, h가 사용된다.
            따라서 reversed()는 불필요하다.
            """

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        return dhs


class TimeAffine:
    """
    다루는 object들의 shape을 잘 만져서 더 빠르게 작동하도록 효율적으로 구현한다.
    """
    def __init__(self, w, b):
        """
        Args:
            w: [H,V]
                V: vocab size
            b: [V,] -> [N*T,V] (broadcasting, N*T개 copy)
        """
        self.params = [w, b]
        self.grads = [np.zeros_like(param) for param in self.params]

        self.hs = None

    def forward(self, hs):
        """
        Args:
            hs: [N,T,H]
        ---
        Returns:
            ls: [N,T,V]
        """
        self.hs = hs

        w, b = self.params
        N, T, H = hs.shape

        rhs = hs.reshape(N*T, -1)   # [N*T,H]
        rls = np.dot(rhs, w) + b    # np.dot([N*T,H], [H,V]) = [N*T,V]
        ls = rls.reshape(N, T, -1)  # [N,T,V]
        """
        N개를 T번 뽑아 matmul하여 쌓지 말고 N*T개를 그냥 하나의 axis에 행으로 쫙 나열하고 matmul한 후 그룹화하자.
        broadcasting도 같은 값을 N개씩 T번 생성하지 말고 한번에 N*T개 생성하자.
        어차피 결과는 같다.
        Bpass도 요대로 하고 shape 바꾼 hs는 마지막에 reshaping으로 재그룹화해서 다시 원본과 shape만 맞춰 주면 결과는 같겠지.  
        ```
        N, T, H, V = 2, 3, 4, 5
        x = np.arange(24).reshape(N, T, H)
        w = np.arange(20).reshape(H, V)
        b = np.arange(V)
        x.reshape(N*T, H)
        ```
        """

        return ls

    def backward(self, dls):
        """
        Args:
            dls: [N,T,V]
        ---
        Returns:
            dhs: [N,T,H]
        """
        w, _ = self.params
        hs = self.hs
        N, T, H = hs.shape

        drls = dls.reshape(N*T, -1)     # [N*T,V]
        rhs = hs.reshape(N*T, -1)       # [N*T,H]

        db = np.sum(drls, axis=0)       # [V,]
        dw = np.dot(rhs.T, drls)        # [H,N*T] * [N*T,V] = [H,V]
        drhs = np.dot(drls, w.T)        # [N*T,V] * [V,H] = [N*T,H]
        dhs = drhs.reshape(*hs.shape)   # [N*T,H] -> [N,T,H]

        self.grads[0][...] = dw
        self.grads[1][...] = db

        return dhs


class TimeEmbedding:
    def __init__(self, w):
        """
        Args:
            w: [V,D]
        """
        self.params = [w]
        self.grads = [np.zeros_like(w)]

        self.layers = None
        self.w = w

    def forward(self, raw_xs):
        """
        Args:
            raw_xs: batch raw sequence (default batch form) | [N,T]
                raw_xs[:, t]: raw chunk raw_x_t | [N,]
        Returns:
            xs: batch sequence | [N,T,D]
                xs[:, t, :]: chunk x_t | [N,D]
                    raw_x_t가 행의 indices가 되어 w[V,D]로부터 indexing한다.
        """
        N, T = raw_xs.shape
        V, D = self.w.shape

        self.layers = []
        xs = np.empty((N, T, D), dtype='f')

        for t in range(T):
            layer = Embedding(self.w)
            xs[:, t, :] = layer.forward(indices=raw_xs[:, t])
            self.layers.append(layer)

        return xs

    def backward(self, dxs):
        """
        Args:
            dxs: [N,T,D]
                dxs[:, t, :] : [N,D]
        """
        N, T, D = dxs.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dxs[:, t, :])
            grad += layer.grads[0]
            """reversed(range(T))
            Embedding node는 time dependency가 없는 일반적인(병럴적) copy gate를 통과한 w를 사용한다.
            따라서 reversed()는 불필요하다.
            """

        self.grads[0][...] = grad

        return None
