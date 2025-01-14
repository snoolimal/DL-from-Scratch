from config import np


class RNN:
    """
    Unit data가 single timestep t의 vector인 chunk x_t를 처리한다.
        cf. Batch form이 default이므로 x_t를 single timestep의 input vector가 아닌 그들이 모인 chunk로 한다.
    ---
    N: batch size
    H: hidden state h의 dimension
        h: [N,H]
    D: single (timestep의) input vector x의 dimension
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
