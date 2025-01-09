from config.np import np
from _common.functions.convolution import img2col, col2img


class Convolution:
    def __init__(self, w, b, stride=1, padding=0):
        self.w = w
        self.b = b

        self.stride = stride
        self.padding = padding

        # bpass에 사용
        self.x = None
        self.x_flatten = None
        self.w_flatten = None

        self.dw = None
        self.db = None

    def forward(self, x):
        w, b, stride, padding = self.w, self.b, self.stride, self.padding

        # setting
        out_channels, _, FH, FW = w.shape   # FN: out_channels
        N, _, _H, _W = x.shape
        # assert self.w.shape[1] == self.x.shape[1]
        kernel_size = (FH, FW)
        H = 1 + int((_H + 2 * padding - FH) / stride)
        W = 1 + int((_W + 2 * padding - FW) / stride)

        # fpass
        x_flatten = img2col(x, kernel_size, stride, padding)    # [N*H*W,C*filter_h*filter_w]
        w_flatten = w.reshape(out_channels, -1).T               # [C*filter_h*filter_w,out_channels]
        y = np.dot(x_flatten, w_flatten) + b                    # [N*H*W,out_channels]
        y = y.reshape(N, H, W, -1).transpose(0, 3, 1, 2)        # [N,H,W,out_channels] -> [N,out_channels,H,W]

        self.x = x
        self.x_flatten = x_flatten
        self.w_flatten = w

        return y

    def backward(self, dy):
        x_flatten, w_flatten = self.x_flatten, self.w_flatten
        out_channels, C, FH, FW = self.w.shape                      # FN: out_channels
        kernel_size = (FH, FW)
        dy = dy.transpose(0, 2, 3, 1).reshape(-1, out_channels)     # [N,H,W,out_channels] -> [N*H*W,out_channels]

        self.db = np.sum(dy, axis=0)
        self.dw = np.dot(x_flatten.T, dy)           # [C*filter_h*filter_w,out_channels]
        self.dw = self.dw.transpose(1, 0).\
            reshape(out_channels, C, FH, FW)        # [out_channels,C*filter_h*filter_w] -> [out_channels,C,FH,FW]

        dx_flatten = np.dot(dy, w_flatten.T)
        dx = col2img(dx_flatten, self.x.shape, kernel_size, self.stride, self.padding)

        return dx


class MaxPool:
    def __init__(self, pool_h, pool_w, stride=2, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, _H, _W = x.shape
        pool_h, pool_w, stride, padding = self.pool_h, self.pool_w, self.stride, self.padding
        kernel_size = (pool_h, pool_w)

        H = int(1 + (_H - pool_h) / stride)
        W = int(1 + (_W - pool_w) / stride)

        """
        pooling은 합성곱과 달리 filter의 적용 영역이 image뿐 아니라
        image의 channel에 대해서도 독립적이다.
        e.g.
        [00, 00, 00, 01,  00, 00, 00, 17],  # N=1st에 대해 (pool_h=0,pool_w=0) 시작점의 kernel이 커버하는 값
                                            # 00, 00, 00, 01은 channel 1, 00, 00, 00, 17은 channel 2
        vs.
        [[00, 00, 00, 01],                  # N=1st, channel=1
         [00, 00, 00, 17]                   # N=1st, channel=2
                ...
         [              ]]
        """
        x_flatten = img2col(x, kernel_size, stride, padding)    # [N*H*W,C*pool_h*pool_w]
        x_flatten = x_flatten.reshape(-1, pool_h * pool_w)      # [N*H*W*C,pool_h*pool_w]

        self.x = x
        self.arg_max = np.argmax(x_flatten, axis=1)                 # 행별 argmax | [N*H*W*C,]

        y = np.max(x_flatten, axis=1)                               # 행별 max | [N*H*W*C,]
        y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)     # [N,H,W,C] -> [N,C,H,W]

        return y

    def backward(self, dy):
        x, stride, padding = self.x, self.stride, self.padding
        dy = dy.transpose(0, 2, 3, 1)   # [N,C,H,W] -> [N,H,W,C]

        """
        Max pooling은 각 pixel에 대한 (filter 적용 영역별) max 연산이므로 element-wise 연산이다.
        따라서 local gradient는 filter 영역의 max에서 1이고 나머지는 0인 행렬이며 upstream gradient와 hadmard product한다.
        고로 downstream gradient는 fpass에셔의 max 위치를 기억했다가 upstream gradient 원소를 그곳에 꼽고 나머지는 0으로 채우면 된다.
        ReLU는 모든 element에 대해 같은 max 비교값을 가질 뿐, 사실상 똑같다.
        """
        pool_size = self.pool_h * self.pool_w
        dx = np.zeros((dy.size, pool_size))                                         # [N*H*W*C,pool_h*pool_w]
        dx[np.arange(self.arg_max.size), self.arg_max.flatten()] = dy.flatten()
        dx = dx.reshape(dy.shape + (pool_size,))                                    # [N,H,W,C,pool_h*pool_w]

        dx_flatten = dx.reshape(dx.shape[0] * dx.shape[1] * dx.shape[2], -1)        # [N*H*W,C*pool_h*pool_w]
        dx = col2img(dx_flatten, x.shape, pool_size, stride, padding)               # [N,C,_H,_W]

        return dx