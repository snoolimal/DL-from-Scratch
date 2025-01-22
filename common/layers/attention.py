from config import np
from common.layers import Softmax


class WeightSum:
    """Decoder Improvement I
    1. Data point의 hs
        hs[T,H]의 행벡터를 그 distribution a로 가중한 후 (가중된 행벡터들을) 그들을 합한다.
        과정은 다음과 같다:
                                                     hs[T,H] ->
            a[T,] -> ar[T,1] -> repeat(H, axis=1) -> ar[T,H] -> ⊙ -> t[T,H] -> sum(axis=0) -> c[H,]
            (행벡터)  (열벡터)
                cf1. repeat 없이 그대로 hadmard product를 수행해도 ar이 [T,H]로 broadcasting되므로 hadmard product 결과는 같다.
                     그러나 fpas와 bpass에서 copy gate를 통과함을 명시적으로 구현하기 위해 비효율적이라도 repeat를 사용하자.
                cf2. 사실 가중합 작업은 행렬곱 np.matmul(a, hs)로 간단하게 처리할 수 있으나,
                     이걸 batch form으로 확장하면 3차원의 tensor곱을 수행해야 하므로 np.tensordot()이나 np.einsum()을 사용해야 한다.
                     복잡하다. 고로 요렇게 곱해지는 대상을 repeat -> 곱하는 대상과의 hadmard product -> sum의 단계로 분해한다.
    2. Batch의 hs
        Batch 처리의 의미는 data point 작동의 병렬적, 독립적 처리이다.
        hs[N,T,H]의 data point [T,H]를 그 distribution a로 가중한 후 합한다.
        Data points는 각자의 distribution을 갖겠지.
        과정은 다음과 같다:
                                                        hs[N,T,H] ->
            a[N,T] -> ar[N,T,1] -> repeat(H, axis=2) -> ar[N,T,H] -> ⊙ -> t[N,T,H] -> sum(axis=1) -> c[N,H]
                cf. t[N,T,H]의 각 data point [T,H]의 행들을 (1.에서 했던 대로) 합해야지.
    ---
    WeightSum node는 (data로부터 (자동으로) 학습한) weight를 받아 가중합할 뿐이므로 학습할 parameter가 없다.
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        """
        Args:
            hs: [N,T,H]
            a:  [N,T]
        ---
        Returns:
            c: context vector   | [N,H]
        """
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)   # step 1. copy node (axis 2)
        t = hs * ar                                 # step 2. hs를 ar로 weighting (hadmard product)
        c = np.sum(t, axis=1)                       # step 3. summation (column-wise)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        """"
        Args:
            dc: [N,H]
        ---
        Returns:
            dhs: [N,T,H]
            da:  [N,T]
        """
        hs, ar = self.cache
        N, T, H = hs.shape

        # step 3
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)

        # step 2
        dhs = ar * dt
        dar = hs * dt

        # step 1
        da = np.sum(dar, axis=2)
        """
        Step 3. sum(axis=1) Node의 Bpass
            Summation한 axis 1을 따라 gradient를 distribte한다.
            repeat()를 사용해 distribution을 구현했다.
        Step 2. Hadmard Product
            Element-wise 연산이다.
            hs에 대한 local gradient는 각 원소에 곱해지는 a로 이루어진 행렬이다.
        Step 1. Copy Gate (repeat)
            Fpass에서 axis 2를 따라 copy했다.
            고로 Bpass에선 axis 2를 따라 gradient를 sum한다.
        """

        return dhs, da


class AttentionWeight:
    """Decoder Improvement II
    입력 sequence의 단어에 해당하는 hs[N,T,H]와 h[N,H]를 내적해 유사도를 구한 후
    그들을 softmax에 통과시켜 hs의 각 행에 mapping된 h와의 유사도를 입력 공간으로 하는 distribution을 만든다.
    요놈이 weight가 되지.
    ---
    h[N,T] -> repeat(T, axis=1) -> hr[N,T,H] -> ⊙ -> t[N,T,H] -> sum(axis=2) -> s[N,T] -> Softmax -> a[N,T]
                                   hs[N,T,H] ->
    └--------------------------- h와 hs의 행들의 내적 ------------------------┘
    ---
    AttentionWeight node는 (data로부터 (자동으로) 학습한) hs와 h를 받아 attention해 context를 뱉을 뿐이므로 학습할 parameter가 없다.
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        """
        Args:
            hs: from enc, 가변 길이 대응              | [N,T,H]
            h: dec의 tstep t의 LSTM layer의 출력     | [N,H]
        ---
        Returns:
            a: weight   | [N,T]
        """
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H).repeat(T, axis=1)   # step 1
        t = hs * hr                                 # step 2
        s = np.sum(t, axis=2)                       # step 3
        a = self.softmax.forward(s)                 # step 4

        self.cache = (hs, hr)
        return a

    def backward(self, da):
        """
        Args:
            da: [N,T]
        ---
        Returns:
            dhs: [N,T,H]
            dh:  [N,T]
        """
        hs, hr = self.cache
        N, T, H = hs.shape

        # step 4
        ds = self.softmax.backward(da)

        # step 3
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)  # sum(axis=2)의 bpass는 axis 2를 따라 gradient distribution

        # step 2
        dhs = hr * dt                               # hadmard product node의 bpass
        dhr = hs * dt

        # step 1
        dh = np.sum(dhr, axis=1)                    # repeat(axis=1)의 bpass는 axis 1을 따라 gradient summation

        return dhs, dh


class Attention:
    """Decoder Improvement III：AttnetionWeight -> WeightSum
    1. AttentionWeight
        Tstep t에서의 decoder의 LSTM layer의 출력인 h[N,T]가 (tstep t에서의) 최종 decoder 출력을 뱉는 데 있어
        각 입력 단어 vector -- i.e., hs의 각 행 -- 에 attention하는 정도인 attention weight a[N,T]를 얻는다.
    2. WeightSum
        Attention을 실제로 반영해야겠지. hs에 a로 각 입력 단어 vector에 attention하여 context vector c[N,H]를 뱉는다.
    ---
    Attention node는 attention의 mechanism을 구현할 뿐이다.
    즉, (잘 학습된 model에 의해) 잘 만들어진 hs와 h에 대해 attention mechanism이 의미적으로 타당하게 수행된다.
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        """
        Args:
            hs: from enc                      | [N,T,H]
            h: Testp t의 LSTM layer의 출력     | [N,T]
        ---
        Returns:
            c: [N,H]
        """
        a = self.attention_weight_layer.forward(hs, h)
        c = self.weight_sum_layer.forward(hs, a)

        self.attention_weight = a
        return c

    def backward(self, dc):
        """
        Args:
            dc: [N,H]
        Returns:
            dhs: [N,T,H]
            dh:  [N,T]
        """
        dhs0, da = self.weight_sum_layer.backward(dc)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1   # copy gate

        return dhs, dh


class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []    # Attention은 학습할 param이 없음 (attention mechanism 자체의 구현일 뿐)
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        """
        Args:
            hs_enc: [N,T,H]
                이건 encoder에서 온 놈이므로 Time 처리와 무관하다.
                Attention은 decoder에서 수행되므로, Time 처리는 Time 처리된 encoder로부터 hs_enc를 받은 decoder에 대한 것이다.
            hs_dec: [N,T,H]
                위의 AttentionWeight node의 h[N,H]가 T개 모인 chunk이다.
        Returns:
            c: [N,T,H]
        """
        N, T, H = hs_dec.shape
        c = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []     # 길이 T가 될 것

        for t in range(T):
            layer = Attention()
            c[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return c

    def backward(self, dc):
        """
        Args:
            dc: [N,T,H]
        ---
        Returns:
            dhs_enc: [N,T,H]
            dhs_dec: [N,T,H]
        """
        N, T, H = dc.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dc)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dc[:, t, :])
            dhs_enc += dhs  # hs는 copy되므로 grad 누적
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec