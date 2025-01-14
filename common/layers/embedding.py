from config import GPU, np


class Embedding:
    """
    w로부터 단어 ID에 해당하는 행, 즉 단어의 분산 표현 밀집 벡터를 추출한다.
    ---
    One-hot form의 context를 행으로 쌓은 행렬과 w의 행렬곱은 w의 행을 꺼내는 작업과 같다.
    따라서 해당 작업을 수행하는 embedding layer를 구현해 matmul node를 대체해 연산의 효율성을 높이자.
    ---
    cf. NLP에서 단어의 밀집 벡터 표현을 단어 임베딩 혹은 단어의 분산 표현이라 한다.
        학습 결과로 (model에 대응하는) 제대로 된 단어의 분산 표현을 얻는 것이 목표니 이 벡터들은 학습 과정에서 계속 갱신되지.
    """
    def __init__(self, w):
        """
        Args:
            w: [V,H]
        """
        self.params = [w]                   # 단어의 밀집 벡터 표현
        self.grads = [np.zeros_like(w)]
        self.indices = None                 # batch의 context indices

    def forward(self, indices):
        """
        Args:
            indices: batch의 단어 IDs | [N,]
        """
        w, = self.params
        self.indices = indices              # 요놈의 dns grad는 dump, self.grads와 self.params에 미포함
        y = w[indices]

        return y

    def backward(self, dy):
        """
        Embedding의 forward pass는 본질적으로 index-based row selection, 즉 element-wise 연산:
            고른 weight row에 해당하는 원소들은 identity function을 통과한다.
            고르지 않은 나머지 원소들은 0을 출력한다.
        이다.
        그러므로 dy와 같은 shape의 행렬을 만들고 골라진 row(원소)들의 index만 1로, 나머지는 0으로 채운 결과가 dloc이다.
        Downstream gradient dw는 dy와 dloc의 hadmard product를 수행하면 바로 얻는다.
        ```
        dw, = self.grads
        dw[...] = 0
        dw[self.indices] = dy
        ```
        ---
        사실 이건 좀 비효율적인 방법이긴 하다.
        원칙대로 dw를 w와 같은 shape의 행렬로 만들면 이는 채워지지 않은 대부분의 0 row를 갖는 행렬이다.
        그러나 하고 싶은 일은 w의 선택된 row를 갱신하는 것이므로 굳이 비효율적으로 많은 메모리를 잡아 먹는 dw를 만들 필요가 없다.
        선택된 row indices에 대응하는 ups grad를 (dy에서 indices로 indexing해) 골라 두면
        w의 특정 row만 갱신하는 작업을 보다 효율적으로 구현할 수 있다.
        더불어 고른 indices 중 batch 내에서 여러 번 등장하는 index가 존재할 수 있겠지. 가중치의 공유가 일어나면 gradient는 합산되어야 한다.
        (해당하는 ups grad를 할당이 아니라) dw를 0으로 초기화한 후 누적합하면 겹치는 경우까지 커버할 수 있다.
        여기서 구현한 건 batch 내에서의 누적합. Embedding layer는 context에 대해 공통적으로 사용:
            e.g.
            ```
            self.in_layers = []
            for i in range(context_window_size):
                layer = Embedding(w_in)     # 공유하는 가중치
                self.in_layers.append(layer)
            ```
        하므로 embedding layer의 param, 위 예시의 w_in은 adjust_grad에 걸려 각 context_window_size번 누적될 것이다.
        ---
        Args:
            dy: 전체 ups grad가 아닌 dy에서 self.indices에 해당하는 골라둔 행(들, batch)
        """
        dw, = self.grads    # dns grad를 별도로 계산하고 self.grads[...] = dw로 self.grads에 준 것이 아니라
                            # self.grads를 dw로 꺼내 직접 다루었으므로 self.grads[...] = dw가 불필요
        dw[...] = 0         # dw 자체를 0으로 설정하는 것이 아닌 dw의 shape을 유지한 체 그 원소들을 0으로 (이미 zeros_like가...?)
        if GPU:
            import cupyx
            cupyx.scatter_add(dw, self.indices, dy)     # dw의 axis=0의 self.indices에 dy를 더함
        else:
            import numpy
            numpy.add.at(dw, self.indices, dy)
        """dw[...] = 0
        dw = np.zeros_like(w)라 해도 이 라인은 반드시 필요하다.
        self.grads에서 dw를 꺼내 직접 다루되 scatter_add()와 add.at()으로 dw는 새로운 dw가 누적합된 결과이다.  
        이때 dw는 model architecture를 생성할 때 __init__()에서 한 번만 zeros_like(w)로 초기화:
            e.g.
            self.in_layers = []
            for i in range(context_window_size):
                layer = Embedding(w_in)     # 공유하는 가중치
                self.in_layers.append(layer)
        되고 이후의 bpass에서는 계속 같은 dw object가 재사용된다.
        그러므로 매 batch 처리로 밟는 gradient step마다 이전까지의 gradient값들이 누적된 dw를 비우고 새롭게 값을 누적시켜야 한다.  
        고로, 매 bpass마다 명시적으로 dw를 0으로 초기화해 optimizer.zero_grad()를 구현해야 한다.
        dw를 별도로 계산하고 self.grads[...] = dw로 self.grads에 값을 주었다면 자연스럽게 매번 backward가 호출될 때마다
        dw가 새롭게 계산되므로 자연스럽게 문제가 발생하지 않고 optimizer.zero_grad()가 실현되지만,
        이 경우 self.grads를 직접 다루어 gradient를 계산하므로 backward()에서 dw[...] = 0로 optimizer.zero_grad()를
        명시적으로 구현해야 한다.
        """

        return None     # optimizer로 보낼 dns grad뿐


class EmbeddingDot:
    """
    Negative sampling을 위한 embedding layer이다.
    ---
    이 layer의 w도 학습되는 main embedding layer의 w이다. 학습을 통해
        w_in: you 벡터, goodbye 벡터를 골라 담고 있는 h 벡터에 대해
        w_out:
            you 벡터, goodbye 벡터와의 내적값이 가장 큰 벡터가
            say 벡터이도록
    say, you, goodbye의 밀집 벡터 표현을 얻는다.
    ---
    Args:
        w: w_out | [V,H]
            cf. H: hidden_size
                V: vocab_size
    """
    def __init__(self, w):
        self.embed = Embedding(w)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, indices):
        """
        w_out을 전치가 아닌 정방향으로 다룬다. 위에서 구현한 embedding node를 사용하는데, w[indices]로 indexing하도록 구현되어 있으니까.
        어차피 w_out은 w_in과 다르고 w_out에 대해 수행되는 연산은 동일하다. 구현에서는 w_out의 행이 단어 밀집 벡터를 의미할 뿐.
        ---
        e.g.
        w: [[0, 1, 2, 3], [4, 5, 6, 7], ..., [16, 17, 18, 19]]      | [H=5,V=4]
        indices: [0, 3, 1]                                          | (N=3)
        target_w: [[0, 1, 2, 3], [12, 13, 14, 15], [4, 5, 6, 7]]    | [N=3,H=4]
        h: [[20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]   | [N=3,H=4]
        target_w * h: [[0, 21, 44, 69], ... ]]                      | [N=3,H=4]
        y: [134, 1382, 654]                                         | [N=3,]
        """
        target_w = self.embed.forward(indices)
        y = np.sum(h * target_w, axis=1)    # 내적 (행렬의 hadmard product 후 axis=1의 합)
        self.cache = (h, target_w)          # bpass에 필요한 값

        return y

    def backward(self, dy):
        """
        EmbeddingDot node의 forward pass는 target의 embedding과 내적:
             w_out ->
            target -> Embedding -> target_w ->
                                          h -> Dot
        이다.따라서 bpass는
            (optimizer) dw_out <-
                (dump) dtarget <- Embedding <- dtarget_w <-
                                             (stream) dh <- Dot
        이다. (fig 4-12 (p.165(166))
        Dot node는 각 행을 matmul node의 special case로 보고 그 결과들을 행으로 쌓은 형태가 최종 downstream gradient이다.
            target_w: [N,H], h: [N,H], y: [N,]에 대해 행이 1개만 존재(N=1)한다고 생각하면
                dtarget_w = np.dot(h.T, dy) = h * dy
                dh = np.dot(dy, target_w.T) = dy * target_w
            이다.
            Dot node의 fpass는 연산이 행 단위로 독립적으로 수행:
                [N,H]와 [N,H]의 batch dim(N의 axis=0)을 따른 독립적인 연산이 이루어진다.
            되므로, bpass에서도 위 결과를 그대로 행 단위로 독립적으로 수행해 행 방향으로 쌓으면 된다.
                d_target_w = h ⊙ dy
                dh = dy ⊙ target_w
        Embedding node는 fpass에서 선택한 rows를 그대로 고르고 그들에만 ups grad를 곱해 optimizer로 흘린다.
            cf. w의 downstream gradient를 optimizer로 보내고 다른 입력의 downstream gradient는 dumping한다.
                위에서 구현해 두었다.
        ---
        Args:
            dy: SigmoidWithLoss node의 downstream gradient [[y1-t1, ..., yN-tN] | [N,]
        """
        h, target_w = self.cache
        dy = dy.reshape(dy.shape[0], 1)     # single data라도 batch form으로 일반화

        # dns grad towards optimizer
        dtarget_w = h * dy                  # hadmard product (따라서 교환법칙 성립, =dy*h)
        self.embed.backward(dtarget_w)      # dw_out 처리

        # dns grad to stream
        dh = dy * target_w                  # hadmard product

        return dh
