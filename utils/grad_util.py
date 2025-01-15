"""
.backward()로 게산한 gradient를 optimizer에 넘기기 전 필요한 처리들을 수행한다.
"""

from config import np


def adjust_grads(params, grads, max_norm):
    params, grads = sum_duplicated_grads(params, grads)
    if max_norm is not None: clip_grads(grads, max_norm)

    return params, grads


def sum_duplicated_grads(params, grads):
    # TODO: RNN까지 보고 CBOW, Skipgram 고려해 이 부분 정리
    #  일반적인 copy gate인 병렬 vs. 시간 의존성을 갖는 copy gate인 직렬 vs. 그냥 여러 node로 취급도 가능한 criterion
    #  그러나 원리 자체는 chain rule에서의 미분의 선형성으로 동일하므로 뭐든 grad는 합산됨
    #  인 듯?
    """
    여러 nodes에서 gradient가 계산되는 공유된 weight가 있다면 그들을 모아 준다.
    Optimizer의 step()의 params argument에 중복된 weight가 존재한다면 그들의 gradient를 더해 하나로:
        RNN의 weight 공유(재사용)는 copy gate와 엄밀히는 다르다.
        Copy gate는 동일한 시점에 입력을 여러 경로로 복사하는 병렬적 복사인 반면
        RNN은 시간축을 따라 순차적으로 weight를 재사용하는 순차적 재사용이기 때문이다.
        하지만 RNN에서든 copy gate에서든 동일한 입력(weight)가 여러 함수에 독립적으로 사용되고,
        loss에 대한 최종 gradient는 미분의 선형성과 chain rule에 의해 각 경로의 gradient 합으로 계산된다는 원리는 동일하다.
        RNN뿐 아니라 CBOW나 Skipgram 등 encoding을 수행하는 weight를 공유하는 구조를 가져
            cf. 요놈들은 시간축을 따라 weight를 공유하진 않는다.
        하나의 weight에 대해 여러 경로에서 gradient가 계산되는 모델이라면, 이러한 원리:
            1. 동일한 입력(weight)이 각각의 loss를 뱉는 여러 함수(node)들에 독립적으로 사용되고
            2. final scalar loss는 이러한 loss들을 합산(평균) 만들어지므로
            3. 미분의 선형성과 chain rule에 따라 각 경로의 gradient가 더해져 loss의 weight에 대한 최종 gradient가 결정된다
        는 모두 동일하다.
        PyTorch나 TensorFlow 등 autodiff를 수행하는 DL framework에서는 자동으로 gradient를 합산해 준다.
        지금과 같이 weight 공유 구조를 명시적으로 구현 -- e.g., 특정 layer를 여러 번 호출 -- 한다면
        gradient 합산을 명시적으로 올바르게 처리해야 한다.
    만든다.
    요렇게 gradient step을 밟을 때 -- i.e., weight update를 수행할 때 -- 전달된 grad의 param의 중복얼 없애야
    구현해 둔 optimizer가 의도대로 동작한다.
    ---
    Model class의 computation graph
    ```
    self.params, self.grads = [], []
    for layer in self.layers:
        self.params += layer.params
        self.grads += layer.grads
    ```
    ---
    Trainer class의 fit()의 gradient step 구현
    ```
    params, grads = gradctrller(model.params, model.grads, max_grad)
    optimizer.step(params, grads)
    ```
    ---
    Optimizer class의 subclass의 step()
    ```
    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
    ```
    """
    params, grads = params[:], grads[:]     # shallow copy
                                            # params, grads의 수정은 model.params, model.grads의 직접 수정

    def check(params, i, j, transpose=False):
        if not transpose: return params[i] is params[j]     # 두 param이 같은 값을 "참조"한다면 ("값"이 같은 param이 아닌)
        else: return (params[i].ndim == 2 and
                      params[j].ndim == 2 and
                      params[i].T.shape == params[j].shape and
                      np.all(params[i].T == params[j]))

    while True:
        find_flg = False

        for i in range(len(params) - 1):
            for j in range(i + 1, len(params)):
                # weight matrix가 공유된 경우
                if check(params, i, j):
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # weight matrix가 전치된 형태로 공유된 경우
                elif check(params, i, j, transpose=True):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads    # 메모리에 새롭게 할당해 (새로운 참조로) 반환 (함수의 반환이므로)


def clip_grads(grads, max_norm):
    """
    Gradient의 explosion을 막기 위해 gradient를 clipping한다.
    Back propagation 수행 시 gradient가 지나치게 커쳐 폭발하면, 학습 과정에서 weight가 너무 크게 update되며 발산해버린다.
    이로 인해 학습이 불안정해지고 network 전체가 발산할 가능성이 높아진다.
    특히 RNN류의 neural network는 본질적으로 하나의 weight만을 공유해 사용하므로,
    each timestep마다 bpass가 진행되면서 gradient가 누적되기에 특정 timestep을 넘어가면 gradient의 크기가 지나치게 커진다.
    ---
    cf. Weight matrix의 eigen value가 1보다 크다면 bpass를 수행할 때마다 그러한 eigen value가 반복적으로 곱해짐에 따라
        gradient의 크기가 지수적으로 증가한다.
    """
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)    # grad의 L2 norm

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:            # grad의 크기가 설정한 max보다 커지면
        for grad in grads:
            grad *= rate    # 그 비율만큼 scaling해 max를 넘지 않도록 clipping

    # return grads
    """
    List나 ndarray와 같은 가변(mutable) obj는 참조로 전달되므로
    함수 내에서 수정한 obj는 함수 외부의 원본 obj를 직접 수정한다.
    sum_duplicated_grads()도 마찬가지이며
    요놈이 반환하는 params와 grads는 model.params와 model.grads는 원본이 수정된 후 메모리에 별도로 저장된 후 반환된다.
    __call__()에서는 sum_duplicated_grads() 호출 후 clip_grads()를 호출하므로
    model.params와 model.grads는 더는 다루지 않고, sum_duplicated_grads()가 반환환 새로운 obj를 다룬다.
    그러므로 이 함수의 return 여부와 무관히 더는 model의 attribute는 다루지 않는다.
    다만 굳이 clipping한 grads를 메모리에 또 만들지 않기 위해 이 return은 생략한다. 
    """
