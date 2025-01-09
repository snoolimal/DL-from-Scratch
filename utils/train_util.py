from config import np


class DataLoader:
    def __init__(self, x, batch_size, t=None, shuffle=True):
        if t is not None and (len(x) != len(t)):
            raise ValueError('Data and target must have the same length.')
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if t is not None and not isinstance(t, np.ndarray):
            t = np.array(t)

        self.x = x
        self.batch_size = batch_size
        self.t = t
        self.shuffle = shuffle

    def __iter__(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indices)     # indices면 매 epoch이 아닌 instance 생성 시에만 shuffle되겠지

        for i in range(0, len(self.x), self.batch_size):
            batch_indices = self.indices[i: i + self.batch_size]

            if self.t is not None:
                yield self.x[batch_indices], self.t[batch_indices]
            else:
                yield self.x[batch_indices]

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size


class GradCntrller:
    def __init__(self):
        pass

    def __call__(self, params, grads, max_norm):
        params, grads = self.sum_duplicated_grads(params, grads)
        if max_norm is not None: grads = self.clip_grads(grads, max_norm)

        return params, grads

    @staticmethod
    def sum_duplicated_grads(params, grads):
        params, grads = params[:], grads[:]

        def check(params, i, j, transpose=False):
            if not transpose: return params[i] is params[j]
            else: return (params[i].ndim == 2 and
                          params[j].ndim == 2 and
                          params[i].T.shape == params[j].shape and
                          np.all(params[i].T == params[j]))

        while True:
            find_flg = False

            for i in range(len(params) - 1):
                for j in range(i + 1, len(params)):
                    if check(params, i, j):
                        grads[i] += grads[j]
                        find_flg = True
                        params.pop(j)
                        grads.pop(j)
                    elif check(params, i, j, transpose=True):
                        grads[i] += grads[j].T
                        find_flg = True
                        params.pop(j)
                        grads.pop(j)

                    if find_flg: break
                if find_flg: break

            if not find_flg: break

        return params, grads

    @staticmethod
    def clip_grads(grads, max_norm):
        total_norm = 0
        for grad in grads:
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)    # grad의 L2-norm

        rate = max_norm / (total_norm + 1e-6)
        if rate < 1:    # grad의 크기가 설정한 max보다 커지면
            for grad in grads:
                grad *= rate

        return grads