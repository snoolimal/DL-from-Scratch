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
