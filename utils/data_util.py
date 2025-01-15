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
            batch_indices = self.indices[i: i + self.batch_size]    # 넘쳐도 자동 넘쳐도 예외 처리

            if self.t is not None:
                yield self.x[batch_indices], self.t[batch_indices]
            else:
                yield self.x[batch_indices]

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size


class TimeDataLoader(DataLoader):
    def __init__(self, seq_len, *args, padding=-1, shuffle=False, **kwargs):
        super().__init__(*args, shuffle=shuffle, **kwargs)

        self.seq_len = seq_len
        self.padding = padding  # TimeSoftmaxWithLoss()의 ignore_label로 지정한 값과 같아야

        self.offsets = None

        # TODO: 이 pad_data()가 필요한가? 어차피 다시 처음으로 돌아가도록 처리했는데.. 아우 헷갈려라.
        # self.pad_data()                                         # 넘치면 padding으로 예외 처리
        self.data_size = len(self.x)  # self.x는 padding된 전체 sequence
        self.jump = self.data_size // self.batch_size

    def __iter__(self):
        # self.indices = np.arange(len(self.x))
        # if self.shuffle:
        #     np.random.shuffle(self.indices)

        if self.offsets is None:
            offsets = [i * self.jump for i in range(self.batch_size)]    # 첫 chunk(의 indices) (각 raw batch의 시작점)
            self.offsets = np.array(offsets)

        for _ in range(self.jump + 1):
            batch_x = np.empty((self.batch_size, self.seq_len), dtype='i')
            batch_t = np.empty_like(batch_x, dtype='i') if self.t is not None else None

            for t in range(self.seq_len):
                batch_indices = (self.offsets + t) % self.data_size       # % data_size를 추가해 끝나면 다시 처음으로 가는 작동 처리!
                batch_x[:, t] = self.x[batch_indices]
                if self.t is not None:
                    batch_t[:, t] = self.t[batch_indices]

            self.offsets += self.seq_len

            if self.t is not None:
                yield batch_x, batch_t
            else:
                yield batch_x

    def pad_data(self):
        _data_size = len(self.x)
        remainder = _data_size % self.batch_size
        if remainder > 0:
            pad_size = self.batch_size - remainder
            self.x = np.concatenate([self.x, np.full((pad_size,), self.padding)], axis=0)
            if self.t is not None:
                self.t = np.concatenate([self.t, np.full((pad_size,), self.padding)], axis=0)

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size