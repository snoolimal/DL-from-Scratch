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

    def __iter__(self):
        # self.indices = np.arange(len(self.x))
        # if self.shuffle:
        #     np.random.shuffle(self.indices)

        self.pad_data()                                         # 넘치면 padding으로 예외 처리
        data_size = len(self.x)                                 # self.x는 padding된 전체 sequence
        jump = data_size // self.batch_size
        offsets = [i * jump for i in range(self.batch_size)]    # 첫 chunk(의 indices) (각 raw batch의 시작점)

        offsets = np.array(offsets)
        batch_x = np.empty((self.batch_size, self.seq_len), dtype='i')
        batch_t = np.empty_like(batch_x, dtype='i')
        for t in range(self.seq_len):
            chunk_indices = offsets
            batch_x[:, t] = self.x[chunk_indices]
            if self.t is not None:
                batch_t[:, t] = self.t[chunk_indices]

            offsets += 1

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