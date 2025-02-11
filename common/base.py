import os
import pickle
from pathlib import Path
from config import *
from utils import cupy_util as cuu


class Model:
    def __init__(self):
        self.params, self.grads = None, None
        self.plot_grad_layer = None
        self.rood_dir = Path(__file__).resolve().parents[1]

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        params = [p.astype(np.float16) for p in self.params]
        if GPU:
            params = [cuu.to_cpu(p) for p in params]

        file_name = str(self.rood_dir / 'model' / 'storage' / (file_name + '.pkl'))
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        if '/' in file_name:
            # file_name = file_name.replace('/', os.sep)
            file_name = Path(file_name)

        file_name = self.rood_dir / 'model' / 'storage' / (file_name + '.pkl')
        # if not os.path.exists(file_name):
        #     raise IOError('No file: ' + file_name)
        if not Path.exists(file_name):
            raise IOError('No file: ' + str(file_name))

        with open(str(file_name), 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        if GPU:
            params = [cuu.to_gpu(p) for p in params]

        # BaseModel obj의 self.params를 방금 로드한 params로 최신화
        for i, param in enumerate(self.params):
            param[...] = params[i]  # ...는 param이 참조하는 메모리의 값을 수정 (param이 참조하는 메모리 주소는 동일하게 유지)


class Optimizer:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, *args):
        raise NotImplementedError
