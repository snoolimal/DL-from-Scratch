import numpy
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from common.base import Model, Optimizer
from utils import DataLoader, adjust_grads


class Trainer:
    def __init__(
            self,
            model: Model,
            optimizer: Optimizer
    ):
        self.model = model
        self.optimizer = optimizer

        self.loss_list = []

    def fit(
            self, x, t,
            max_epoch: int = 10,
            batch_size: int = 32,
            max_grad: bool = None
    ):
        model = self.model
        optimizer = self.optimizer
        self.loss_list = []

        dataloader = DataLoader(x, batch_size, t)

        start_time = time.time()
        pbar = tqdm(range(1, max_epoch + 1))
        for epoch in pbar:
            total_loss = 0
            for batch_x, batch_t in dataloader:
                # optimizer.zero_grad()
                """
                각 node에서 계산된 dw는 이전 batch에서 계산된 dw를
                self.grads에서 곧바로 dw를 꺼내 수정하거나 dw를 계산한 후 `self.grads[...] = dw`로 덮어 쓰므로
                처음에 gradient 그릇을 비울 필요는 없다. 
                """
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = adjust_grads(model.params, model.grads, max_grad)
                optimizer.step(params, grads)
                """
                model instance는
                ```
                # Computation Graph
                self.params, self.grads = [], []
                for layer in self.layers:
                    self.params += layer.params
                    self.grads += layer.grads
                ```
                로 params, grads instance var로 가진다.
                e.g.
                총 layer가 2개이고 layer 1의 params가 [...]_1, layer 2의 params가 [[...]_21, [...]_22]면
                self.params는 list의 += 연산으로
                [[...]_1, [...]_21, [...]_22]가 그대로 담긴다.
                self.grads에도 같은 순서로 각 param의 gradient 그릇
                [g[...]_1, g[...]_21, g[...]_22]로 그대로 만들어지고,
                각 layer의 .backward()가 호출될 때마다 해당하는 grad가 덮어 씌워진다.    
                """

                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            elapsed_time = time.time() - start_time
            pbar.set_description(
                f'Epoch {epoch} || Avg Train Loss: {avg_loss:.3f}  ({elapsed_time:.3f}s)'
            )
            if hasattr(avg_loss, 'get'): avg_loss = avg_loss.get()  # cupy array라면 GPU에서 CPU로
            self.loss_list.append(float(avg_loss))

    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        min_loss = min(self.loss_list)
        min_epoch = int(numpy.argmin(self.loss_list))

        plt.figure(figsize=(10, 6))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train', color='#1f77b4')
        plt.axhline(y=min_loss, color='r', linestyle='--', alpha=0.5, linewidth=0.5)
        plt.axvline(x=min_epoch, color='r', linestyle='--', alpha=0.5, linewidth=0.5)
        plt.plot(min_epoch, min_loss, 'ro', label=f'min: ({min_epoch + 1}, {min_loss:.4f})')
        plt.annotate(f'({min_epoch + 1}, {min_loss:.2f})',
                     xy=(min_epoch, min_loss), xytext=(-40, 20), textcoords='offset points', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Train Loss')

        plt.show()
