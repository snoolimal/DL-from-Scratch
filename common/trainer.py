import numpy
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from common.base import Model, Optimizer
from util.train_util import DataLoader, GradCntrller


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
        gradctrller = GradCntrller()

        start_time = time.time()
        pbar = tqdm(range(1, max_epoch + 1))
        for epoch in pbar:
            total_loss = 0
            for batch_x, batch_t in dataloader:
                # optimizer.zero_grad()
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = gradctrller(model.params, model.grads, max_grad)
                optimizer.step(params, grads)

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
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Train Loss')
        plt.show()
