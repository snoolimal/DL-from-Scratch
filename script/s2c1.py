from config import np
from data import spiral
from model import TwoLayerNet
from common import Trainer, SGD


seed = 42

x, t = spiral.load_data(seed)
input_size = int(np.prod(np.array(x.shape[1:])))
num_classes = t.shape[-1]

hidden_size = 10
max_epoch = 300
batch_size = 32
lr = 1.0

model = TwoLayerNet(input_size, hidden_size, num_classes)
optimizer = SGD(lr)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size)
trainer.plot()
