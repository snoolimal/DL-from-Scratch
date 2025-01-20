from typing import List, Optional
from config import np
from common.functions import softmax
from model import RnnLM, BetterRnnLM


class RNNLMGen(RnnLM):
    def generate(self, start_id, skip_ids: Optional[List] = None, sample_size=100):
        """
        Args:
            start_id: 최초의 단어 ID
            skip_ids: 제외할 단어 IDs
                PTB dataset의 <unk>이나 N 등 전처리된 단어를 sampling하지 않기 위한 arg
                    cf. PTB dataset에서 희소한 단어는 <unk>으로, 숫자는 N으로 대체되어 있다.
            sample_size: sampling할 단어의 수
        ---
        Returns:
            word_ids: [sample_size,]
                첫 ID는 start_id, 나머지는 sample_size까지 sampling된 IDs이다.
        """
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)   # [N=1,T=1]
                                            # predict()는 batch, 즉 2D array에 대해 작동하므로 single raw x를 batch form으로
            y = self.predict(x)             # [N=1,T=1,V]
            p = softmax(y.flatten())        # distribution | [V,]

            sampled = np.random.choice(len(p), size=1, p=p)     # int라면 range(len(p))로 간주 | [scalar]
            if (skip_ids is None) or (sampled not in skip_ids):
                x = int(sampled.item())
                word_ids.append(x)

        return word_ids


class BetterRnnlmGen(BetterRnnLM):
    def generate(self, start_id, skip_ids: Optional[List] = None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            y = self.predict(x)
            p = softmax(y.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = int(sampled.item())
                word_ids.append(x)

        return word_ids