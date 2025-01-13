<style>
body {
    line-height: 1.6;
    font-size: 12px;
    /* font-family: Arial, sans-serif; */
}
.indent{
    margin-left: 2em; /* 전체 들여쓰기 */
    line-height: 1.8;
}
</style>



$T$개의 단어열로 나타난 corpus $(w_1,w_2,\cdots,w_T)$를 고려하자.

CBOW는 contexts로부터 target을 추측한다. 즉, (window size가 $1$이라면) 분포 $\mathcal P(W_t|W_{t-1},W_{t+1})$를 모델링한다. Window를 좌우 대칭이 아닌 왼쪽으로만 한정한다면 CBOW는 $\mathcal P(W_t|W_{t-2},W_{t-1})$을 모델링할 것이다. (SL 모델인) CBOW의 objective function은 size $N$의 batch에 대해
<br> &emsp;
$
\displaystyle
\mathbf L = \frac{1}{N}\sum_{t=1}^{N}-\log p_\theta(w_t|w_{t-2},w_{t-1})
$
<br>
이다. $\theta$가 최적화될수록 CBOW는 context로부터 target을 더 정확하게 추측한다.

위의 obj func에서 보듯이 CBOW의 학습은 context로부터 target을 정확하게 추측하는 $\theta$를 찾는 작업이며, 그 과정에서 부산물로 단어의 의미가 encoding된 vector인 단어의 분산 표현을 얻게 되므로 word2vec 모델로써 기능하게 된다.

---

CBOW의 본래 작동인 context로부터의 target의 추측은 언어 모델에 활용될 수 있다. 언어 모델은 단어의 나열에 확률을 부여하는 모델이다. 특정한 단어 sequence가 일어날 가능성이 어느 정도인지 — i.e., 얼마나 자연스러운 순서로 배열된 단어 sequence인지 — 를 출력하는 것이다.

그러므로 언어 모델은 새로운 문장을 생성하는 용도로도 이용할 수 있다. 언어 모델은 단어 순서의 자연스러움을 확률적으로 평가하므로, 그 분포에서의 sampling으로 다음에 등장할 적절한 단어를 생성할 수 있기 때문이다.

길이 $m$의 단어 sequence $(w_1,\cdots,w_m)$를 고려하자. 결합확률인 단어 sequence의 발생 확률 $\mathrm P(w_1,\cdots,w_m)$을 chain rule로 factorize하면
<br> &emsp;
$
\displaystyle\mathrm P(w_1,\cdots,w_m) = \prod_{t=1}^{m}\mathrm P(w_t|w_1,\cdots,w_{t-1})\quad\small{(\mathrm P(w_1)\coloneqq\mathrm P(w_1|w_0))}
$
<br>
이다. 고로 궁극적으로 모델링할 목표 분포 — i.e., 언어 모델 — 는 이전에 발생한 단어 sequence를 조건부로 한 조건부 분포 p(w_t|w_1,w_2,\cdots,w_{t-1})$이다. 이 분포를 규명하면 전체 단어 sequence의 발생 확률 $\mathrm P(w_1,\cdots,w_m)$를 계산할 수 있으니까.

---

Word2vec 모델인 CBOW로 언어 모델을 규명(학습)하려면, 특정 개수의 왼쪽 단어를 context로 하여 근사:
<br> &emsp;
$
\begin{aligned}
p(w_t|w_1,w_2,\cdots,w_{t-1})
&= \prod_{t=1}^{m}p(w_t|w_1,\cdots,w_{t-1})
\\
&\approx \prod_{t=1}^{m}p(w_t|w_{t-2},w_{t-1})
\end{aligned}
$
<br>
해 얻을 수 있다. 즉, CBOW를 학습해 언어 모델로 하는 작업은 (temporal process인 단어 sequence의 분포인) 언어 모델을 직전 $2$개의 사건에만 의존하는 $2$ state markov chain으로 근사시켜 규명하는 일이다.

그러나 context가 커버하지 못하는 앞쪽 단어를 반영하지 못한다 — i.e., 조건부로 할 수 없다 — . WIndow size를 키워 context에 포함되는 단어의 개수를 늘릴 순 있다.

하지만 context의 길이를 늘리는 것만으론 불충분하다. CBOW는 context 안에서의 단어의 순서를 무시하기 때문다. Continous bag-of-words라는 이름에서도 알 수 있듯, bag에 담긴 단어들에는 순서가 무시된다 — 대신 bag, 즉 순서 대신 분포를 이용한다 — .

이것은 CBOW의 모델 구조를 보면 명백하다. Batch
$\begin{bmatrix}
\text{\small you}(1) ~ \text{\small say}(3)
\\
\text{\small say}(3) ~ \text{\small you}(1)
\\
\vdots
\\
\end{bmatrix}_{[N,2]}$
를 예로 들면, $w_{\text{in}}$:
<br> &emsp;
$
w_{\text{in}} = \begin{bmatrix}
w_1 \\ w_2 \\ \vdots
\end{bmatrix}_{[V,H]}
$
<br>
를 갖는 embedding layer로부터 각각
$\begin{bmatrix}
\text{\small you}(1) \\ \text{\small say}(3) \\ \vdots
\end{bmatrix}_{[N,]}$
과
$\begin{bmatrix}
\text{\small say}(3) \\ \text{\small you}(1) \\ \vdots
\end{bmatrix}_{[N,]}$
에 해당하는
$\begin{bmatrix}
w_1 \\ w_3 \\ \vdots
\end{bmatrix}_{[N,H]}$
와
$\begin{bmatrix}
w_3 \\ w_1 \\ \vdots
\end{bmatrix}_{[N,H]}$
를 얻고 둘을 더해 $h$:
<br> &emsp;
$
h = \begin{bmatrix}
w_1 \\ w_2 \\ \vdots
\end{bmatrix} + \begin{bmatrix}
w_2 \\ w_1 \\ \vdots
\end{bmatrix} = \begin{bmatrix}
w_1+w_2 \\ w_2+w_1 \\ \vdots
\end{bmatrix}_{[N,H]}
$
<br>
를 얻는다. 즉, (batch에 담긴) context $\small{\text{(you, say)}}$와 $\small{\text{(say, you)}}$는 똑같이 취급된다 — 이 문제에 대처하기 위해 context vector들을 concatenate할 수도 있지만, 그렇다면 $w_{\text{in}}$의 크기는 매우 커진다 — . 그러므로 CBOW를 언어 모델로 사용하는 것은 적합하지 않다.
<div class="indent">
cf. word2vec은 단어의 분산 표현을 얻을 목적으로 고안된 기법이므로 이를 언어 모델로 사용하는 경우는 드물다. RNN을 이용한 언어 모델에서 단어의 분산 표현을 얻을 수 있지만, 단어 수 증가에 대응하고 분산 표현의 질 개선을 위해 word2vec이 제안되었다.
</div>

---

RNN은 context가 아무리 길더라도 context의 정보를 기억하는 mechanism을 가지고 있다. 따라서 RNN을 사용하면 아무리 긴 temporal(time series) data에 대응할 수 있다.
<div class="indent">
#TODO: 확인 필요 <br>
오래 전의 시간을 커버하지 못하는 것은 RNN도 어쩔 수 없다. 이 문제는 context의 길이를 늘려 이 문제를 커버할 수 밖에 없다. 이때 (그러한 긴) context 속 단어들의 순서를 고려하지 않는 CBOW와 달리, RNN은 context의 정보 -- i.e., 단어의 배열 순서 -- 를 기억하도록 한다. 따라서 RNN은 단어 sequence의 자연스러움을 다루는 언어 모델로서 사용될 수 있다. 또한 LSTM은 RNN을 발전시켜 오래 전의 시간을 커버하지 못하는 문제를 더욱 보완하였다.
</div>