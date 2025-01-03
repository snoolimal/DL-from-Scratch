from utils.np import np


def img2col(x, kernel_size=(3, 3), stride=1, padding=0):
    """
    합성곱 연산을 image와 kernel의 행렬곱으로 수행하기 위해
    kernel이 훑고 지나가는 영역을 한 행으로 하는 행렬로 image를 flatten한다.
    ---
    Args:
        x: [N,C,_H,_W]
    Returns:
        x_flatten: [N*H*W,C*filter_h*filter_w]
            H, W: 출력 image의 크기
            axis 0: filter 적용 영역의 index
            axis 1: filter 적용 영역의 pixel값 -- i.e., filter와 내적할 벡터 --
            합성곱 연산은 x_flatten 행렬과 kernel 행렬:
                shape: [C*filter_h*filter_w,output_channels]
                    out_channels개의 (열 벡터인) kernel을 열 방향으로 쌓은 행렬
            과의 행렬곱을 통해 효율적으로 수행할 수 있다.
    """
    N, C, _H, _W = x.shape
    if np.isscalar(kernel_size):
        filter_h = filter_w = kernel_size
    else:
        assert kernel_size.__len__() == 2, "Pass (filter_height, filter_width) to kernel_size argument."
        filter_h = kernel_size[0]
        filter_w = kernel_size[1]

    H = (_H + 2 * padding - filter_h) // stride + 1
    W = (_W + 2 * padding - filter_w) // stride + 1

    # padding
    x_padded = np.pad(
        x,
        pad_width=(
            (0, 0),                 # batch axis에 앞뒤로 0개의 batch 추가 (axis 0)
            (0, 0),                 # channel axis에 앞뒤로 0개의 channel 추가 (axis 1)
            (padding, padding),     # height axis에 앞뒤로 padding개의 pad 추가 (axis 2)
            (padding, padding)      # width axis에 앞뒤로 padding개의 pad 추가 (axis 3)
        ),
        mode='constant', constant_values=0
    )  # [N,C,_H+(2*padding),_W+(2*padding)]

    # flattening
    x_flatten = np.zeros((N, C, filter_h, filter_w, H, W))
    for h in range(filter_h):
        h_max = h + stride * H
        for w in range(filter_w):
            w_max = w + stride * W
            x_flatten[:, :, h, w, :, :] = x_padded[:, :, h:h_max:stride, w:w_max:stride]
    x_flatten = x_flatten.transpose(0, 4, 5, 1, 2, 3)   # [N,H,W,C,filter_h,filter_w]
    x_flatten = x_flatten.reshape(N * H * W, -1)        # [N*H*W,C*filter_h*filter_w]

    return x_flatten


def col2img(x_flatten, x_shape, kernel_size, stride=1, padding=0):
    """
    역과정을 수행한다.
    Image 행렬을 받아 [N,C,_H,_W]의 batch image를 반환한다.
    ---
    Args:
        x_flatten: [N*H*W,C*filter_h*filter_w]
        x_shape: [N,C,_H,_W]
    ---
    Returns:
        x: [N,C,_H,_W]
    """
    N, C, _H, _W = x_shape
    if np.isscalar(kernel_size):
        filter_h = filter_w = kernel_size
    else:
        assert kernel_size.__len__() == 2, "Pass (filter_height, filter_width) to kernel_size argument."
        filter_h = kernel_size[0]
        filter_w = kernel_size[1]

    H = (_H + 2 * padding - filter_h) // stride + 1
    W = (_W + 2 * padding - filter_w) // stride + 1

    x_flatten = x_flatten.reshape(N, H, W, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    x = np.zeros((N, C, _H + 2 * padding + stride - 1, _W + 2 * padding + stride - 1))
    count = np.zeros_like(x)    # 각 (h,w)의 중첩 횟수를 기록
    for h in range(filter_h):
        h_max = h + stride * H
        for w in range(filter_w):
            w_max = w + stride * W
            x[:, :, h:h_max:stride, w:w_max:stride] += x_flatten[:, :, h, w, :, :]
            count[:, :, h:h_max:stride, w:w_max:stride] += 1

    # 중첩 횟수로 나누어 평균 get
    count[count == 0] = 1
    x = x / count

    return x[:, :, padding:_H + padding, padding:_W + padding]


# e.g.
if __name__ == '__main__':
    x_shape = (N, C, _H, _W) = (2, 2, 4, 4)
    x = np.arange(1, 65).reshape(*x_shape)

    """x_padded
    1st image, 1st channel              1st image, 2nd channel              
    [[00, 00, 00, 00, 00, 00],          [[00, 00, 00, 00, 00, 00],          
     [00, 01, 02, 03, 04, 00],           [00, 17, 18, 19, 20, 00],
     [00, 05, 06, 07, 08, 00],           [00, 21, 22, 23, 24, 00],
     [00, 09, 10, 11, 12, 00],           [00, 25, 26, 27, 28, 00],
     [00, 13, 14, 15, 16, 00],           [00, 29, 30, 31, 32, 00],
     [00, 00, 00, 00, 00, 00]]           [00, 00, 00, 00, 00, 00]]
    
    2nd image, 1st channel              2nd image, 2nd channel              
    [[00, 00, 00, 00, 00, 00],          [[00, 00, 00, 00, 00, 00],          
     [00, 33, 34, 35, 36, 00],           [00, 49, 50, 51, 52, 00],
     [00, 37, 38, 39, 40, 00],           [00, 53, 54, 55, 56, 00],
     [00, 41, 42, 43, 44, 00],           [00, 57, 58, 59, 60, 00],
     [00, 45, 46, 47, 48, 00],           [00, 61, 62, 63, 64, 00],
     [00, 00, 00, 00, 00, 00]]           [00, 00, 00, 00, 00, 00]]
    """

    conv_args = (kernel_size, stride, padding) = (2, 2, 1)  # -> H=3, W=3
                                                            # -> filter_h=filter_w=2 -> h=0~1, w=0~1
    x_flatten = img2col(x, *conv_args)

    """for loop
    h=0, w=1이라면
        [N=2,C=2,6,6]의 shape을 갖는 x_padded 대해
        x_padded의 h=0에서 시작해서 stride=2로 H=3개의 값을 봐야 하므로 h_max=6까지,
        x_padded의 w=1에서 시작해서 stride=2로 W=3개의 값을 봐야 하므로 w_max=7까지 indexing
    한다.
    
    x_flatten[:, :, h=0, w=1, :, :] = x_padded[:, :, 0:6:2, 1:7:2]의
    x_padded[:, :, 0:6:2, 1:7:2]는 N, C에 걸쳐 (h=0,w=1)에서 시작하여 stride=2의 간격으로 x_padded로부터 위치:
        (0,1), (0,3), (0,5)
        (2,1), (2,3), (2,5)
        (4,1), (4,3), (4,5)
    의 pixel값들을 sampling:
        [[[[ 0,  0,  0],
           [ 5,  7, 10],
           [13, 15,  0]],   # 위 위치의 [N=1st,C=1st]
          [[ 0,  0,  0],
           [21, 23,  0],
           [29, 31,  0]]],  # 위 위치의 [N=1st,C=2nd]
         [[[ 0,  0,  0],
           [37, 39,  0],
           [45, 47,  0]],
          [[ 0,  0,  0],
           [53, 55,  0],
           [61, 63,  0]]]]
    한다. 고로 sampled_x_padded의 shape은 [N=2,C=2,H=3,W=3]이 된다.
    요 값이 x_flatten의 corresponding 위치인 (h=0,w=1)에 할당된다.
    """

    """x_flatten
    reshaping을 거친 x_flatten은 다음과 같다: 
    [[00, 00, 00, 01,  00, 00, 00, 17],     # N=1st에 대해 (filter_h=0,filter_w=0) 시작점의 kernel이 커버하는 값
     [00, 00, 02, 03,  00, 00, 18, 19],     # N=1st에 대해 (filter_h=0,filter_w=2) 시작점의 kernel이 커버하는 값
                   ...
     [46, 67, 00, 00,  62, 63, 00, 00],     # N=2nd에 대해 (filter_h=4,filter_w=2) 시작점의 kernel이 커버하는 값
     [48, 00, 00, 00,  64, 00, 00, 00]]     # N=2nd에 대해 (filter_h=4,filter_w=4) 시작점의 kernel이 커버하는 값
    """

    x_origin_1 = col2img(x_flatten, x_shape, *conv_args)
    print(np.array_equal(x, x_origin_1))

    # breakpoint()

    conv_args = ((3, 2), 2, 1)
    x_flatten = img2col(x, *conv_args)
    x_origin_2 = col2img(x_flatten, x_shape, *conv_args)
    print(np.array_equal(x, x_origin_2))

    # breakpoint()

    print(np.array_equal(x_origin_1, x_origin_2))
