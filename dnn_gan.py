# dnn_gan.py
# GAN 신경망 활용 손글씨 흉내내기 이미지 만듦

# GAN(Generative Adversarial Network)
# 딥러닝 모델 중 이미지 생성에 널리 쓰이는 모델임
# 이미지 데이터셋과 유사한 이미지를 만들려고 하는 것임

# 패키지 임포트
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os
import glob # 파일들의 리스트를 뽑을 때 사용함
# glob('*.확장자') : 해당 패턴의 파일 리스트를 지정 디렉토리에서
#           읽어와서 반환함
# 파일 관련 모듈 : pickle, glob, os.path

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.models import Sequential

# 하이퍼 파라미터
M_GEN = 128
M_DIS = 128
M_NOISE = 100

M_SHAPE = (28, 28, 1)
M_EPOCH = 5000
M_BATCH = 300

# 출력 이미지(만든 가짜 이미지) 저장 폴더 생성
M_FOLDER = 'output/'
os.makedirs(M_FOLDER, exist_ok=True)
# exist_ok=True : 생성할 폴더가 이미 존재하는 경우 에러 표시 안함
# False : 생성할 폴더가 존재하면 에러 표시함 (FileExistError)

# 만약, output/ 폴더가 이미지 존재한다면, 폴더 안의 파일을 삭제처리
# glob() : 파일 형식(확장자)을 이용해서 파일들을 선택하는 함수임
#             파일 목록을 list 로 리턴함

for f in glob.glob(M_FOLDER + '*'):
    os.remove(f)
# 함수 단위로 작성하고 실행 테스트하는 구조로 작성할 것임

## 데이터 준비 ###
def read_data():
    # 학습용 입력값만 사용 (GAN은 비지도학습)
    (X_train, _), (_, _) = mnist.load_data()

    # 데이터 모양 확인
    print('데이터 모양 확인 : ', X_train.shape)
    plt.imshow(X_train[0], cmap='gray')
    plt.show()

    # 데이터 스케일링 [-1, 1] : 0 --> -1, 255 --> 1
    X_train = X_train / 127.5 - 1.0

    # 채널 정보 추가
    X_train = np.expand_dims(X_train, axis=3)
    print('데이터 모양 : ', X_train.shape)

    return X_train
# ------------------------------------------------

### 인공신경망 구현 ###

# 생성자 설계 구현 : 가짜 이미지 생성하는 학습 모델
def build_generator():
    model = Sequential()

    # 입력층 + 은닉층 1
    model.add(Dense(M_GEN, input_dim=M_NOISE))
    model.add(LeakyReLU(alpha=0.01))
    # 은닉층의 활성화함수는 입력값이 작으면 작은 값을 출력
    # 입력값이 크면 큰 값을 출력하는 함수를 사용함
    # 은닉층에서 활성화함수로 sigmoid 함수를 사용하면,
    # 경사 소실 문제가 발생하기 때문에 다른 활성화함수를 사용해서
    # 경사 소실 문제를 해결해야 함
    
    # LeakyReLu 함수(LReLu 라고도 함)
    # ReLu 함수를 개량한 함수로 알파값을 0.01 곱하는 함수
    
    # 은닉층 2
    model.add(Dense(M_GEN))
    model.add()
    
    # 은닉층 3 + 출력층
    # tanh 활성화는 [-1, 1] 스케일링
    model.add(Dense(28*28*1, activation='tanh'))
    model.add(Reshape(M_SHAPE))

    print('생성자 신경망 요약 정보')
    model.summary()
    return model

######################################################
### 컨트롤 타워 : 위에 작성된 함수 실행하는 부분임

# 데이터 준비
X_train = read_data()

# GAN 신경망 구현
