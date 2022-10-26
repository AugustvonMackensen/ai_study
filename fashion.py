# fashion.py
# 딥러닝 CNN 적용 패션 아이템 구분하기

import numpy as np
import matplotlib.pyplot as plt
from time import time

from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.layers import Flatten
from keras.layers import InputLayer
from keras.layers import Conv2D, MaxPool2D

# 하이퍼 파라미터
M_EPOCH = 3
M_BATCH = 300

### 데이터 준비하기 ###
# 데이터 읽어들이기 : keras가 제공하는 7만개의 패션 아이템 이미지
# 28 X 28 흑백 이미지, 4분할되어 있음
# 결과 데이터 타입 : numpy의 n-차원 행렬임(배열)
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# 데이터 타입 확인
print('학습용 입력 데이터 타입 : ', X_train.shape)
print('학습용 출력 데이터 타입 : ', Y_train.shape)
print('테스트용 입력 데이터 타입 : ', X_test.shape)
print('테스트용 출력 데이터 타입 : ', Y_test.shape)

# 샘플 이미지 출력 확인
print(X_train[0])
plt.imshow(X_train[0], cmap='gray')
plt.show()
print('샘플 이미지 라벨 : ', Y_train[0])

# 입력 데이터 스케일링 처리 (데이터 전처리)
# 픽셀에 기록된 0~255 범위의 값을 0~1 범위로 바꿈
X_train = X_train / 255.0
X_test = X_test / 255.0

# 스케일링 후 확인
print(X_train[0])
plt.imshow(X_train[0], cmap='gray')
plt.show()

# channel 추가함 => 케라스 CNN에서는 4차원 정보가 필요함
# 4차원 행렬 == Tensor 타입이라고 함
# 1차원 배열 == Vector 타입이라고 함
# 2차원 행렬 == pandas.DataFrame == numpy 2d Array(matrix)

# Tensor 타입으로 바꾸기 위해, 훈련용 입력데이터 추출함
train = X_train.shape[0]    # 60000 추출
X_train = X_train.reshape(train, 28, 28, 1) # 차원 늘림
test = X_test.shape[0] # 10000 추출
X_test = X_test.reshape(test, 28, 28, 1) # 차원 늘림

# Tensor(4d matrix)로 처리후 확인
print(X_train[0])
plt.imshow(X_train[0], cmap='gray')
plt.show()

# 출력 데이터(라벨 정보) : One-hot encoding
# 찾고자 하는 특징 라벨이면 1, 아니면 0 처리하는 것
print('One-hot encoding 전: ', Y_train[0])   # 10진수 분류값
Y_train = to_categorical(Y_train, 10)
print('One-hot encoding 후: ', Y_train[0])
# 학습 후 나온 예측결과값이 9.0이 나오면 라벨 9와 일치하지 않게 됨
# 이런 문제를 막기 위해 one-hot 인코딩으로 라벨값을 바꿈

Y_test = to_categorical(Y_test, 10)

print('학습용 출력데이터 모양 : ', Y_train.shape)
print('평가용 출력데이터 모양 : ', Y_test.shape)

### 인공신경망 구축 ###
# CNN 구현 (순차적 방법)
model = Sequential()
# Sequential 모델 : 각 레이어에 정확히 하나의 입력 텐서와
#       하나의 출력 텐서가 있는 일반 레이어 스택을 구성하는 모델임

# 입력층
model.add(InputLayer(input_shape=(28, 28, 1)))

# 첫 번째 합성곱 블럭
# kernel_size : filter size 를 의미함, 2: (2,2), 5 : (5,5)
# 32 : channel 갯수임
model.add(Conv2D(32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))

# 두번째 합성곱 블럭
model.add(Conv2D(64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))

# fully-connected 층으로 마무리
model.add(Flatten()) # 1차원 변환
model.add(Dense(128, activation='relu'))
# 가중치를 적용해서 특징값 찾음
model.add(Dense(10, activation='softmax'))
# softmax : 클래스 분류 문제에 사용하는 활성화 함수임

print('CNN 요약')
model.summary()

### 인공신경망 학습 ###

# 최적화 함수와 손실 함수 지정
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

begin = time()
print('CNN 학습 시작')

model.fit(X_train, Y_train, epochs=M_EPOCH, batch_size=M_BATCH, verbose=1)

end = time()
print('총 학습 시간 : {:.1f}초'.format(end - begin))

### 인공신경망 모델 평가
# CNN 평가
_, score = model.evaluate(X_test, Y_test, verbose=1)

print('최종 정확도 : {:.2f}%'.format(score * 100))

# CNN 테스트 데이터 적용 예측값 확인하기
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
trust = np.argmax(Y_test, axis=1)

# 혼동 행렬 (정확도, 정밀도, 재현율, F1 점수라고 함)
# 모델의 성능을 평가할 때 사용하는 지표
print('혼동 행렬')
print(confusion_matrix(trust, pred))
# 평가데이터와 결과데이터 사용
# 실제값과 예측값이 일치할 때 값 출력됨(대각선)