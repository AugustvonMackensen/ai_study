# hello_tensor.py

# 텐서플로우 불러오기
import tensorflow as tf

# 텐서플로우로 문장 출력해보기
msg = tf.constant('Hello Tensorflow')
tf.print(msg)

# MNIST (손글씨 숫자(0~9) 이미지) 데이터를 훈련 데이터로
# 사용한 DNN 학습 예제
# 빅데이터, AI, 패키지들이 제공하고 있음 : 가져다 사용하면 됨

# 1. MNIST 손글씨 숫자 이미지 데이터셋 불러오기
# import tensorflow.keras
mnist = tf.keras.datasets.mnist

# 2. 제공된 이미지와 데이터 확인 --------------------------------------
# MNIST 4분할로 데이터 처리하기
# 훈련용 데이터와 테스트용 데이터로 분류하는 작업
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print('학습용 입력 데이터 모양 : ', X_train.shape)
print('학습용 출력 데이터 모양 : ', Y_train.shape)
print('훈련용 입력 데이터 모양 : ', X_test.shape)
print('훈련용 출력 데이터 모양 : ', Y_test.shape)

# 손글씨 숫자 이미지 데이터 원본 출력해보기
# matplotlib 사용시 패키지 설치 확인하고 설치함
# pip install --U matplotlib

import matplotlib.pyplot as plt
# 라이브러리 중복관련 에러 발생하면 아래 코드 추가함
import os
os.environ['TP_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ---------------------------------------------------

plt.imshow(X_train[0], cmap='gray')
plt.show()

# 손글씨 숫자 이미지의 데이터 정보 확인
print('첫번째 학습용 데이터 입력값 : ', X_train[0])
print('첫번째 학습용 데이터 출력값 : ', Y_train[0])

# 3. 손글씨 숫자 분류하기 연습 ---------------------------------------
# 이미지 데이터 [0,1]로 스케일링(Scalling) ==> 수치값의 축소
# 0~255 사이의 값을 0~1 사이의 값으로 값의 범위를 축소함.
# 각 픽셀의 색상값이 255이면 그대로 1.0으로,
# 각 픽셀의 색상값이 0이거나 255가 아니면 0.xxx로 바꾸는 것.
X_train = X_train / 255.0 # 나누기한 몫 : 실수형
X_test = X_test / 255.0

# 스케일링 후 이미지 확인 : 이미지 변화없음.
plt.imshow(X_train[0], cmap='gray')
plt.show()

# 스케일링 후 픽셀 색상 데이터 확인
print('첫번째 학습용 데이터 입력값 : ', X_train[0])

# 인공신경망 모델링 구축함
# 모델 메소드 : Sequential()
# 모델 객체 준비
model = tf.keras.models.Sequential()
# 레이어 변수 준비
layers = tf.keras.layers

# 모델에 레이어 구성함 : 설계 단계임
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

# 인공신경망 요약 내용 보기
model.summary()