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
# Flatten() : 읽은 2차원(가로x세로) 데이터를 1차원 데이터로
#       바꾸는 layer임
model.add(layers.Flatten(input_shape=(28, 28))) # 784 개.
# Dense() : 추출된 데이터를 하나의 레이어로 모아, 원하는 차원으로
#       축소시키는 layer 임
model.add(layers.Dense(128, activation='relu'))
# Dropout() : 서로 연결된 연결망(layer)에서 0에서 1사이의 확률로
#           뉴런(신경망)을 제거(drop)하는 기법을 적용한 layer 임
model.add(layers.Dropout(0.2))  # 신경망을 20%로 줄임
model.add(layers.Dense(10, activation='softmax'))

# 인공신경망 요약 내용 보기
model.summary()

# 구축된 모델을 준비된 테이터로 학습시키기
# 인공신경망 학습 환경 설정
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델을 적용해서 학습 시킴 (머신 러닝) : fit
# fit(입력데이터, 출력데이터, 학습횟수)
model.fit(X_train, Y_train, epochs=5)

# 모델을 평가
# evaluate(테스트용 입력데이터, 테스트용 출력데이터) 사용
model.evaluate(X_test, Y_test)

# 인공신경망 사용(적용) 테스트 : 예측 테스트
pick = X_test[0].reshape(1, 28, 28) # 채널, 가로, 세로
pred = model.predict(pick) # 모델 적용
answer = tf.argmax(pred, axis=1) # 가장 큰 값의 인덱스 리턴

print('원본 예측 결과 확인', pred)
print('해석 결과 : ', answer)
print('정답 : ', Y_test[0])

# 학습한 모델 저장하기
# from sklearn.externals import joblib
import joblib
joblib.dump(model,'digits.pkl')
# 저장된 모델을 읽어들이기 : joblib.load()

# 또는 텐서플로우에서는
model.save('./mnist_model')

import cv2

# 파이썬에서 저장된 숫자 이미지 읽어들임
test_img = cv2.imread('number.png')
# 컬러 이미지를 그레이스케일로 변경함
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
test_img = cv2.resize(test_img, (28, 28))
test_img = test_img.reshape(1, 28, 28)

# 위에서 학습된 모델에 적용해 봄
result = model.predict(test_img)
print(result)
print(result[0])
print(tf.argmax(result, axis=1))



