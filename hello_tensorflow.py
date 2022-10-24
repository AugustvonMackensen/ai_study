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

# 2. 손글씨 숫자 분류하기
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