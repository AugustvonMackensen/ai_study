# airline.py
# RNN (순환 신경망)을 이용한 항공 여행자수 예측하기
# CNN-RNN : 사진이미지에 캡션(설명)달기 AI 프로그램 대회

# RNN(Recurrent Neural Network, 순환 신경망)
# 앞서의 신경망들은 모두 은닉층에서 활성화함수(activation)를 가진 값들을
# 출력층으로 내보내는 구조였음. (feed-forward 신경망)
# 입력층 --> 히든레이어(활성화함수) --> 출력층 방향으로 진행

# RNN은 은닉층의 노드에서 활성화함수를 통해 나온 결과값을
# 출력층 방향으로 보내면서, 다시 다음 은닉층 노드의 입력값으로도
# 보내는 특징이 있음
# 이전 노드의 값을 기억하는 셀(메모리 셀, RNN 셀이라고 함)
# ==> LSTM(Long Short Term Memory) 셀

# 필요 패키지, 함수 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import  InputLayer, Dense, LSTM

# 하이퍼 파라미터
M_PAST = 12 # 데이터 분할 갯수
M_SPLIT = 0.8 # 훈련데이터 분할 비율(80% 훈련용, 20% 테스트용)
M_UNIT = 300
M_SHAPE = (M_PAST, 1) # (입력 12, 출력 1) : 2차원 선언, 입력데이터

M_EPOCH = 300
M_BATCH = 64 # 가중치

# 부동소수점형 실수값 출력시 소숫점 3번째 지정
np.set_printoptions(precision=3)

### 데이터 준비 ###
# 제공된 arline.csv 파일 읽기
# 읽을 결과가 pandas의 DataFrame
raw = pd.read_csv('airline.csv', header=None, usecols=[1])
# usecols : 불러올 컬럼의 인덱스 순번이나 컬럼이름 지정
# usecol=[1] : 데이터에서 두번째 열(컬럼) [1]

# 데이터프레임 시각화 (시계열 데이터 : 시간과 연결된 데이터)
# plt.plot(raw) # 년-월 비행기 이용자수
# plt.show()

# 원본 데이터 10개 출력 확인
# print('원본 데이터 샘플 12개')
# print(raw.head(12))

# print('원본 데이터 통계 정보')
# print(raw.describe())

# MinMax 데이터 정규화
# MinMaxScaler 는 값의 범위를 조정할 때 사용함
# 모든 데이터를 0과 1 사이의 값으로 조정함
# 최솟값(min)이 0이고, 최대값(max)이 1이 됨
scaler = MinMaxScaler()
s_data = scaler.fit_transform(raw)

# 스케일링 후의 데이터 타입 확인
# print('원본 데이터 타입 : ', type(raw))
# print('스케일링 후 데이터 타입 : ', type(s_data)) # <class 'numpy.ndarray'>

# 정규화된 데이터 출력 확인 : array 를 DataFrame 으로 변환함
df = pd.DataFrame(s_data)
# print('정규화된 데이터 샘플 출력 : ', df.head(12))
# print('정규화된 데이터 통계 : ', df.describe())

# 입력 데이터 분할
# 데이터 13개씩 각 묶음으로 데이터 분할(folding)
# 13개(입력 : 12개, 출력 1) : 입력데이터 shape 지정되어 있음
# 묶음 타입은 파이썬의 list
bundle = []
for i in range(len(s_data) - M_PAST):
    bundle.append(s_data[i: i + M_PAST + 1])
    
# 데이터 분할 결과 확인
# print('총 묶음 갯수 : ', len(bundle))
# print('0번째 묶음 : ', bundle[0])
# print('1번째 묶음 : ', bundle[1])

# 분할 데이터 타입 확인
# print('분할 데이터 타입 : ', type(bundle))

# list를 numpy 의 array 로 바꿈
bundle = np.array(bundle)
# print('변환 데이터 타입 : ', type(bundle))
# print('입력 데이터 모양 : ', bundle.shape)

# 각 묶음 안의 13개의 값을 입력용, 출력용으로 분할
X_data = bundle[:, 0:M_PAST] # 모든 행의 0열~11열(12개) 슬라이싱
Y_data = bundle[:, -1] # 모든 행의 마지막 열(12열, 13번째)

# 132개 묶음을 학습용(0.8)과 테스트용(0.2)로 분리(split)
split = int(len(bundle) * M_SPLIT)
# print('분리 갯수 : ', split)
X_train = X_data[ :split] # X_data 에서 80%가 학습데이터로 분리
X_test =X_data[split: ] # X_data 에서 20%가 평가데이터로 분리

Y_train = Y_data[ :split] # Y_data 에서 80%가 학습데이터로 분리
Y_test =Y_data[split: ] # Y_data 에서 20%가 평가데이터로 분리

# 최종 데이터 확인
# print('학습용 입력데이터 모양 : ', X_train.shape)
# print('학습용 출력데이터 모양 : ', Y_train.shape)
# print('평가용 입력데이터 모양 : ', X_test.shape)
# print('평가용 출력데이터 모양 : ', Y_test.shape)

### RNN 인공신경망 구축
# RNN 구현
# 케라스 RNN은 2차원(2개) 입력만 허용함 :
# 계산된 입력값이 사라지는 문제를 해결하는 방법 (2개 전달)
model = Sequential()
model.add(InputLayer(input_shape=M_SHAPE))
model.add(LSTM(M_UNIT))
model.add(Dense(1, activation='sigmoid'))

# print('RNN 요약')
# model.summary()

### 인공신경망 학습 ###

# 최적화함수와 손실함수 지정
model.compile(optimizer='rmsprop', loss='mse')
# 최적화 알고리즘 : adam, momenturm, adagrad, rmsprop 등
# rmsprop : 기울기를 단순 누적하지 않고 지수 가중 이동 평균을
#           계산해서 최신 기울기들이 더 크게 반영되도록 하는 알고리즘

# 손실함수 : 실제값과 예측값의 차이를 계산하는 함수
# mse(평균 제곱 오차), cross-entropy(cee, 교차 엔트로피 오차)

print('RNN 학습 시작')
begin = time()

model.fit(X_train, Y_train, epochs=M_EPOCH, batch_size=M_BATCH, verbose=0)

end = time()

print('총 학습시간 : {:.1f}초'.format(end - begin))

### 인공신경망 평가 ###
# RNN 평가
loss = model.evaluate(X_test, Y_test, verbose=0)
print('최종 MSE 손실값 : {:.3f}'.format(loss))

# RNN 예측값 확인 : 테스트용 데이터 사용 => 실제 데이터 적용해도 됨
pred = model.predict(X_test)

# (0, 1) 값을 원래 값으로 변환 : inverse_transform() 사용
pred = scaler.inverse_transform(pred)
# 1차원 배열로 바꾸고 , 정수값으로 변환함
pred = pred.flatten().astype(int)
print('예측된 결과 데이터 : ', pred)

# 예측값과 비교할 평가용 출력 데이터도 0~1 값에서 실제 값으로 변환
truth = scaler.inverse_transform(Y_test)
truth = truth.flatten().astype(int)
print('비교할 실제 데이터 : ', truth)

# 선 그래프(line plot) 로 시각화 처리
axes = plt.gca()
axes.set_ylim([0, 650])

sns.lineplot(data=pred, label='pred', color='blue')
sns.lineplot(data=truth, label='truth', color='red')

plt.show()