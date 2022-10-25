# housing.py
# 보스톤 주택 가격 예측 딥러닝 모델 테스트

# 필요 패키지 확인 : tensorflow, matplotlib, seaborn
# pip install -U seaborn

# 파이썬 패키지 임포트
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from keras.models import Sequential
from keras.layers import Dense

# 오류 발생시 아래와 같이 수정함
# from tensorflow.keras.model import Sequential
# from tensorflow.keras.layers import LSTM.Dropout.Dense

# sklearn 패키지 추가 설치함
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 하이퍼 파라미터 (전역변수)
M_EPOCH = 500 # 학습할 횟수 : 오차율 (0.11)
# M_EPOCH = 0 # 오차율의 차이 : 증가(1)
# M_EPOCH = 2000 # 오차율의 차이 없음
M_BATCH = 64    # 가중치를 적용할 샘플의 갯수
# M_BATCH = 16    # 학습시간 차이 : 늘어남
# M_BATCH = 354    # 학습시간 차이 : 줄어듦

### 데이터 준비하기 ###
# 데이터 파일 읽기 => 읽은 결과는 pandas의 데이터프레임 형식임
# pandas dataframe == 표(table) == numpy 2d Array
# dataframe : columns X rows, 2d Array(행열) : rows X columns
raw = pd.read_csv('housing.csv')
heading = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','LSTAT','MEDV']

# 읽어들인 데이터 확인 : 위에서 아래로 10줄만 확인한다면
print('원본 데이터 샘플 10개 출력 확인')
print(raw.head(10))

print('원본 데이터 통계')
print(raw.describe())

### 실행 확인 시 .... cuda .... GPU ..... 2줄의 빨간색 문장이 나타나면 에러
### CUDA Toolkit 11.0 설치 후 파이썬(파이참) 종료후
### 재실행하면 없어짐!
### 참고 : https://leunco.tistory.com/13

# 입력 데이터 정규화 처리(입력데이터를 값의 분포를 정리하는 것)
# 데이터 전처리에 해당됨

# Z-점수 정규화(Z-Score Normalization) 처리
# 어떤 데이터가 표준정규분포에 해당되도록 값을 바꾸는 것
# Z = (X-평균) / 표준편차
# 계산에 사용할 입력데이터(x)를 사용해서 나온 결과타입은
# numpy의 n-차원 행렬 형식이다.
scaler = StandardScaler() # 표준정규분포로 만드는 객체 생성
# 표준편차(std : 평균값의 차이)를 1과 가깝게 바꾸는 정규화 객체임.
Z_data = scaler.fit_transform(raw) # 정규화처리가 된 행렬 반환
# 컬럼라벨을 제외한 데이터로만 구성된 행렬(matrix) 만들기함

# numpy에서 pandas로 전환(matrix --> DataFrame)
# header (컬럼라벨) 정보 사용해서 복구함
Z_data = pd.DataFrame(Z_data, columns=heading)

# 정규화된 데이터 출력 확인
print('정규화된 데이터 샘플 10개 확인')
print(Z_data.head(10))

print('정규화된 데이터 통계')
print(Z_data.describe())

# 상자 그림(boxplot) 출력 확인
sns.set(font_scale=1)
sns.boxplot(data=Z_data, palette='dark')
# sns.boxplot(data=raw, palette='dark')
plt.show()

### data preprocessing (데이터 전처리) ------------------------------------

# 데이터 입력과 출력으로 분리
print('분리전 데이터 형태 : ', Z_data.shape)
X_data = Z_data.drop('MEDV', axis=1)
# 'MEDV' 를 제외한 모든 값(12열)
Y_data = Z_data['MEDV'] # 'MEDV' 컬럼의 값들(1열)

# 정규화하지 않은 원래의 입력데이터로 학습데이터를 만들 경우
# X_data = raw.drop('MEDV', axis=1) # 'MEDV'를 제외한 모든값(12열)
# Y_data = raw['MEDV'] # 'MEDV'(집값) 컬럼의 값들(1열)

# 데이터 분리 : 학습용, 테스트용
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3)
print('학습용 입력데이터 형태 : ', X_train.shape)
print('학습용 출력데이터 형태 : ', Y_train.shape)
print('테스트용 입력데이터 형태 : ', X_test.shape)
print('테스트용 출력데이터 형태 : ', Y_test.shape)

### 인공신경망 모델 설계 구현(구축) ###

# 케라스 DNN구현
model = Sequential() # 모델 객체 생성
input = X_train.shape[1] # 입력레이어 뉴런 12개 지정
model.add(Dense(200, input_dim=input, activation='relu'))
model.add(Dense(1000, activation='relu'))
# 히든 레이어 2개 추가함

# 500개의 뉴런을 가진 히든 레이어 2개 추가해 봄
# 결론 : 학습시간 늘어남, 오차율 변화없음 확인
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))

# 출력층
model.add(Dense(1))

print('DNN 모델 요약')
print(model.summary())

# 인공신경망 모델 구축 ------------------------------------------

### 인공신경망 모델을 이용해서 학습시키기 ###

# 최적화 함수(가중치 보정)와 손실함수(오차율) 지정
model.compile(optimizer='sgd', loss='mse')

print('DNN 딥러닝 학습 시작')
begin = time()

model.fit(X_train, Y_train, epochs=M_EPOCH, batch_size=M_BATCH, verbose=0)
# verbose mode : 0(silent : 안보임), 1(progress bar : 진행률), 2(one line per epoch) : 학습 한 줄씩 표시
#                   학습의 진행을 표시하는 모드임
# batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 설정
# epochs : 전체 데이터셋을 몇 번 반복 학습할지 횟수 지정

end = time()
print('총 딥러닝 학습 시간 : {:.1f} 초'.format(end - begin))

### 인공신경망 모델 평가 및 활용 ###
loss = model.evaluate(X_test, Y_test, verbose=0)
print('DNN 평균 제곱 오차(MSE) : {:.2f}'.format(loss))

# 신경망 활용 및 산포도 출력
pred = model.predict(X_test)
sns.regplot(x=Y_test, y=pred) # 회귀선 그래프

plt.xlabel('Actual Values') # 실제 주택가격
plt.ylabel('Predicted Values') # 예상 주택가격
plt.show()

# 예측된 주택가격(MEDV) 확인
print(pd.DataFrame(pred).head(10))
