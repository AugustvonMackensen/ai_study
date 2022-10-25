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
M_EPOCH = 500 # 학습할 횟수
# M_EPOCH = 0 # 오차율의 차이
# M_EPOCH = 2000 # 오차율의 차이
M_BATCH = 64    # 가중치를 적용할 샘플의 갯수
# M_BATCH = 16    # 학습시간 차이
# M_BATCH = 354    # 학습시간 차이

### 데이터 준비하기 ###
# 데이터 파일 읽기 => 읽은 결과는 pandas의 데이터프레임 형식임
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