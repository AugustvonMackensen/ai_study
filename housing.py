# housing.py
# 보스톤 주택 가격 예측 딥러닝 모델 테스트

# 필요 패키지 확인 : tensorflow, matplotlib, seaborn
# pip install -U seaborn

# 파이썬 패키지 임포트
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from keras.models import Sequential
from keras.layers import Dense

# sklearn 패키지 추가 설치함
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split