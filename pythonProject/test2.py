from pydoc import describe

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import matplotlib.pyplot as plt

# 윈도우 크기 및 몇 초 뒤 예측
WINDOWSIZE = 5
PRE_INDEX = 3

# 학습 및 테스트 데이터 로드
train_data = pd.read_csv('data/_Merged_Data_data_set_00000.csv',
                         header=None,
                         names=['Sec', 'Temp1', 'Press1', 'Temp2', 'Press2', 'Accel', 'Gas'],
                         index_col='Sec')

test_data = pd.read_csv('data/_Merged_Data_data_set_01500.csv',
                        header=None,
                        names=['Sec', 'Temp1', 'Press1', 'Temp2', 'Press2', 'Accel', 'Gas'],
                        index_col='Sec')

print(type(train_data))
print(train_data.head())
print(train_data.info())
print(test_data.info())

# 데이터 정규화
train_min = train_data.min()
train_max = train_data.max()
train_data = (train_data - train_min) / (train_max - train_min)

test_min = test_data.min()
test_max = test_data.max()
test_data = (test_data - test_min) / (test_max - test_min)

print(type(train_min))
print(type(train_max))
print(train_data.head())
print(train_data.info())
print(test_data.info())
print(describe(test_data))
print(describe(train_data))

# 슬라이딩 윈도우 생성 (학습 데이터)
X_train, Y_train = [], []
i = 0
while i < len(train_data) - PRE_INDEX - WINDOWSIZE:
    X_train.append(train_data.iloc[i:i + WINDOWSIZE].values)  # WINDOWSIZE 만큼의 데이터
    Y_train.append(train_data.iloc[i + WINDOWSIZE + PRE_INDEX].values)  # PRE_INDEX 뒤의 출력값
    i += 1

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# 슬라이딩 윈도우 생성 (테스트 데이터)
X_test, Y_test = [], []
i = 0
while i < len(test_data) - PRE_INDEX - WINDOWSIZE:
    X_test.append(test_data.iloc[i:i + WINDOWSIZE].values)
    Y_test.append(test_data.iloc[i + WINDOWSIZE + PRE_INDEX].values)
    i += 1

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Conv2D 입력 형식으로 변환 (4D 텐서: [샘플 수, 높이, 너비, 채널])
X_train = X_train.reshape(-1, WINDOWSIZE, X_train.shape[2], 1)
X_test = X_test.reshape(-1, WINDOWSIZE, X_test.shape[2], 1)

print(f"Training Data Shape: X={X_train.shape}, Y={Y_train.shape}")
print(f"Testing Data Shape: X={X_test.shape}, Y={Y_test.shape}")

# 모델 구성
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(WINDOWSIZE, 6, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='sigmoid'))  # 출력 노드 수는 Y_train의 열 개수
model.summary()

# 모델 컴파일
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 모델 학습
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=32, verbose=1)

# 모델 평가
loss, mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# 학습 과정 시각화
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Loss')  # 손실 함수 (Loss)
plt.plot(history.history['mae'], label='MAE')   # 평균 절대 오차 (MAE)
plt.xlabel('Epochs')  # x축: 에포크
plt.ylabel('Value')   # y축: 값
plt.legend()          # 범례 표시
plt.title('3sec Prediction Model')  # 그래프 제목
plt.grid()            # 격자 표시
plt.tight_layout()    # 레이아웃 자동 조정
plt.show()

y_pred = model.predict(X_test)  # 모델 예측값

# 센서 이름 정의
sensor_names = ["Temp1", "Press1", "Temp2", "Press2", "Accel", "GasLeak"]

plt.figure(figsize=(10, 6))

# 모든 센서 데이터를 같은 그래프에 시각화
for i in range(6):  # 6개의 센서 데이터 반복
    plt.plot(y_pred[:, i], label=f"{sensor_names[i]} Predicted")  # 예측값

# 그래프 설정
plt.xlabel("Sample Index")  # x축: 샘플 인덱스
plt.ylabel("Rate (0~1)")  # y축: 정규화된 값
plt.title("Sensor Predictions (Predicted Only)")  # 그래프 제목
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)  # 범례 위치 조정
plt.grid()  # 격자 추가
plt.tight_layout()  # 레이아웃 조정
plt.show()

# 실제값 그래프
plt.figure(figsize=(10, 6))

# 실제값 그래프를 센서별로 시각화 (range 사용)
for i in range(6):  # 센서 개수만큼 반복
    plt.plot(Y_test[:, i], label=f"{sensor_names[i]} True")  # 실제값 시각화

# 그래프 설정
plt.xlabel("Sample Index")  # x축 레이블
plt.ylabel("Rate (0~1)")  # y축 레이블
plt.title("Sensor True Values")  # 그래프 제목
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)  # 범례 설정
plt.grid()  # 격자 추가
plt.tight_layout()  # 레이아웃 자동 조정
plt.show()

# 오차 계산
errors = np.abs(Y_test - y_pred)  # 절대 오차 계산

# 센서별 평균, 최소, 최대 오차 계산
mean_error = np.mean(errors, axis=0)  # 평균 오차
min_error = np.min(errors, axis=0)    # 최소 오차
max_error = np.max(errors, axis=0)    # 최대 오차


# 결과 출력
print("=== Sensor-wise Errors and Accuracy ===")
for i, sensor_name in enumerate(sensor_names):
    print(f"{sensor_name}:")
    print(f"  Mean Error: {mean_error[i]:.6f}")
    print(f"  Min Error: {min_error[i]:.6f}")
    print(f"  Max Error: {max_error[i]:.6f}")
    print("-" * 40)

# 평균 오차와 정확도 요약 출력
print("\nOverall:")
print(f"Average Mean Error: {np.mean(mean_error):.6f}")

# 추가: Temp1 센서의 오차 그래프
temp1_true = Y_test[:, 0]  # Temp1의 실제값
temp1_pred = y_pred[:, 0]  # Temp1의 예측값
temp1_error = temp1_pred - temp1_true  # Temp1 오차 계산

# Temp1 센서의 오차 그래프 시각화
plt.figure(figsize=(8, 6))
plt.plot(temp1_error, label="3sec Error", color="red")  # 3초 모델의 Temp1 오차
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)  # 0 기준선

# 그래프 설정
plt.title("Error of 3sec Prediction Model (Temp1)")
plt.xlabel("Samples")
plt.ylabel("Error Rate")
plt.ylim(-1.0, 1.0)  # 오차 범위
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

