import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import matplotlib.pyplot as plt

# 윈도우 크기 및 몇 초 뒤 예측
WINDOWSIZE = 10
PRE_INDEX = 10

# 사용자 지정: 불러올 학습 데이터 파일 개수
num_train_files = 1000
# 수정 가능 (불러올 파일 개수)

# 학습 데이터 로드
train_data_list = []

# _Merged_Data 디렉토리 내 파일을 순회하며 CSV 파일만 읽기
base_dir = '_Merged_Data'
file_count = 0

for file_name in sorted(os.listdir(base_dir)):
    if file_name.endswith('.csv') and file_count < num_train_files:
        file_path = os.path.join(base_dir, file_name)
        temp_data = pd.read_csv(file_path,
                                header=None,
                                names=['Sec', 'Temp1', 'Press1', 'Temp2', 'Press2', 'Accel', 'Gas'],
                                index_col='Sec')
        train_data_list.append(temp_data)
        file_count += 1

# 학습 데이터 결합
train_data = pd.concat(train_data_list)

# 테스트 데이터 로드
test_data = pd.read_csv('data/_Merged_Data_data_set_01500.csv',
                        header=None,
                        names=['Sec', 'Temp1', 'Press1', 'Temp2', 'Press2', 'Accel', 'Gas'],
                        index_col='Sec')

# 데이터 정보 확인
print("Train Data Info:")
print(train_data.info())
print("Test Data Info:")
print(test_data.info())

# 데이터 분석 추가
# 1. 데이터 요약 통계
print("Train Data Summary:")
print(train_data.describe())

print("Test Data Summary:")
print(test_data.describe())

# 2. 데이터 분포 시각화
# print("Visualizing Train Data Distributions...")
# train_data.hist(bins=50, figsize=(12, 10))
# plt.suptitle("Train Data Distributions")
# plt.show()
#
# print("Visualizing Test Data Distributions...")
# test_data.hist(bins=50, figsize=(12, 10))
# plt.suptitle("Test Data Distributions")
# plt.show()

# 3. 결측치 확인
print("Train Data Missing Values:")
print(train_data.isnull().sum())

print("Test Data Missing Values:")
print(test_data.isnull().sum())

# 데이터 정규화
train_min = train_data.min()
train_max = train_data.max()
train_data = (train_data - train_min) / (train_max - train_min)

test_min = test_data.min()
test_max = test_data.max()
test_data = (test_data - test_min) / (test_max - test_min)

# 데이터 분석 추가
# 센서 데이터 변환 정보 계산 및 시각화

# 입력자료의 스케일 변경 정보 출력

# 센서 이름과 단위 정보 정의
sensor_names = ["온도센서1", "압력센서1", "온도센서2", "압력센서2", "진동센서", "가스누출센서"]
units = ["℃", "bar", "℃", "bar", "m/s2", "ppm"]

# 최소값 및 최대값 계산
original_min = train_min.values  # 학습 데이터의 실제 최소값
original_max = train_max.values  # 학습 데이터의 실제 최대값

# 변환된 데이터의 최소값 및 최대값 (정규화 값)
transformed_min = train_data.min().values  # 정규화된 데이터 최소값 (0.0이어야 함)
transformed_max = train_data.max().values  # 정규화된 데이터 최대값 (1.0이어야 함)

# DataFrame 생성
sensor_df = pd.DataFrame({
    "센서": sensor_names,
    "최소값": [f"{original_min[i]:.2f} {units[i]}" for i in range(len(sensor_names))],
    "최대값": [f"{original_max[i]:.2f} {units[i]}" for i in range(len(sensor_names))],
    "변환 최소값": [f"{transformed_min[i]:.2f}" for i in range(len(sensor_names))],
    "변환 최대값": [f"{transformed_max[i]:.2f}" for i in range(len(sensor_names))]
})

# 콘솔 출력
print(sensor_df.to_string(index=False))


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
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=512, verbose=1)

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
plt.title(f'{PRE_INDEX}sec Prediction Model')  # 그래프 제목
plt.grid()            # 격자 표시
plt.tight_layout()    # 레이아웃 자동 조정
plt.show()

y_pred = model.predict(X_test)  # 모델 예측값

# 센서별 정확도 및 표준편차 계산
sensor_errors = np.abs(Y_test - y_pred)  # 절대 오차
sensor_accuracies = 100 * (1 - np.mean(sensor_errors, axis=0))  # 센서별 정확도
sensor_std_devs = np.std(sensor_errors, axis=0) * 100  # 센서별 표준편차

# 최소 정확도 및 해당 센서의 표준편차 추출
min_accuracy = np.min(sensor_accuracies)  # 최소 정확도
min_std_dev = sensor_std_devs[np.argmin(sensor_accuracies)]  # 최소 정확도의 표준편차
min_sensor_name = sensor_names[np.argmin(sensor_accuracies)]  # 최소 정확도 센서 이름


# 센서 이름 정의
sensor_names = ["Temp1", "Press1", "Temp2", "Press2", "Accel", "GasLeak"]

plt.figure(figsize=(10, 6))

# 모든 센서 데이터를 같은 그래프에 시각화
for i in range(6):  # 6개의 센서 데이터 반복
    plt.plot(y_pred[:, i], label=f"{sensor_names[i]} Predicted")  # 예측값

# 그래프 설정
plt.xlabel("Sample Index")  # x축: 샘플 인덱스
plt.ylabel("Rate (0~1)")  # y축: 정규화된 값
plt.title(f"{PRE_INDEX}ec Sensor Predictions (Predicted Only)")  # 그래프 제목
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
plt.title(f"{PRE_INDEX}ec Sensor True Values")  # 그래프 제목
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

# 특정 센서 오차 범위 시각화 (예: Temp1)
sensor_index = 0  # Temp1의 인덱스
true_values = Y_test[:, sensor_index]
predicted_values = y_pred[:, sensor_index]
errors = predicted_values - true_values

plt.figure(figsize=(10, 6))
plt.plot(errors, label=f"{PRE_INDEX}sec Error", color="red")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Samples")
plt.ylabel("Error Rate")
plt.title(f"Error Range for {sensor_names[sensor_index]} ({PRE_INDEX}sec Model)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 모델 평가 후 정확도 및 표준편차 계산
# 예측값과 실제값의 절대 오차 계산
errors = np.abs(Y_test - y_pred)

# 평균 정확도 계산 (여기서는 간단히 1 - 평균 오차로 정의)
accuracy = (1 - np.mean(errors)) * 100  # 정규화된 값이므로 100% 기준
std_dev = np.std(errors) * 100  # 표준편차 (백분율로 변환)

# 그래프 생성
model_label = f'{PRE_INDEX}s'  # PRE_INDEX 초 모델 이름
plt.figure(figsize=(6, 4))
bars = plt.bar([model_label], [accuracy], color='gray', alpha=0.8, label='Accuracy')  # 막대그래프
plt.errorbar([model_label], [accuracy], yerr=[std_dev], fmt=' ', ecolor='red', elinewidth=2, capsize=5, label='Std. Dev')  # 표준편차

# 정확도 텍스트 표시
plt.text(0, accuracy + 0.5, f'{accuracy:.2f}', ha='center', va='bottom', fontsize=10)

# 그래프 설정
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy [%]', fontsize=12)
plt.title(f'Prediction Model Accuracy ({PRE_INDEX}s)', fontsize=14)
plt.ylim(0, 100)
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 그래프 출력
plt.show()

# 정확도 그래프 시각화
plt.figure(figsize=(6, 4))
bar_positions = [0]  # 하나의 막대 위치
bar_heights = [min_accuracy]  # 최소 정확도
bar_errors = [min_std_dev]  # 표준편차

# 막대그래프 생성
plt.bar(bar_positions, bar_heights, yerr=bar_errors, capsize=5, color='gray', alpha=0.8, edgecolor='black',
        error_kw={'ecolor': 'red', 'elinewidth': 2})

# X축 레이블 설정
plt.xticks(bar_positions, [f"{min_sensor_name} Model"])  # 최소 정확도 센서 이름 표시
plt.ylim(0, 100)  # 정확도 범위

# 정확도 텍스트 표시
for i, (height, err) in enumerate(zip(bar_heights, bar_errors)):
    plt.text(i, height + 2, f"{height:.2f}%", ha='center', color='red', fontsize=10)

# 그래프 설정
plt.xlabel("Sensor")
plt.ylabel("Accuracy [%]")
plt.title("Minimum Sensor Accuracy with Std. Dev")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()