import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import matplotlib.pyplot as plt

# === 사용자 정의 변수 ===
WINDOWSIZE = 30  # 슬라이딩 윈도우 크기
PREDICTION_TIMES = [3, 5, 10, 30, 60, 120]  # 다양한 예측 시간 (초)
BATCH_SIZE = 128  # 배치 크기
NUM_FILES_TO_LOAD = 1000  # 한 번에 처리할 파일 개수

# === 데이터 로드 및 병합 ===
def load_data(file_paths, window_size, prediction_time):
    """파일 리스트를 읽고 슬라이딩 윈도우 데이터 생성"""
    X, Y = [], []
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None, names=['Sec', 'Temp1', 'Press1', 'Temp2', 'Press2', 'Accel', 'Gas'], index_col='Sec')
        data = (data - data.min()) / (data.max() - data.min())  # 정규화
        for i in range(len(data) - window_size - prediction_time):
            X.append(data.iloc[i:i + window_size].values)
            Y.append(data.iloc[i + window_size + prediction_time].values)
    X = np.array(X).reshape(-1, window_size, len(data.columns), 1)  # Conv2D 입력 형식
    Y = np.array(Y)
    return X, Y

# 학습 데이터 폴더
data_folder = '_Merged_Data'
file_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

# === 모델 정의 ===
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(6, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 결과 저장용 딕셔너리
results = {}

# === 학습 및 평가 ===
for pred_time in PREDICTION_TIMES:
    print(f"\n=== Processing for Prediction Time: {pred_time} sec ===")
    model = None
    history = None

    for start in range(0, len(file_list), NUM_FILES_TO_LOAD):
        batch_files = file_list[start:start + NUM_FILES_TO_LOAD]
        print(f"Processing files {start} to {start + NUM_FILES_TO_LOAD}...")
        X_train, Y_train = load_data(batch_files, WINDOWSIZE, pred_time)

        if model is None:
            model = build_model(X_train.shape[1:])
            model.summary()

        history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=10, verbose=1)
        del X_train, Y_train  # 메모리 해제

    # 테스트 데이터
    test_file = 'data/_Merged_Data_data_set_01500.csv'
    X_test, Y_test = load_data([test_file], WINDOWSIZE, pred_time)

    # 평가
    loss, mae = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

    # 예측
    y_pred = model.predict(X_test)

    # 결과 저장
    results[pred_time] = {
        "model": model,
        "history": history,
        "y_test": Y_test,
        "y_pred": y_pred
    }

# === 결과 비교 및 시각화 ===
sensor_names = ["Temp1", "Press1", "Temp2", "Press2", "Accel", "Gas"]

# 1. 예측 시간별 센서 예측값 비교
for pred_time, result in results.items():
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(result['y_pred'][:, i], label=f"{sensor_names[i]} Predicted")
    plt.title(f"Predicted Sensor Values ({pred_time} sec)")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# 2. 예측 시간별 오차 비교
for pred_time, result in results.items():
    errors = np.abs(result['y_test'] - result['y_pred'])  # 절대 오차
    mean_error = np.mean(errors, axis=0)

    print(f"\n=== Errors for Prediction Time: {pred_time} sec ===")
    for i, sensor_name in enumerate(sensor_names):
        print(f"{sensor_name}: Mean Error = {mean_error[i]:.6f}")

# 3. Temp1 오차 그래프 비교
plt.figure(figsize=(10, 6))
for pred_time, result in results.items():
    temp1_error = result['y_pred'][:, 0] - result['y_test'][:, 0]  # Temp1 오차
    plt.plot(temp1_error, label=f"{pred_time} sec Error")

plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Error Comparison of Temp1 (Different Prediction Times)")
plt.xlabel("Samples")
plt.ylabel("Error Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
