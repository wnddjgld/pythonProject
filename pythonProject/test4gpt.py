import os
import shutil  # 폴더 삭제를 위해 추가
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import matplotlib.pyplot as plt

# === 사용자 정의 변수 ===
WINDOWSIZE = 10  # 슬라이딩 윈도우 크기
PREDICTION_TIMES = [3, 5, 10, 30, 60, 120]  # 예측 초 리스트
NUM_FILES_TO_LOAD = 500  # 학습 데이터로 사용할 파일 개수
RESULTS_DIR = "all_results"  # 모든 결과 파일 저장 폴더

# === 결과 저장 폴더 경로 설정 ===
RESULTS_DIR = "all_results"

# === 결과 저장 폴더 삭제 및 재생성 ===
if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)  # 기존 폴더 삭제
os.makedirs(RESULTS_DIR)  # 새 폴더 생성

# 이후 학습 및 결과 저장 코드를 작성

# === 결과 저장 폴더 생성 ===
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# === 데이터 로드 및 병합 ===
# 학습 데이터 폴더
train_folder = '_Merged_Data'
train_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if '_Merged_Data' in f and f.endswith('.csv')]

# 파일 정렬 및 제한
train_files.sort()
train_files = train_files[:NUM_FILES_TO_LOAD]

print(f"Selected Training Files: {train_files}")

# 학습 데이터 병합
train_data = pd.concat([pd.read_csv(f, header=None,
                                    names=['Sec', 'Temp1', 'Press1', 'Temp2', 'Press2', 'Accel', 'Gas'],
                                    index_col='Sec') for f in train_files])

# 테스트 데이터 로드
test_data = pd.read_csv('data/_Merged_Data_data_set_01500.csv',
                        header=None,
                        names=['Sec', 'Temp1', 'Press1', 'Temp2', 'Press2', 'Accel', 'Gas'],
                        index_col='Sec')

# === 데이터 정규화 ===
train_min, train_max = train_data.min(), train_data.max()
train_data = (train_data - train_min) / (train_max - train_min)

test_min, test_max = test_data.min(), test_data.max()
test_data = (test_data - test_min) / (test_max - test_min)

print(f"Train Data Shape: {train_data.shape}, Test Data Shape: {test_data.shape}")

# === 슬라이딩 윈도우 생성 ===
def create_windows(data, window_size, prediction_time):
    """슬라이딩 윈도우 데이터 생성"""
    X, Y = [], []
    for i in range(len(data) - window_size - prediction_time):
        X.append(data.iloc[i:i + window_size].values)
        Y.append(data.iloc[i + window_size + prediction_time].values)
    return np.array(X), np.array(Y)

X_train = {}
Y_train = {}
X_test = {}
Y_test = {}

for pred_time in PREDICTION_TIMES:
    X_train[pred_time], Y_train[pred_time] = create_windows(train_data, WINDOWSIZE, pred_time)
    X_test[pred_time], Y_test[pred_time] = create_windows(test_data, WINDOWSIZE, pred_time)

# Conv2D 입력 형식으로 변환
for pred_time in PREDICTION_TIMES:
    X_train[pred_time] = X_train[pred_time].reshape(-1, WINDOWSIZE, X_train[pred_time].shape[2], 1)
    X_test[pred_time] = X_test[pred_time].reshape(-1, WINDOWSIZE, X_test[pred_time].shape[2], 1)

print(f"Sample Training Shape for {PREDICTION_TIMES[0]} sec: {X_train[PREDICTION_TIMES[0]].shape}")

# === 모델 정의 및 학습 ===
def build_model(input_shape):
    """Conv2D 모델 생성"""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(6, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

results = {}  # 각 모델 결과 저장
for pred_time in PREDICTION_TIMES:
    print(f"\nTraining model for {pred_time} seconds prediction...")

    # 예측 초별 결과 폴더 생성
    pred_folder = os.path.join(RESULTS_DIR, f"{pred_time}_sec")
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    # 모델 생성 및 체크포인트 경로 설정
    model = build_model(X_train[pred_time].shape[1:])
    checkpoint_path = os.path.join(pred_folder, f"model_{pred_time}_sec.ckpt")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1, save_best_only=True
    )

    # 학습
    history = model.fit(
        X_train[pred_time], Y_train[pred_time],
        validation_data=(X_test[pred_time], Y_test[pred_time]),
        epochs=50, batch_size=64, verbose=1, callbacks=[checkpoint_callback]
    )
    y_pred = model.predict(X_test[pred_time])
    results[pred_time] = {
        "model": model,
        "history": history,
        "y_pred": y_pred,
        "y_test": Y_test[pred_time]
    }

    # 히스토리 저장
    history_file = os.path.join(pred_folder, f"history_{pred_time}_sec.npy")
    np.save(history_file, history.history)

    # 모델 평가
    print(f"\nEvaluating model for {pred_time} seconds prediction...")
    loss, mae = results[pred_time]['model'].evaluate(
        X_test[pred_time], Y_test[pred_time], verbose=0)
    print(f"Test Loss for {pred_time} sec: {loss:.4f}, Test MAE: {mae:.4f}")

    # 학습 과정 시각화
    plt.figure(figsize=(6, 4))
    plt.plot(results[pred_time]['history'].history['loss'], label='Loss')  # 손실 함수
    plt.plot(results[pred_time]['history'].history['mae'], label='MAE')  # 평균 절대 오차
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.title(f'{pred_time} sec Prediction Model Training')
    plt.grid()
    plt.tight_layout()
    plt.show()

# === 결과 시각화 ===
sensor_names = ["Temp1", "Press1", "Temp2", "Press2", "Accel", "GasLeak"]

# 예측값과 실제값 비교 그래프
for pred_time in PREDICTION_TIMES:
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(results[pred_time]['y_pred'][:, i], label=f"{sensor_names[i]} Predicted")
    plt.title(f"Predicted Sensor Values ({pred_time} sec)")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# 실제값 비교 그래프
for pred_time in PREDICTION_TIMES:
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(results[pred_time]['y_test'][:, i], label=f"{sensor_names[i]} True")
    plt.title(f"Sensor True Values ({pred_time} sec)")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# 정확도 막대그래프
accuracies = []
for pred_time in PREDICTION_TIMES:
    mae = np.mean(np.abs(results[pred_time]['y_test'] - results[pred_time]['y_pred']))
    accuracy = 100 - mae * 100
    accuracies.append(accuracy)

plt.figure(figsize=(8, 6))
plt.bar([str(t) for t in PREDICTION_TIMES], accuracies, alpha=0.6, color='gray')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 2, f"{acc:.2f}%", ha='center', fontsize=10, color="red")
plt.title("Prediction Model Accuracy")
plt.ylabel("Accuracy [%]")
plt.xlabel("Prediction Times (sec)")
plt.grid()
plt.tight_layout()
plt.show()
