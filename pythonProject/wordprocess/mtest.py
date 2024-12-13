import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 파일 읽어오기
file_path = '../data/measuredData.csv'  # 파일 경로
df = pd.read_csv(file_path)

# 2. 기본 정보 및 기초 통계량 확인
print("데이터프레임 구성:")
print(df.head())
print("\n데이터프레임 정보:")
print(df.info())
print("\n기초 통계량:")
print(df.describe())

# 3. 컬럼명 한글에서 영어로 변경
df.rename(columns={
    '날짜': 'date',
    '아황산가스': 'SO2',
    '일산화탄소': 'CO',
    '오존': 'O3',
    '이산화질소': 'NO2',
    'PM10': 'PM10',
    'PM2.5': 'PM2.5'
}, inplace=True)

print("\n컬럼명 변경 완료:\n", df.columns)

# 4. 데이터 타입 변경
# 날짜 열 정리 (문자열 공백 제거)
df['date'] = df['date'].str.strip()
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H', errors='coerce')

# 5. 결측치 확인 및 처리
print("\n결측치")
print(df.isnull().sum())


df.fillna(df.mean(numeric_only=True), inplace=True)
print("\n처리 후 결측치")
print(df.isnull().sum())

# 6. 상관계수 확인
correlation_matrix = df.drop(columns=['date']).corr()
print("\n상관계수:\n", correlation_matrix)

# 7. 히스토그램으로 데이터 분포 시각화
numeric_columns = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM2.5']

plt.figure(figsize=(12, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(3, 2, i)
    plt.hist(df[column], bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 8. 막대그래프로 일별 PM10 현황
if not df['date'].isnull().all():
    daily_avg = df.groupby(df['date'].dt.date).mean()
    plt.figure(figsize=(14, 6))
    plt.bar(daily_avg.index, daily_avg['PM10'], color='skyblue', alpha=0.7, edgecolor='black')
    plt.title("Daily PM10 Average")
    plt.xlabel("Date")
    plt.ylabel("PM10 (µg/m³)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 9. 히트맵으로 상관관계 시각화
# 히트맵으로 상관관계 시각화 (seaborn 사용)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 10. PM10과 CO의 관계 확인 (산점도 1)
pm10_scores = np.array(df['PM10'])
co_scores = np.array(df['CO'])

fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(111)
ax1.scatter(co_scores, pm10_scores, alpha=0.5)
ax1.set_xlabel("CO (ppm)")
ax1.set_ylabel("PM10 (µg/m³)")
ax1.set_title("CO vs PM10")
plt.show()

# 11. PM10과 PM2.5 관계 확인 (산점도 2)
pm25_scores = np.array(df['PM2.5'])

fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(111)
ax2.scatter(pm10_scores, pm25_scores, alpha=0.5)
ax2.set_xlabel("PM10 (µg/m³)")
ax2.set_ylabel("PM2.5 (µg/m³)")
ax2.set_title("PM10 vs PM2.5")
plt.show()

# 12. 데이터 분석 정리
print("\n데이터 분석 정리:")
print("1. PM10과 PM2.5는 매우 강한 양의 상관관계를 가짐.")
print("2. PM10의 일별 평균값에서 특정 날짜에 높은 농도가 확인됨.")
print("3. CO와 NO2는 강한 양의 상관관계를 가지며, NO2와 O3는 음의 상관관계를 보임.")
