import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
df = pd.read_csv('../data/medical.csv')
print(df.head())

df.rename(columns={'Hipertension':'Hypertension', 'Handcap':'Handicap'},inplace=True)
print(df.columns)
print(df.info())

print(df.isnull().any(axis=1))
print(df.isnull().any(axis=0))
print(df.describe())

df=df[df.Age >= 0]
print(df.Age.min())

df = df[(df.Handicap==0) | (df.Handicap==1)]
print(df['Handicap'].value_counts())

df['No-show'] = df['No-show'].map({'Yes':1, 'No': 0})
print(df['No-show'].value_counts())

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df.info()

df['waiting_day'] = df['AppointmentDay'].dt.dayofyear - df['ScheduledDay'].dt.dayofyear
print(df.info())
print(df.describe())

df = df[df.waiting_day>=0]
print(df['waiting_day'].min())

print(df.Age.unique())

df=df[df.Age<=110]
plt.figure(figsize=(16,2))
sns.boxplot(x=df.Age)
plt.show()


a = df[df.waiting_day==0]['waiting_day'].value_counts()
b = df[(df['waiting_day']==0) & (df['No-show']==1)]['waiting_day'].value_counts()
print(b/a)

no_show = df[df['No-show']==1]
show = df[df['No-show']==0]

# no_show[no_show['waiting_day'] <= 10]['waiting_day'].hist(alpha=0.7, label='no_show')
# show[show['waiting_day'] <= 10]['waiting_day'].hist(alpha=0.3, label='show')
# plt.legend()
# plt.show()
#
# no_show['ScheduledDay'].hist(alpha=0.7, label='no_show')
# show['ScheduledDay'].hist(alpha=0.3, label='show')
# plt.legend()
# plt.show()
#
# no_show['AppointmentDay'].hist(alpha=0.7, label='no_show')
# show['AppointmentDay'].hist(alpha=0.3, label='show')
# plt.legend()
# plt.show()
#
# print(df.PatientId.value_counts().iloc[0:10])
#
# data = df[(df['waiting_day']>=50) & (df['No-show']==1)].PatientId.value_counts().iloc[0:10]
#
# F = df[(df['Gender']=='F') & (df['No-show']==1)]['Gender'].value_counts()
# M = df[(df['Gender']=='M') & (df['No-show']==1)]['Gender'].value_counts()
# total_F = df[df['Gender']=='F']['Gender'].value_counts()
# total_M = df[df['Gender']=='M']['Gender'].value_counts()
# print(F/total_F)
# print(M/total_M)
#
# sns.barplot(y='waiting_day', x='SMS_received', hue='No-show', data=df)
# plt.show()

tmp=df[['waiting_day', 'SMS_received', 'No-show']].corr()
sns.heatmap(tmp, annot= True)
plt.show()