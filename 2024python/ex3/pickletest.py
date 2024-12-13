import pickle

data_L =['와플', '아이스크림', '커피']
data_T =('55', '손흥민',187.3)
data_D ={'kr':'한글', 'en':'english'}

with open('test.pkl','wb') as fw: # == WRITE
    pickle.dump(data_L,fw)
    pickle.dump(data_T,fw)
    pickle.dump(data_D,fw)

with open('test.pkl','rb') as fr: # == READ
    data_1=pickle.load(fr)
    data_2=pickle.load(fr)
    data_3=pickle.load(fr)

print(data_1); print(data_2);print(data_3)