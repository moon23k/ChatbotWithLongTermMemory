'''
AI 감성데이터셋으로 준비하였으며 데이터는 최대 4개의 멀티턴으로 구성되어있습니다.
목적은 답변이 아닌 연관이 있는 문장을 리트리벌하는 것이고, 각 텍스트의 앞뒤문장을 Label로 만들었습니다.
주변 문장을 훈련하는 것을 통해 입력문장에대한 연관이 있는 문장을 검출하는 효과를 기대하고있습니다.
Input : Q > Output : A
Input : A > Output : Q
'''

import pandas as pd
import time


def make_it_QAAQ(df):
    diction = {'Q':[],'A':[]}
    col_sen =  ['사람문장1','시스템응답1',
                '사람문장2','시스템응답2',
                '사람문장3','시스템응답3',
                '사람문장4','시스템응답4']
    starts = time.time()
    for i in range(len(df)):
    for sen in range(len(col_sen)) :
        if type(df.iloc[i][col_sen[sen]])!=float:
        if sen != 7:
            if type(df.iloc[i][col_sen[sen+1]])!=float:
            diction['Q'].append(df.iloc[i][col_sen[sen]])
            diction['A'].append(df.iloc[i][col_sen[sen+1]])
        if sen != 0:
            if type(df.iloc[i][col_sen[sen-1]])!=float:
            diction['Q'].append(df.iloc[i][col_sen[sen]])
            diction['A'].append(df.iloc[i][col_sen[sen-1]])
    if i%1000==0:
        print(i,'/',len(df),"\t",round(i/len(df)*100,2),"%" ,"\t", round(time.time()-starts,2),'s')
        starts = time.time()
    return pd.DataFrame.from_dict(diction)


if __name__ == "main":
    df = pd.read_excel("your/dir/filename")
    df_train = make_it_QAAQ(df)
    df_train.to_csv("your/dir/filename.csv",encoding='utf-8',index=False)




