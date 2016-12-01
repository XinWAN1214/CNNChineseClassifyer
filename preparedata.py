#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import jieba
import codecs
import gensim
import logging


def getColumn(strcol,df):
    if strcol in df.columns:
        return df[strcol]
    else:
        return -1

# r id2, id, zx_id, zx_floor, nr, sj, block_id, interview, tel, surgery, chinese_medicine, nosurgery, notreatment, local_and_other, detailofdrugusing, uncertain, refuse, unrelated, I131
# 0 u'id2', u'nr', u'block_id', u'tag', u'interview2', u'tel2', u'surgery2', u'chinese_medicine2', u'nosurgery2', u'notreatment2', u'local_and_other2', u'detailofdrugusing2', u'uncertain2', u'refuse2', u'unrelated2', u'id', u'block_id.1', u'interview', u'tel', u'surgery', u'chinese_medicine', u'nosurgery', u'notreatment', u'local_and_other', u'detailofdrugusing', u'uncertain', u'refuse', u'unrelated'
def insertIntoresult(src):
    return pandas.DataFrame({'id2':getColumn('id2',src),
                                'id':getColumn('id',src),
                                'zx_id':getColumn('zx_id',src),
                                'zx_floor':getColumn('zx_floor',src),
                                'nr':getColumn('nr',src),
                                'sj':getColumn('sj',src),
                                'block_id':getColumn('block_id',src),
                                'interview':getColumn('interview',src),
                                'tel':getColumn('tel',src),
                                'surgery':getColumn('surgery',src),
                                'chinese_medicine':getColumn('chinese_medicine',src),
                                'nosurgery':getColumn('nosurgery',src),
                                'notreatment':getColumn('notreatment',src),
                                'local_and_other':getColumn('local_and_other',src),
                                'detailofdrugusing':getColumn('detailofdrugusing',src),
                                'uncertain':getColumn('uncertain',src),
                                'refuse':getColumn('refuse',src),
                                'unrelated':getColumn('unrelated',src),
                                'interview2': getColumn('interview2', src),
                                'tel2': getColumn('tel2', src),
                                'surgery2': getColumn('surgery2', src),
                                'chinese_medicine2': getColumn('chinese_medicine2', src),
                                'nosurgery2': getColumn('nosurgery2', src),
                                'notreatment2': getColumn('notreatment2', src),
                                'local_and_other2': getColumn('local_and_other2', src),
                                'detailofdrugusing2': getColumn('detailofdrugusing2', src),
                                'uncertain2': getColumn('uncertain2', src),
                                'refuse2': getColumn('refuse2', src),
                                'unrelated2': getColumn('unrelated2', src),
                                'I1312':getColumn('I1312',src)})


'''
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

# STEP1 ---------------- define original manual classification data as objects here.

df = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/full_raw_data.xlsx',encoding='utf-8')

d0 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/0.xlsx',encoding='utf-8')
d1 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/1.xlsx',encoding='utf-8')
d2 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/2.xlsx',encoding='utf-8')
d3 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/3.xlsx',encoding='utf-8')
d4 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/4.xlsx',encoding='utf-8')
d5 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/5.xlsx',encoding='utf-8')
d6 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/6.xlsx',encoding='utf-8')
d7 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/7.xlsx',encoding='utf-8')
d8 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/8.xlsx',encoding='utf-8')
d9 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/9.xlsx',encoding='utf-8')
d10 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/10.xlsx',encoding='utf-8')
d11 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/11.xlsx',encoding='utf-8')
d12 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/12.xlsx',encoding='utf-8')
d13 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/13.xlsx',encoding='utf-8')
d14 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/14.xlsx',encoding='utf-8')
d15 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/15.xlsx',encoding='utf-8')
d16 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/16.xlsx',encoding='utf-8')
d17 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/17.xlsx',encoding='utf-8')
d18 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/18.xlsx',encoding='utf-8')
d19 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/19.xlsx',encoding='utf-8')
d20 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/20.xlsx',encoding='utf-8')
d21 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/21.xlsx',encoding='utf-8')
d22 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/22.xlsx',encoding='utf-8')
d23 = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/23.xlsx',encoding='utf-8')

datalist = [d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23]

currentColumns = list(df.columns.values)
lengthOfcolumns = len(currentColumns)
print('[-1]0' + ' ' + str(lengthOfcolumns) + str(currentColumns))

goodresult = pandas.DataFrame(columns=currentColumns)
badresult = pandas.DataFrame(columns=currentColumns)
# print(result)

# STEP2 ---------------- loop all batches result and split them into consistant result (good result) and inconsistant result (bad result)

for i in range(0,24):
    currentdf = datalist[i]
    currentColumns = list(currentdf.columns.values)
    lengthOfcolumns = len(currentColumns)
    tags = currentdf.groupby('tag').size()
    print('['+str(i)+']'+ str(tags[1]) + ' ' + str(lengthOfcolumns) + str(currentColumns))
    # print(currentdf)

    for j in range(0,len(currentdf.index)-1):
        # print('this is '+str(j)+' in '+str(i))
        currentLine = currentdf.iloc[[j]]
        # print(currentLine)
        if(int(currentLine['tag'])==1):
            #print('find a tag =1')
            goodresult = goodresult.append(insertIntoresult(currentLine))
        else:
            badresult = badresult.append(insertIntoresult(currentLine))

#print(result)

# STEP3 ---------------- So good result will be used as training samples; bad result will be confirmed by testing.

goodresult.to_excel('/Users/Winnerineast/Documents/haodaifu/NewData/goodresult.xlsx', encoding='utf-8')
badresult.to_excel('/Users/Winnerineast/Documents/haodaifu/NewData/badresult.xlsx', encoding='utf-8')
'''

# use goodresult file to generate positivate and negative files for each type

# STEP4 ---------------- Knowing the categories are many, each time choose one type only.

data = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/goodresult.xlsx', encoding='utf-8')

# r id2, id, zx_id, zx_floor, nr, sj, block_id, interview, tel, surgery, chinese_medicine, nosurgery, notreatment, local_and_other, detailofdrugusing, uncertain, refuse, unrelated, I131
positive = pandas.DataFrame()
negative = pandas.DataFrame()

data = data.fillna(-1)

types = ['interview', 'tel', 'surgery', 'chinese_medicine', 'nosurgery', 'notreatment', 'local_and_other', 'detailofdrugusing', 'uncertain', 'refuse', 'unrelated']

for type in types:
    print type
    grouped = data.groupby(type)
    if len(grouped.get_group(1)['nr']) > 0:
        pandas.DataFrame(grouped.get_group(1)['nr']).to_csv('/Users/Winnerineast/Documents/haodaifu/NewData/'+type+'_pos.csv', header=False, encoding='utf-8')
    if len(grouped.get_group(-1)['nr']) > 0:
        pandas.DataFrame(grouped.get_group(-1)['nr']).to_csv('/Users/Winnerineast/Documents/haodaifu/NewData/'+type+'_neg.csv', header=False, encoding='utf-8')


# STEP5 ---------------- Further process testing data as you wish

# df = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/badresult.xlsx',encoding='utf-8')


df = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/goodresult.xlsx', encoding='utf-8')
i = 0
with codecs.open('/Users/Winnerineast/Documents/haodaifu/NewData/jieba_goodresult.csv', 'w','utf-8') as result:
    for strline in df['nr']:
        # print strline
        if isinstance(strline, (int,float)):
            continue
        else:
            seg_list = jieba.cut(strline)
        result.write(",".join(seg_list))
        result.write('\n')
    result.close()

# STEP6 ---------------- Further process training data as you wish.

df = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/goodresult.xlsx', encoding='utf-8')
pandas.DataFrame(df['nr']).to_csv('/Users/Winnerineast/Documents/haodaifu/NewData/goodresult.csv', encoding='utf-8')


# STEP7 ---------------- Using gensim word2vec to training words before feed them into the codes.

df = pandas.read_excel('/Users/Winnerineast/Documents/haodaifu/NewData/badresult.xlsx', encoding='utf-8')
pandas.DataFrame(df['nr']).to_csv('/Users/Winnerineast/Documents/haodaifu/NewData/tobetrained.csv', encoding='utf-8')
