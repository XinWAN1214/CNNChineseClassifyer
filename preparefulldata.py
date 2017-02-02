#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import csv
import pandas
from os import listdir


all = codecs.open('all.csv','w',encoding='gb3212')
spamwriter = csv.writer(all, delimiter=',')
for f in listdir('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs'):
    filename = '/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/'+f+'/prediction.csv'
    print(filename)


df1 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485417285/prediction.csv')
df2 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485424096/prediction.csv')
df3 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485442006/prediction.csv')
df4 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485486559/prediction.csv')
df5 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485508568/prediction.csv')
df6 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485515722/prediction.csv')
df7 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485530081/prediction.csv')
df8 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485532277/prediction.csv')
df9 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485574421/prediction.csv')
df10 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485602383/prediction.csv')
df11 = pandas.read_csv('/Users/liling/PycharmProjects/cnn-text-classification-tf/runs/1485609762/prediction.csv')

print(df1.index)
print(df2.index)
print(df3.index)
print(df4.index)
print(df5.index)
print(df6.index)
print(df7.index)
print(df8.index)
print(df9.index)
print(df10.index)
print(df11.index)

for i in df1.index:
    print(i)
    # print(df1.iloc[i,0])
    spamwriter.writerow([df1.iloc[i,0],df1.iloc[i,1],df2.iloc[i,1],df3.iloc[i,1],df4.iloc[i,1],df5.iloc[i,1],df6.iloc[i,1],df7.iloc[i,1],df8.iloc[i,1],df9.iloc[i,1],df10.iloc[i,1],df11.iloc[i,1]])
