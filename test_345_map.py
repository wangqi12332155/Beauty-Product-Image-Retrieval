import e2LSH
from test_helper import readData,euclideanDistance
from extract_cnn_345 import VGGNet
import csv
import platform
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image
import numpy as np
from numpy import *
from itertools import chain
import os

if platform.system() == "Windows":
    dataPath = r"J:/perfect_500K_datasets/"
else:
    dataPath =  '/media/leiliang/新加卷/perfect_500K_datasets/'
    
file_class=["0001_180889/","0002_112119/","0003_108112/","0004_57203/","0005_42338/","0006_8444/","0007_7572/","0008_4050/"]
picture_name="n0001_0000001.jpg"
class_index=int(picture_name[4])-1
query_all=[]
k1=0.1
k2=0.2
k3=0.7
if __name__ == "__main__":
    C = pow(2, 32) - 5
    dataSet = readData("feature_mean_norm_f1cn_model.49-1.6.csv")
    print("reader datas ok!")
    model = VGGNet()
    image_list=os.listdir(dataPath+"acmmm/Locality-sensitive-hashing-master/result")#sample/sample_pic")
    for image in image_list[0:110]:
        #queryVec = model.extract_feat(dataPath+file_class[class_index]+picture_name)
        image_path=dataPath+"acmmm/Locality-sensitive-hashing-master/result/"+image
        queryVec,queryVec_norm,queryVec_mean_norm = model.extract_feat(image_path)
        #queryVec=np.hstack((np.hstack((k1*queryVec[0:256],k2*queryVec[256:768])),k3*queryVec[768:1280]))
        query_all.append(queryVec_norm)
    print("feature_extract is ok!")
    query =dataSet[0]
    query_list=[]
    query_totle=[]
    A=np.array(dataSet)
    B=np.array(query_all)
    scores = np.dot(B, A.T)
    rank_ID = np.argsort(-scores,axis=1)
    #rank_ID = np.argsort(scores)[::-1]
    #rank_score = scores[rank_ID]
    top_seven=[]
    top_seven_name=[]    
    for rank in rank_ID:
        top_seven.append(rank[0:7])
        
    name_query=[]  
    top_seven_name=[]
    all_index=list(chain.from_iterable(top_seven))
    print("top7 list is ok")    
    with open('query_vgg_all_f1cn_model.49-1.6.csv','r') as csvfile:
       reader = csv.reader(csvfile)                    
       for csv in reader:
            name_query.append(csv)
    for s in all_index:
        for j,rows in enumerate(name_query):
            if(j==int(s)):
                top_seven_name.append(rows[1])
   
    name_seven=[]
    result=[]               
    for i,name in enumerate(top_seven_name):
        name_seven.append(name)
        if (i+1)%7==0:
            result.append(name_seven)
            name_seven=[]
"""
    for k in range(10):
        indexes = e2LSH.nn_search(dataSet, query, k=20, L=5, r=1, tableSize=20)
        for i,index in enumerate(indexes):
            query_list_x=[euclideanDistance(dataSet[index], query),index]
            query_list.append(query_list_x)
            query_list.sort(key=lambda x:x[0],reverse=False)
           # print(i)
        query_totle.append(query_list[0])

    query_totle.sort(key=lambda x:x[0],reverse=False)        
    index_num=query_totle[0][1]        
    with open('query_csv.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for j,rows in enumerate(reader):
            if j== int(index_num):
                out_picture_name=rows[1]
                path=dataPath+file_class[int(out_picture_name[4])-1]+out_picture_name
                print(out_picture_name,query_totle[0])
                #srcImg=Image.open(path) 
               # srcImg.show()
               
picture_name="n0007_0000601.jpg"
class_index=int(picture_name[4])-1              
#queryVec = model.extract_feat(dataPath+file_class[class_index]+picture_name) 
queryVec = model.extract_feat(dataPath+"acmmm/Locality-sensitive-hashing-master/result/gen_0_2464.jpeg")
scores = np.dot(queryVec, A.T)
rank_ID = np.argsort(scores)[::-1]
       
"""