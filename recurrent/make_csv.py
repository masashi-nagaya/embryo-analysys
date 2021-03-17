import os
import random
import pandas as pd
import csv
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--time', default=25, type=int,
                    help='using frma after designated time')

parser.add_argument('--csv', default="all", type=str,
                    choices=["all","last"],
                    help='using all-frame or only last-frame')              

parser.add_argument('--data_dir', default="/home/masashi_nagaya/M2/dataset_9_08/all/", type=str,
                    help='data directory')  


args = parser.parse_args()

path_sawada = "/home/masashi_nagaya/M2/dataset_9_08/sawada/"
path_ncu = "/home/masashi_nagaya/M2/dataset_9_08/ncu/"

movie2time={}
df = pd.read_csv("/home/masashi_nagaya/M2/dataset_9_08/time_step/time_step.csv",header=None)
for index ,row in df.iterrows():
    if not row[0] in movie2time.keys():
        movie2time[row[0]]=row[1]

failure_list = []
for movie in os.listdir(path_sawada+"failure/"):
    failure_list.append(movie)

for movie in os.listdir(path_sawada+"failure_/"):
    failure_list.append(movie)

for movie in os.listdir(path_ncu+"failure/"):
    failure_list.append(movie)

birth_list = []
for movie in os.listdir(path_sawada+"birth/"):
    birth_list.append(movie)

for movie in os.listdir(path_sawada+"birth_/"):
    birth_list.append(movie)

for movie in os.listdir(path_ncu+"birth/"):
    birth_list.append(movie)

random.seed(0)
random.shuffle(birth_list)
random.shuffle(failure_list)

q_birth = len(birth_list) // 5
mod_birth  = len(birth_list) % 5

split_number_birth = []
count = 1 
for i in range(5):
    if i+1 > 5-mod_birth: 
        split_number_birth.append((i+1)*q_birth+count)
        count += 1 
    else:
        split_number_birth.append((i+1)*q_birth)

val1_birth = birth_list[0:split_number_birth[0]]
val2_birth = birth_list[split_number_birth[0]:split_number_birth[1]]
val3_birth = birth_list[split_number_birth[1]:split_number_birth[2]]
val4_birth = birth_list[split_number_birth[2]:split_number_birth[3]]
val5_birth = birth_list[split_number_birth[3]:split_number_birth[4]]

q_failure = len(failure_list) // 5
mod_failure  = len(failure_list) % 5

split_number_failure = []
count = 1
for i in range(5):
    if i+1 > 5-mod_failure: 
        split_number_failure.append((i+1)*q_failure+count)
        count += 1
    else:
        split_number_failure.append((i+1)*q_failure)

val1_failure = failure_list[0:split_number_failure[0]]
val2_failure = failure_list[split_number_failure[0]:split_number_failure[1]]
val3_failure = failure_list[split_number_failure[1]:split_number_failure[2]]
val4_failure = failure_list[split_number_failure[2]:split_number_failure[3]]
val5_failure = failure_list[split_number_failure[3]:split_number_failure[4]]


train1_birth = val2_birth + val3_birth + val4_birth + val5_birth
train2_birth = val1_birth + val3_birth + val4_birth + val5_birth
train3_birth = val1_birth + val2_birth + val4_birth + val5_birth
train4_birth = val1_birth + val2_birth + val3_birth + val5_birth
train5_birth = val1_birth + val2_birth + val3_birth + val4_birth

train1_failure = val2_failure + val3_failure + val4_failure + val5_failure
train2_failure = val1_failure + val3_failure + val4_failure + val5_failure
train3_failure = val1_failure + val2_failure + val4_failure + val5_failure
train4_failure = val1_failure + val2_failure + val3_failure + val5_failure
train5_failure = val1_failure + val2_failure + val3_failure + val4_failure

dir = args.data_dir

train_birth = []
train_failure = []
val_birth = []
val_failure = []

val_birth.append(val1_birth)
val_birth.append(val2_birth)
val_birth.append(val3_birth)
val_birth.append(val4_birth)
val_birth.append(val5_birth)

val_failure.append(val1_failure)
val_failure.append(val2_failure)
val_failure.append(val3_failure)
val_failure.append(val4_failure)
val_failure.append(val5_failure)

train_birth.append(train1_birth)
train_birth.append(train2_birth)
train_birth.append(train3_birth)
train_birth.append(train4_birth)
train_birth.append(train5_birth)

train_failure.append(train1_failure)
train_failure.append(train2_failure)
train_failure.append(train3_failure)
train_failure.append(train4_failure)
train_failure.append(train5_failure)

###train_sawada###
for i in range(5):
    with open('./csv/train'+str(i+1)+'_sawada.csv', 'w') as f:
        writer = csv.writer(f)
        for movie in train_birth[i]:
            img_list = []
            for img in os.listdir(dir+movie):
                img = img.split(".")
                img_list.append(img[0])
                img_list.sort(key=int)
            
            time_step = float(movie2time[movie])

            for img in img_list:
                if len(movie) > 5:
                    if time_step >args.time: 
                        writer.writerow([dir+movie+"/"+img+".jpg "+str(1)+" "+str(round(time_step,3))])
                    time_step += 0.25   

        for movie in train_failure[i]:
            img_list = []
            for img in os.listdir(dir+movie):
                img = img.split(".")
                img_list.append(img[0])
                img_list.sort(key=int)

            time_step = float(movie2time[movie])

            for img in img_list:
                if len(movie) > 5: 
                    if time_step >args.time:      
                        writer.writerow([dir+movie+"/"+img+".jpg "+str(-1)+" "+str(round(time_step,3))])
                    time_step += 0.25

###train_ncu###
for i in range(5):
    with open('./csv/train'+str(i+1)+'_ncu.csv', 'w') as f:
        writer = csv.writer(f)
        for movie in train_birth[i]:
            img_list = []
            for img in os.listdir(dir+movie):
                img = img.split(".")
                img_list.append(img[0])
                img_list.sort(key=int)
            
            time_step = float(movie2time[movie]/60)

            for img in img_list:                            
                if len(movie) < 5:
                    if time_step >25: 
                        writer.writerow([dir+movie+"/"+img+".jpg "+str(1)+" "+str(round(time_step,3))])
                    time_step += 0.167

        for movie in train_failure[i]:
            img_list = []
            for img in os.listdir(dir+movie):
                img = img.split(".")
                img_list.append(img[0])
                img_list.sort(key=int)

            time_step = float(movie2time[movie]/60)

            for img in img_list:
                if len(movie) < 5:
                    if time_step >args.time: 
                        writer.writerow([dir+movie+"/"+img+".jpg "+str(-1)+" "+str(round(time_step,3))])   
                    time_step += 0.167


###val_sawada###
for i in range(5):
    with open('./csv/test'+str(i+1)+'_sawada.csv', 'w') as f:
        writer = csv.writer(f)
        for movie in val_birth[i]:
            img_list = []
            for img in os.listdir(dir+movie):
                img = img.split(".")
                img_list.append(img[0])
                img_list.sort(key=int)
            
            time_step = float(movie2time[movie])

            for img in img_list:
                if len(movie) > 5:
                    if time_step >args.time: 
                        writer.writerow([dir+movie+"/"+img+".jpg "+str(1)+" "+str(round(time_step,3))])
                    time_step += 0.25   

        for movie in val_failure[i]:
            img_list = []
            for img in os.listdir(dir+movie):
                img = img.split(".")
                img_list.append(img[0])
                img_list.sort(key=int)

            time_step = float(movie2time[movie])

            for img in img_list:
                #sawada#
                if len(movie) > 5: 
                    if time_step >args.time:      
                        writer.writerow([dir+movie+"/"+img+".jpg "+str(-1)+" "+str(round(time_step,3))])
                    time_step += 0.25

###val_ncu###
for i in range(5):
    with open('./csv/test'+str(i+1)+'_ncu.csv', 'w') as f:
        writer = csv.writer(f)
        for movie in val_birth[i]:
            img_list = []
            for img in os.listdir(dir+movie):
                img = img.split(".")
                img_list.append(img[0])
                img_list.sort(key=int)
            
            time_step = float(movie2time[movie]/60)

            for img in img_list:
                if len(movie) < 5:
                    if time_step >args.time: 
                        writer.writerow([dir+movie+"/"+img+".jpg "+str(1)+" "+str(round(time_step,3))])
                    time_step += 0.167

        for movie in val_failure[i]:
            img_list = []
            for img in os.listdir(dir+movie):
                img = img.split(".")
                img_list.append(img[0])
                img_list.sort(key=int)

            time_step = float(movie2time[movie]/60)

            for img in img_list:
                if len(movie) < 5:
                    if time_step >args.time: 
                        writer.writerow([dir+movie+"/"+img+".jpg "+str(-1)+" "+str(round(time_step,3))])
                    time_step += 0.167