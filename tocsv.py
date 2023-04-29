import os
import pandas as pd

BASE_DIR = 'DogSegmentation/'
train_folder = BASE_DIR+'Images/'
train_annotation = BASE_DIR+'Labels/'
fileType = "jpg"
outPutFileName = "dog.csv"

files_in_train = sorted(os.listdir(train_folder))
files_in_annotated = sorted(os.listdir(train_annotation))

images = []
images2 = []
files=[i for i in files_in_train]
files2=[i for i in files_in_annotated]
for file in files:
    length = len(file)
    if file[length-3:length] == fileType:
        images.append(file)

for file in files2:
    length = len(file)
    if file[length-3:length] == fileType:
        images2.append(file)

df = pd.DataFrame()
df['images']=[train_folder+str(x) for x in images]
df['labels']=[train_annotation+str(x) for x in images2]


df.to_csv(outPutFileName, header=True, index=False)