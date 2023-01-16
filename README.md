# VisionLabs
Face Landmarks Detection (Face Alignment)

The task is to implement an algorithm for detecting 68 special points on a person's face (face alignment), to test this algorithm on public datasets, and to compare it with analogues.

## Set up 
Before running train and test of the model, set up docker for that by doing the following:

1. git clone this repository by doing the following:
```
git clone https://github.com/Allessyer/VisionLabs.git
```
2. go to the directory `VisionLabs`
```
cd ISSAI/Task_1/task_1_1
```
3. build docker image
```
docker build . -t face_landmarks:0.0
docker run -it --cpuset-cpus 10-30 --name face_landmarks face_landmarks:0.0
```
Now you are inside the docker container.

4. download and unzip dlib face landmarks detection model
```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
```
5. [download](https://drive.google.com/file/d/0B8okgV6zu3CCWlU3b3p4bmJSVUU/view?usp=sharing) datasets to your local computer 

5.1. untar landmarks_task.tgz
```
tar zxvf landmarks_task.tgz
```
5.2. find docker container id
```
docker ps
```
5.3. copy directory landmarks_task to the docker container
```
docker cp <CONTAINER ID>:/workdir/
```
5.4. Move annotation file to landmarks_task directory
```
mv annotations_file_all_clean.csv landmarks_task
```
6. install necessary libraries
```
pip install -r requirements.txt
```

## Train
To run train with default parameters(ONet model, 300W train dataset):
```
python task_train.py 
```
To see parameters of train function:
```
python task_train.py -h
```

## Test
To run test with default parameters:
```
python task_test.py 
```
To see parameters of test function:
```
python task_test.py -h
```

