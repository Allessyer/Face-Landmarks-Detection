# VisionLabs
Face Landmark Detection

1. cd VisionLabs

docker build . -t face_landmarks:0.0
docker run -it --cpuset-cpus 10-30 --name face_landmarks face_landmarks:0.0
