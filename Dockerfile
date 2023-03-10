FROM ghcr.io/osai-ai/dokai:22.03-pytorch
ARG DEBIAN_FRONTEND=noninteractive

ENV TORCH_HOME /workdir/data/.torch
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# COPY *.* /workdir/
# COPY face_landmarks /workdir/face_landmarks
# COPY annotation_files /workdir/annotation_files

COPY landmarks_task.tgz /workdir/
COPY YinNet_exp1_3_model_best_auc.pth /workdir/

RUN git clone https://github.com/Allessyer/VisionLabs.git

RUN cd VisionLabs && mv * ../

RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
RUN bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
RUN rm shape_predictor_68_face_landmarks.dat.bz2
RUN wget http://dlib.net/files/mmod_human_face_detector.dat.bz2
RUN bzip2 -dk mmod_human_face_detector.dat.bz2
RUN rm mmod_human_face_detector.dat.bz2

RUN tar zxvf landmarks_task.tgz && rm landmarks_task.tgz

RUN mkdir results
RUN mkdir weights
RUN mv /workdir/YinNet_exp1_3_model_best_auc.pth /workdir/weights

WORKDIR /workdir

ENTRYPOINT tail -f /dev/null


