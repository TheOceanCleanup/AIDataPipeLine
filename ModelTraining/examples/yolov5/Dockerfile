FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04:20201019.v1

# RUN pip install Cython matplotlib>=3.2.2 numpy>=1.18.5 opencv-python>=4.1.2 pillow \
#     PyYAML>=5.3 scipy>=1.4.1 tensorboard>=2.2 torch>=1.6.0 torchvision>=0.7.0 tqdm>=4.41.0 \
#     azureml-core==1.17.0 azureml-defaults==1.17.0 azureml-telemetry==1.17.0 azureml-train-restclients-hyperdrive==1.17.0 \
#     azureml-train-core==1.17.0 mkl future

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
