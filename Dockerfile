FROM iphaeton/tensorflow-notebook:latest

USER root
WORKDIR /home/jovian/work/home

ADD ./home/modules ./modules
ADD ./home/utils ./utils
ADD ./home/app.py .
ADD ./home/constants.py .

ADD ./input/models-detection ../input/models-detection
ADD ./input/base-models ../input/base-models
ADD ./input/model ../input/model
ADD ./input/preview ../input/preview

EXPOSE 5000

CMD [ "python", "app.py" ]