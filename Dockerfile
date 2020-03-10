FROM python:3

COPY ./vae.py .
COPY ./metasaver.py .
COPY ./X_train.csv .
COPY ./test_runner.py .
COPY ./requirements.txt .

RUN pip install -r ./requirements.txt

EXPOSE 6006


CMD ['tensorboard', '--logdir', '.']
CMD ['python', './test_runner.py']

COPY ./experiments ./experiments
