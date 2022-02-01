FROM python:3.8.12

RUN python3 -m pip install --upgrade pip

WORKDIR /FakeNewsDetectionWeb

COPY requirements.txt .

RUN python3 -m venv venv

RUN pip install -r requirements.txt
