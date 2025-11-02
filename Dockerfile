FROM python:3.10-slim

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY . /app

CMD python3 -u /app/my_worker.py