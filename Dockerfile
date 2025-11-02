FROM python:3.10-slim

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        python3-pip \
        build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY . /app

RUN pip3 install -e .

CMD ["python3", "-u", "/app/my_worker.py"]