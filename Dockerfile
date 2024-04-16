FROM python:3.8

WORKDIR /application

COPY requirements.txt /application/requirements.txt

RUN pip install --no-cache-dir -r /application/requirements.txt

EXPOSE 8080

COPY . /application

CMD [ "python", "./app/api.py" ]