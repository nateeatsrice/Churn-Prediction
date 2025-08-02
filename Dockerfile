FROM python:3.13-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system 

COPY ["predict.py", "predict-test.py", "churn-model.bin", "./"]

EXPOSE 9698

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"] 