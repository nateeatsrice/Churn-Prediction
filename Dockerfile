FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system 

COPY ["*.py", "churn-model.bin", "./"]

EXPOSE 9698

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9698", "churn_serving:app"]