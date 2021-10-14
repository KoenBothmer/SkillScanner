# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-distilroberta-v1")'
COPY . .
# Set env variables for Cloud Run
ENV PORT 5000
ENV HOST 0.0.0.0

# Open port 5000
EXPOSE 5000:5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]