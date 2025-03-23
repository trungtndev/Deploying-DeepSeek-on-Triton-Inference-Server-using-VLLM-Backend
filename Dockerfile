# Dockerfile
FROM python:3.10.16-bookworm

# Set the working directory
WORKDIR /app

RUN pip install fastapi[standard] tritonclient[all]

COPY api/ ./api
