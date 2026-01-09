#!/usr/bin/env bash
set -e

IMAGE_NAME=churn-api

docker build -t $IMAGE_NAME .
docker run --rm -p 8000:8000 --name churn-api $IMAGE_NAME