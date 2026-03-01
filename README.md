# nfl-receiver-yards-ai-api

A production-deployed AI inference service that predicts NFL wide receiver performance against a given defense.

This project demonstrates how to design, containerize, and deploy an AI system to production using FastAPI, Docker, GitHub Actions CI/CD, and AWS ECS (Fargate) behind an Application Load Balancer (ALB).

## Live Demo

Base URL:
http://nfl-api-alb-2067157598.us-east-2.elb.amazonaws.com

Endpoints:

- GET /health
- GET /docs
- POST /predict


### Example Request (PowerShell)

Invoke-RestMethod -Method Post `
  -Uri "http://nfl-api-alb-2067157598.us-east-2.elb.amazonaws.com/predict" `
  -ContentType "application/json" `
  -Body '{"receiver":"J.Jefferson","defteam":"CLE"}'

Example Response:

{
  "predicted_yards": 69.304
}

Note:
The receiver format must be FirstInitial.LastName (ex: J.Jefferson).


## Architecture Overview

Push to GitHub triggers GitHub Actions.

GitHub Actions:
1. Builds Docker image
2. Tags image with Git commit SHA
3. Pushes image to Amazon ECR
4. Renders new ECS task definition revision
5. Deploys to ECS Fargate
6. Waits for ALB health checks
7. Rolls back automatically if unhealthy

Deployment Flow:

GitHub → GitHub Actions → ECR → ECS (Fargate) → ALB → Public API


## Local Development

Create virtual environment:

python -m venv .venv

Activate:

Windows:
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run locally:

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Open:
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs


## Docker

Build image:

docker build -t nfl-api .

Run container:

docker run -p 8000:8000 nfl-api


## Testing

Run tests:

pytest


## What This Project Demonstrates

- ML model served as production API
- Docker containerization
- Immutable deployments (image tagged by commit SHA)
- AWS ECS Fargate deployment
- Application Load Balancer health checks
- Automatic rollback via ECS circuit breaker
- CI/CD automation with GitHub Actions
- Production debugging using CloudWatch + target groups


## Future Improvements

- Add player lookup endpoint
- Add name normalization (Justin Jefferson → J.Jefferson)
- Add metrics / monitoring dashboard
- Add retraining pipeline
