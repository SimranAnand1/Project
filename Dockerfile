# Stage 1: Build the frontend
FROM node:alpine as frontend-build
WORKDIR /frontend
COPY ./frontend/ /frontend/
RUN npm install && npm run build

# Stage 2: Build the backend
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app/
COPY --from=frontend-build /frontend/build /app/static

# Copy the backend source code
WORKDIR /app
COPY ./backend/ /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV ENVIRONMENT=production
ENV STATIC_DIR=/app/static
ENV MODELS_PATH=/app/src/models
