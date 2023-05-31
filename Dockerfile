# Stage 1: Build the frontend
FROM node:alpine as frontend-build
WORKDIR /frontend
COPY ./frontend/ /frontend/
RUN npm install && npm run build

# Stage 2: Build the backend
FROM python:3.10-slim

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

# Expose the specified port
ARG PORT
ENV PORT=$PORT
EXPOSE $PORT

# Start the server
CMD ["sh", "-c", "uvicorn main:app --workers 2 --host 0.0.0.0 --port $PORT"]
