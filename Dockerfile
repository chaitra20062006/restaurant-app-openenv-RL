FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first (to use Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
# This includes models.py, server folder, etc.
COPY . .

# Set Environment Variables so Python can find your modules
ENV PYTHONPATH="/app"
ENV API_BASE_URL="http://127.0.0.1:7860"

# Hugging Face and Docker use port 7860
EXPOSE 7860

# Start the unified FastAPI + Gradio server
CMD ["python", "-m", "server.app"]