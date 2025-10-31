# Use a specific, stable Python version
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directories for the database and documents
RUN mkdir -p ./chroma_db ./documents

# Expose the port the app runs on
EXPOSE 5000

# Specify the command to run on container startup
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]