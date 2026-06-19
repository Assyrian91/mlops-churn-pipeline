# Use an official Python runtime
# We use 3.11-slim as a base for stability
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the app (We will update this later when we build the API)
# CMD ["python", "src/api/main.py"]