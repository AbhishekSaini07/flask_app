# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy dependency list first for better caching
COPY requirements.txt ./

# Install dependencies
# Increase timeout if needed, --no-cache-dir saves space
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# Copy the rest of the application code and data
COPY . .

# Let Hugging Face Spaces handle the port exposure automatically.
# It will map the internal port (5000 in our app.py) to the public URL.
# EXPOSE 5000 # Generally not needed for HF Spaces

# Command to run the application when the container starts
# Uses the __main__ block in app.py
CMD ["python", "app.py"]