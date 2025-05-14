# Python image
FROM python:3.12-slim

# Directory
WORKDIR /app

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn pydantic-settings

# Install ai-cdss from GitHub Release (v0.1.1 tarball)
RUN pip install https://github.com/dabadav/ai-cdss/releases/download/v0.1.1/ai_cdss-0.1.1.tar.gz

# Copy only FastAPI app code
COPY . .

# Expose the default FastAPI port
EXPOSE 8000

# Launch FastAPI app via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]