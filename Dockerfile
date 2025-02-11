# Start from Python Alpine base image
FROM rocm/dev-ubuntu-22.04:6.0

# Set working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip


# Copy requirements file
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r amd_requirements.txt

COPY . ./

# Set ROCm environment variables
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV ROCR_VISIBLE_DEVICES=0

# Command to run tests
CMD ["streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
