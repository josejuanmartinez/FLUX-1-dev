FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV FOUNDATIONAL_MODEL="black-forest-labs/FLUX.1-dev"

# Set environment variables
ENV MPLCONFIGDIR=/tmp/matplotlib \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Install system dependencies and Python 3.10 + headers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        wget \
        curl \
        tzdata \
        software-properties-common \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        libpython3.10-dev \
        python3-pip \
        python-is-python3 && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip && \
    test -f /usr/include/python3.10/Python.h && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set root working directory
WORKDIR /code

# Copy and install Python requirements
COPY ./requirements.txt /code/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m -u 1000 user

# Set working dir for non-root user
WORKDIR $HOME/app

# Hugging Face login using Docker secret
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    huggingface-cli login --add-to-git-credential --token $(cat /run/secrets/HF_TOKEN)

# Copy application code
COPY --chown=user . $HOME/app

RUN mkdir -p /home/user/.triton && \
    chown -R user:user /home/user/.triton
RUN mkdir -p /home/user/.cache/huggingface/hub && \
    chown -R user:user /home/user/.cache
RUN mkdir -p /home/user/app/lora_output && \
    chown -R user:user /home/user/app/lora_output

# Switch to non-root user
USER user

# Launch the app
CMD ["python3", "main.py"]
