# Use the official Python 3.10.12 image as the base
FROM python:3.11.4 AS base

# Create and set a working directory
RUN mkdir -p /Maize_Leaf_Disease_Detection
WORKDIR /Maize_Leaf_Disease_Detection

# Create app and models directory
RUN mkdir app models

# Create and activate the virtual environment
RUN python -m venv venv
RUN /bin/bash -c "source venv/bin/activate"

# Copy and run/install the dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --timeout 1000

# Copy all the files in the main/root path
COPY .gitignore main.py ./

# Copy the directories to their correct folder
COPY app ./app
COPY models ./models

# Install the dependencies
# RUN pip3 install -r requirements.txt

# The command to run when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]