FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create volume for the database
VOLUME /app/data

# Expose ports for both applications
EXPOSE 8501 8502

# Create a script to run both applications
RUN echo '#!/bin/bash\n\
streamlit run auth_app.py --server.port=8501 &\n\
streamlit run machine_learning.py --server.port=8502\n' > /app/run.sh

RUN chmod +x /app/run.sh

# Run both applications
CMD ["/app/run.sh"]