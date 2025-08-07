FROM python:3.10-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY first_screen_fileinput.py second_screen_fileinput.py .
COPY Data/ ./Data/

EXPOSE 5006

# CMD ["bokeh", "serve", "first_screen_fileinput.py", "--allow-websocket-origin=*", "--port=5006", "--websocket-max-message-size=1073741824"]
# CMD ["bokeh", "serve", "first_screen_fileinput.py", "second_screen_fileinput.py", "--port", "5006", "--allow-websocket-origin=*", "--websocket-max-message-size=1073741824"]
CMD ["bokeh", "serve", "zaslon_1.py", "zaslon_2.py", "--port", "5006", "--allow-websocket-origin=*", "--websocket-max-message-size=1073741824"]

