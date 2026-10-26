# Python 3.10 Image without Dependencies
FROM python:3.10-slim

LABEL org.opencontainers.image.source="https://github.com/AstraZeneca/ConfiDock"

RUN pip install "poetry>=2.0,<3"

ENV VIRTUAL_ENV=/opt/venv
RUN python -m virtualenv --python=/usr/local/bin/python $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . /app
WORKDIR /app

RUN poetry install

# Add your bits here.
