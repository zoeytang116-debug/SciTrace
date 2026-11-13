FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /home/user/app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend ./backend
COPY frontend/dist ./frontend/dist
COPY pdf ./pdf
COPY data ./data

RUN useradd --create-home --uid 1000 user \
    && chown -R user:user /home/user/app
USER user

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-7860}"]

