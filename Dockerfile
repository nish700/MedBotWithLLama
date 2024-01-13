FROM python:3.9.0

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt --default-timeout=100 future

COPY . .

EXPOSE 8000

CMD ["chainlit","run","chat_model.py"]

LABEL authors="nishant"

ENTRYPOINT ["top", "-b"]