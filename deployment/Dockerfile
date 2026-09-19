FROM python:3.12-slim

WORKDIR /app

COPY deployment/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY deployment/app.py .

COPY models/best_hospital_quality_model.pkl .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]