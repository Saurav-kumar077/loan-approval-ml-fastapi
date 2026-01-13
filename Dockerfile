# 1️⃣ Python base image
FROM python:3.10-slim

# 2️⃣ Working directory inside container
WORKDIR /app

# 3️⃣ Copy requirements first (for faster builds)
COPY requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy all project files
COPY . .

# 6️⃣ Expose FastAPI port
EXPOSE 8000

# 7️⃣ Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
