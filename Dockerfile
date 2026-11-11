# Control plane and CPU-only fake worker.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY manyworlds ./manyworlds
RUN pip install --no-cache-dir ".[test]"
COPY policy ./policy
COPY tests ./tests
ENTRYPOINT ["mw"]
CMD ["status"]
