FROM python:3.13-alpine
COPY --from=ghcr.io/astral-sh/uv:0.1.0 /uv /uvx /bin/
WORKDIR /app
COPY . /app
RUN uv sync --frozen --no-cache
CMD ["uv", "run", "python", "app.py"]
