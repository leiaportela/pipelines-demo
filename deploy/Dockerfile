FROM node:22-slim AS ui
WORKDIR /src/app/frontend
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /srv
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --group app --no-install-project
ENV PATH="/srv/.venv/bin:$PATH"
COPY app/__init__.py app/__init__.py
COPY app/backend/ app/backend/
COPY --from=ui /src/app/frontend/dist app/frontend/dist
CMD exec uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT}
