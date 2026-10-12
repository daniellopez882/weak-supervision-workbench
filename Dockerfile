# syntax=docker/dockerfile:1

# Stage 1 resolves uv.lock into a self-contained virtual environment (project installed
# as a wheel, not editable, so the runtime stage needs no source tree).
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/wsw
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Stage 2 is the web app: python:3.12-slim plus the environment, run as uid 10001.
FROM python:3.12-slim AS runtime
RUN groupadd --gid 10001 wsw \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/wsw --shell /usr/sbin/nologin wsw
COPY --from=build /opt/wsw /opt/wsw
ENV PATH=/opt/wsw/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    WSW_DATA_DIR=/home/wsw/data
WORKDIR /home/wsw
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"]
ENTRYPOINT ["wsw"]
CMD ["app", "--host", "0.0.0.0", "--port", "8080", "--dataset", "demo"]
