FROM python:3.12-slim

LABEL org.opencontainers.image.title="CrashDiag safe mock sandbox"
LABEL org.opencontainers.image.description="Stateful in-memory training sandbox; performs no host fault injection"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CRASHDIAG_SANDBOX_HOST=0.0.0.0 \
    CRASHDIAG_SANDBOX_PORT=8765

WORKDIR /app

RUN groupadd --system --gid 10001 crashdiag \
    && useradd --system --uid 10001 --gid crashdiag --no-create-home \
       --home-dir /nonexistent --shell /usr/sbin/nologin crashdiag

COPY --chown=crashdiag:crashdiag crashdiag/ ./crashdiag/

USER 10001:10001

EXPOSE 8765

HEALTHCHECK --interval=15s --timeout=3s --start-period=3s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=2).read()"]

CMD ["python", "-m", "crashdiag.sandbox_server"]
