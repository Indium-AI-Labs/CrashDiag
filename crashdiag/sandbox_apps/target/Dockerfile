FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CRASHDIAG_VICTIM_ROOT=/var/lib/crashdiag \
    CRASHDIAG_DISK_ROOT=/tmp/crashdiag-disk

WORKDIR /app

RUN groupadd --system --gid 10002 victim \
    && useradd --system --uid 10002 --gid victim --no-create-home \
       --home-dir /nonexistent --shell /usr/sbin/nologin victim \
    && mkdir -p /var/lib/crashdiag /tmp/crashdiag-disk \
    && chown -R victim:victim /var/lib/crashdiag /tmp/crashdiag-disk

COPY crashdiag/ /app/crashdiag/

USER 10002:10002

CMD ["python", "-m", "crashdiag.sandbox_apps.target.app"]
