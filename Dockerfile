FROM python:3.10-slim

ENV PLEX_SERVER_URL=
ENV PLEX_TOKEN=
ENV MCP_LOG_LEVEL=INFO
ENV MCP_TRANSPORT=stdio
ENV MCP_PORT=
ENV MCP_HOST=
ENV MCP_MOUNT=

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:${MCP_PORT}/health || exit 1
WORKDIR /app

# Copy project files
COPY . .
ENTRYPOINT ["/app/entrypoint.sh"]
# Upgrade pip and install dependencies
RUN pip install --no-cache-dir uv==0.7.17 \
    && uv build

# Expose port if needed (not used for stdio transport)