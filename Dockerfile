FROM python:3.10-slim

ENV PLEX_SERVER_URL=
ENV PLEX_TOKEN=
ENV MCP_LOG_LEVEL=INFO
ENV MCP_TRANSPORT=stdio
ENV MCP_PORT=
ENV MCP_HOST=
ENV MCP_MOUNT=

WORKDIR /app

# Copy project files
COPY . .
ENTRYPOINT ["/app/entrypoint.sh"]
# Upgrade pip and install dependencies

# Expose port if needed (not used for stdio transport)