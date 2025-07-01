FROM python:3.10-slim

USER mcp

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

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir uv==0.7.17 \
    && uv build

# Expose port if needed (not used for stdio transport)

# Run the MCP server
CMD [ "uv", "run", "plex-mcp.py", "-u", "$PLEX_SERVER_URL", "-k", "$PLEX_TOKEN", "-t", "$MCP_TRANSPORT", "-b", "$MCP_HOST", "-p", "$MCP_PORT", "-m", "$MCP_MOUNT", "-l", "$MCP_LOG_LEVEL"]
