# Graphiti MCP Server 🧠

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> 🌟 A powerful knowledge graph server for AI agents, built with Neo4j and integrated with Model Context Protocol (MCP).

## 🚀 Features

- 🔄 Dynamic knowledge graph management with Neo4j
- 🤖 Seamless integration with OpenAI models
- 🔌 MCP (Model Context Protocol) support
- 🐳 Docker-ready deployment
- 🎯 Custom entity extraction capabilities
- 🔍 Advanced semantic search functionality

## 🛠️ Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.10 or higher
- OpenAI API key
- Minimum 4GB RAM (recommended 8GB)
- 2GB free disk space

### Quick Start 🚀

1. Clone the repository:
```bash
git clone https://github.com/gifflet/graphiti-mcp-server.git
cd graphiti-mcp-server
```

2. Set up environment variables:
```bash
cp .env.sample .env
```

3. Edit `.env` with your configuration:
```bash
# Required for LLM operations
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4.1-mini

# Optional: Custom OpenAI endpoint (e.g., for proxies)
# OPENAI_BASE_URL=https://api.openai.com/v1

# Neo4j Configuration (defaults work with Docker)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=demodemo
```

4. Start the services:
```bash
docker compose up -d
```

5. **Verify installation:**
```bash
# Check if services are running
docker compose ps

# Check logs
docker compose logs graphiti-mcp
```

### Alternative: Environment Variables

You can run with environment variables directly:
```bash
OPENAI_API_KEY=your_key MODEL_NAME=gpt-4.1-mini docker compose up
```

## 🔧 Configuration

### Service Ports 🌐

| Service | Port | Purpose |
|---------|------|---------|
| Neo4j Browser | 7474 | Web interface for graph visualization |
| Neo4j Bolt | 7687 | Database connection |
| Graphiti MCP | 8000 | MCP server endpoint |

### Environment Variables 🔧

#### OpenAI Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | - | Your OpenAI API key |
| `OPENAI_BASE_URL` | ❌ | - | Custom OpenAI API endpoint (consumed by OpenAI SDK) |
| `MODEL_NAME` | ❌ | `gpt-4.1-mini` | Main LLM model to use |
| `SMALL_MODEL_NAME` | ❌ | `gpt-4.1-nano` | Small LLM model for lighter tasks |
| `LLM_TEMPERATURE` | ❌ | `0.0` | LLM temperature (0.0-2.0) |
| `EMBEDDER_MODEL_NAME` | ❌ | `text-embedding-3-small` | Embedding model |

#### Neo4j Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEO4J_URI` | ❌ | `bolt://neo4j:7687` | Neo4j connection URI |
| `NEO4J_USER` | ❌ | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | ❌ | `demodemo` | Neo4j password |

#### Server Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MCP_SERVER_HOST` | ❌ | - | MCP server host binding |
| `SEMAPHORE_LIMIT` | ❌ | `10` | Concurrent operation limit for LLM calls |

#### Azure OpenAI Configuration (Optional)

For Azure OpenAI deployments, use these environment variables instead of the standard OpenAI configuration:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | ✅* | - | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | ✅* | - | Azure OpenAI API version |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | ✅* | - | Azure OpenAI deployment name |
| `AZURE_OPENAI_USE_MANAGED_IDENTITY` | ❌ | `false` | Use Azure managed identity for auth |
| `AZURE_OPENAI_EMBEDDING_ENDPOINT` | ❌ | - | Separate endpoint for embeddings |
| `AZURE_OPENAI_EMBEDDING_API_VERSION` | ❌ | - | API version for embeddings |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | ❌ | - | Deployment name for embeddings |
| `AZURE_OPENAI_EMBEDDING_API_KEY` | ❌ | - | Separate API key for embeddings |

\* Required when using Azure OpenAI

**Notes:**
- `OPENAI_BASE_URL` is consumed directly by the OpenAI Python SDK, useful for proxy configurations or custom endpoints
- `SEMAPHORE_LIMIT` controls concurrent LLM API calls - decrease if you encounter rate limits, increase for higher throughput
- Azure configuration is an alternative to standard OpenAI - don't mix both configurations

### Neo4j Settings 🗄️

Default configuration for Neo4j:
- Username: `neo4j`
- Password: `demodemo`
- URI: `bolt://neo4j:7687` (within Docker network)
- Memory settings optimized for development

### Docker Environment Variables 🐳

You can run with environment variables directly:
```bash
OPENAI_API_KEY=your_key MODEL_NAME=gpt-4.1-mini docker compose up
```

For Azure OpenAI:
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com \
AZURE_OPENAI_API_VERSION=2024-02-01 \
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment \
OPENAI_API_KEY=your_key \
docker compose up
```

## 🔌 Integration

### Cursor IDE Integration 🖥️

1. **Configure Cursor MCP settings:**
```json
{
  "mcpServers": {
    "Graphiti": {
      "command": "uv",
      "args": ["run", "graphiti_mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your_key_here"
      }
    }
  }
}
```

2. **For Docker-based setup:**
```json
{
  "mcpServers": {
    "Graphiti": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

3. **Add Graphiti rules to Cursor's User Rules** (see `graphiti_cursor_rules.mdc`)
4. **Start an agent session in Cursor**

### Other MCP Clients

The server supports standard MCP transports:
- **SSE (Server-Sent Events):** `http://localhost:8000/sse`
- **WebSocket:** `ws://localhost:8000/ws`
- **Stdio:** Direct process communication

## 💻 Development

### Local Development Setup

1. **Install dependencies:**
```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or using pip
pip install -r requirements.txt
```

2. **Start Neo4j locally:**
```bash
docker run -d \
  --name neo4j-dev \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/demodemo \
  neo4j:5.26.0
```

3. **Run the server:**
```bash
# Set environment variables
export OPENAI_API_KEY=your_key
export NEO4J_URI=bolt://localhost:7687

# Run with stdio transport
uv run graphiti_mcp_server.py

# Or with SSE transport
uv run graphiti_mcp_server.py --transport sse --use-custom-entities
```

### Testing

```bash
# Run basic connectivity test
curl http://localhost:8000/health

# Test MCP endpoint
curl http://localhost:8000/sse
```

## 🔍 Troubleshooting

### Common Issues

#### 🐳 Docker Issues
```bash
# Clean up and restart
docker compose down -v
docker compose up --build

# Check disk space
docker system df
```

### Logs and Debugging

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f graphiti-mcp
docker compose logs -f neo4j

# Enable debug logging
docker compose up -e LOG_LEVEL=DEBUG
```

### Performance Issues

- **Memory:** Increase Neo4j heap size in `docker-compose.yml`
- **Storage:** Monitor Neo4j data volume usage
- **Network:** Check for firewall blocking ports 7474, 7687, 8000

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │    │  Graphiti MCP    │    │     Neo4j       │
│   (Cursor)      │◄──►│     Server       │◄──►│   Database      │
│                 │    │   (Port 8000)    │    │  (Port 7687)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   OpenAI API     │
                       │   (LLM Client)   │
                       └──────────────────┘
```

### Components

- **Neo4j Database:** Graph storage and querying
- **Graphiti MCP Server:** API layer and LLM operations
- **OpenAI Integration:** Entity extraction and semantic processing
- **MCP Protocol:** Standardized AI agent communication

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Neo4j](https://neo4j.com/) team for the amazing graph database
- [OpenAI](https://openai.com/) for their powerful LLM models
- [MCP community](https://modelcontextprotocol.io/) for the protocol specification
- [Graphiti Core](https://github.com/griptape-ai/graphiti) for the knowledge graph framework

---

**Need help?** Open an issue or check our [troubleshooting guide](#-troubleshooting) above.