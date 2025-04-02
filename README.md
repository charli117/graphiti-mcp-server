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
MODEL_NAME=gpt-4o
```

4. Start the services:
```bash
docker compose up
```

## 🔧 Configuration

### Neo4j Settings 🗄️

Default configuration for Neo4j:
- Username: `neo4j`
- Password: `demodemo`
- URI: `bolt://neo4j:7687` (within Docker network)
- Memory settings optimized for development

### Docker Environment Variables 🐳

You can run with environment variables directly:
```bash
OPENAI_API_KEY=your_key MODEL_NAME=gpt-4o docker compose up
```

## 🔌 Integration

### Cursor IDE Integration 🖥️

1. Configure Cursor to connect to Graphiti:
```json
{
  "mcpServers": {
    "Graphiti": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

2. Add Graphiti rules to Cursor's User Rules (see `graphiti_cursor_rules.md`)
3. Start an agent session in Cursor

## 🏗️ Architecture

The server consists of two main components:
- Neo4j database for graph storage
- Graphiti MCP server for API and LLM operations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Neo4j team for the amazing graph database
- OpenAI for their powerful LLM models
- MCP community for the protocol specification