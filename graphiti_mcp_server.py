#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

# Import tomllib with backward compatibility
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python 3.10

import anyio
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()

DEFAULT_LLM_MODEL = 'gpt-4.1'
SMALL_LLM_MODEL = 'gpt-4.1'
DEFAULT_EMBEDDER_MODEL = 'text-embedding-3-small'

# Semaphore limit for concurrent Graphiti operations.
# Decrease this if you're experiencing 429 rate limit errors from your LLM provider.
# Increase if you have high rate limits.
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))

# Timeout configuration for Azure OpenAI requests
AZURE_OPENAI_TIMEOUT = float(os.getenv('AZURE_OPENAI_TIMEOUT', 300.0))
AZURE_OPENAI_MAX_RETRIES = int(os.getenv('AZURE_OPENAI_MAX_RETRIES', 5))


class Requirement(BaseModel):
    """A Requirement represents a specific need, feature, or functionality that a product or service must fulfill.

    Always ensure an edge is created between the requirement and the project it belongs to, and clearly indicate on the
    edge that the requirement is a requirement.

    Instructions for identifying and extracting requirements:
    1. Look for explicit statements of needs or necessities ("We need X", "X is required", "X must have Y")
    2. Identify functional specifications that describe what the system should do
    3. Pay attention to non-functional requirements like performance, security, or usability criteria
    4. Extract constraints or limitations that must be adhered to
    5. Focus on clear, specific, and measurable requirements rather than vague wishes
    6. Capture the priority or importance if mentioned ("critical", "high priority", etc.)
    7. Include any dependencies between requirements when explicitly stated
    8. Preserve the original intent and scope of the requirement
    9. Categorize requirements appropriately based on their domain or function
    """

    project_name: str = Field(
        ...,
        description='The name of the project to which the requirement belongs.',
    )
    description: str = Field(
        ...,
        description='Description of the requirement. Only use information mentioned in the context to write this description.',
    )


class Preference(BaseModel):
    """A Preference represents a user's expressed like, dislike, or preference for something.

    Instructions for identifying and extracting preferences:
    1. Look for explicit statements of preference such as "I like/love/enjoy/prefer X" or "I don't like/hate/dislike X"
    2. Pay attention to comparative statements ("I prefer X over Y")
    3. Consider the emotional tone when users mention certain topics
    4. Extract only preferences that are clearly expressed, not assumptions
    5. Categorize the preference appropriately based on its domain (food, music, brands, etc.)
    6. Include relevant qualifiers (e.g., "likes spicy food" rather than just "likes food")
    7. Only extract preferences directly stated by the user, not preferences of others they mention
    8. Provide a concise but specific description that captures the nature of the preference
    """

    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')",
    )
    description: str = Field(
        ...,
        description='Brief description of the preference. Only use information mentioned in the context to write this description.',
    )


class Procedure(BaseModel):
    """A Procedure informing the agent what actions to take or how to perform in certain scenarios. Procedures are typically composed of several steps.

    Instructions for identifying and extracting procedures:
    1. Look for sequential instructions or steps ("First do X, then do Y")
    2. Identify explicit directives or commands ("Always do X when Y happens")
    3. Pay attention to conditional statements ("If X occurs, then do Y")
    4. Extract procedures that have clear beginning and end points
    5. Focus on actionable instructions rather than general information
    6. Preserve the original sequence and dependencies between steps
    7. Include any specified conditions or triggers for the procedure
    8. Capture any stated purpose or goal of the procedure
    9. Summarize complex procedures while maintaining critical details
    """

    description: str = Field(
        ...,
        description='Brief description of the procedure. Only use information mentioned in the context to write this description.',
    )


ENTITY_TYPES: dict[str, BaseModel] = {
    'Requirement': Requirement,  # type: ignore
    'Preference': Preference,  # type: ignore
    'Procedure': Procedure,  # type: ignore
}


# Type definitions for API responses
class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class NodeResult(TypedDict):
    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    status: str
    message: str


class HealthCheckResponse(TypedDict):
    status: str
    timestamp: str
    version: str | None
    services: dict[str, str]


def create_azure_credential_token_provider() -> Callable[[], str]:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, 'https://cognitiveservices.azure.com/.default'
    )
    return token_provider


def get_version() -> str | None:
    """Get the version from pyproject.toml if available."""
    try:
        pyproject_path = os.path.join(os.path.dirname(__file__), 'pyproject.toml')
        if os.path.exists(pyproject_path):
            with open(pyproject_path, 'rb') as f:
                data = tomllib.load(f)
                return data.get('project', {}).get('version')
    except Exception as e:
        logger.debug(f'Could not read version from pyproject.toml: {e}')
    return None


# Server configuration classes
# The configuration system has a hierarchy:
# - GraphitiConfig is the top-level configuration
#   - LLMConfig handles all OpenAI/LLM related settings
#   - EmbedderConfig manages embedding settings
#   - Neo4jConfig manages database connection details
#   - Various other settings like group_id and feature flags
# Configuration values are loaded from:
# 1. Default values in the class definitions
# 2. Environment variables (loaded via load_dotenv())
# 3. Command line arguments (which override environment variables)
class GraphitiLLMConfig(BaseModel):
    """Configuration for the LLM client.

    Centralizes all LLM-specific configuration parameters including API keys and model selection.
    """

    api_key: str | None = None
    model: str = DEFAULT_LLM_MODEL
    small_model: str = SMALL_LLM_MODEL
    temperature: float = 0.0
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiLLMConfig':
        """Create LLM configuration from environment variables."""
        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL

        # Get small_model from environment, or use default if not set or empty
        small_model_env = os.environ.get('SMALL_MODEL_NAME', '')
        small_model = small_model_env if small_model_env.strip() else SMALL_LLM_MODEL

        azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', None)
        azure_openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION', None)
        azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', None)
        azure_openai_use_managed_identity = (
                os.environ.get('AZURE_OPENAI_USE_MANAGED_IDENTITY', 'false').lower() == 'true'
        )

        # Validate and fix Azure OpenAI endpoint format if provided
        if azure_openai_endpoint:
            if not azure_openai_endpoint.endswith('/'):
                azure_openai_endpoint += '/'
                logger.info(f'Auto-corrected Azure OpenAI endpoint to include trailing slash: {azure_openai_endpoint}')
            if '.openai.azure.com' not in azure_openai_endpoint:
                logger.warning(f'Azure OpenAI endpoint may have incorrect format: {azure_openai_endpoint}')

        # Log Azure OpenAI configuration for debugging
        if azure_openai_endpoint:
            logger.info(f'Using Azure OpenAI endpoint: {azure_openai_endpoint}')
            logger.info(f'Azure OpenAI API version: {azure_openai_api_version}')
            logger.info(f'Azure OpenAI deployment name: {azure_openai_deployment_name}')

        if azure_openai_endpoint is None:
            # Setup for OpenAI API
            # Log if empty model was provided
            if model_env == '':
                logger.debug(
                    f'MODEL_NAME environment variable not set, using default: {DEFAULT_LLM_MODEL}'
                )
            elif not model_env.strip():
                logger.warning(
                    f'Empty MODEL_NAME environment variable, using default: {DEFAULT_LLM_MODEL}'
                )

            return cls(
                api_key=os.environ.get('OPENAI_API_KEY'),
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
            )
        else:
            # Setup for Azure OpenAI API
            # Log if empty deployment name was provided
            if azure_openai_deployment_name is None:
                logger.error('AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set')

                raise ValueError('AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set')
            if not azure_openai_use_managed_identity:
                # For Azure OpenAI, prioritize AZURE_OPENAI_API_KEY, then fall back to OPENAI_API_KEY
                api_key = os.environ.get('AZURE_OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY', None)
                if not api_key:
                    logger.error('Neither AZURE_OPENAI_API_KEY nor OPENAI_API_KEY environment variable is set')
            else:
                # Managed identity
                api_key = None

            return cls(
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_openai_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
            )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiLLMConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables."""
        # Start with environment-based config
        config = cls.from_env()

        # CLI arguments override environment variables when provided
        if hasattr(args, 'model') and args.model:
            # Only use CLI model if it's not empty
            if args.model.strip():
                config.model = args.model
            else:
                # Log that empty model was provided and default is used
                logger.warning(f'Empty model name provided, using default: {DEFAULT_LLM_MODEL}')

        if hasattr(args, 'small_model') and args.small_model:
            if args.small_model.strip():
                config.small_model = args.small_model
            else:
                logger.warning(f'Empty small_model name provided, using default: {SMALL_LLM_MODEL}')

        if hasattr(args, 'temperature') and args.temperature is not None:
            config.temperature = args.temperature

        return config

    def create_client(self) -> LLMClient:
        """Create an LLM client based on this configuration.

        Returns:
            LLMClient instance
        """

        if self.azure_openai_endpoint is not None:
            # Validate required Azure OpenAI parameters
            if not self.azure_openai_deployment_name:
                raise ValueError('AZURE_OPENAI_DEPLOYMENT_NAME is required when using Azure OpenAI')
            if not self.azure_openai_api_version:
                raise ValueError('AZURE_OPENAI_API_VERSION is required when using Azure OpenAI')
            # Azure OpenAI API setup
            if self.azure_openai_use_managed_identity:
                # Use managed identity for authentication
                try:
                    token_provider = create_azure_credential_token_provider()
                    return AzureOpenAILLMClient(
                        azure_client=AsyncAzureOpenAI(
                            azure_endpoint=self.azure_openai_endpoint,
                            azure_deployment=self.azure_openai_deployment_name,
                            api_version=self.azure_openai_api_version,
                            azure_ad_token_provider=token_provider,
                            timeout=AZURE_OPENAI_TIMEOUT,
                            max_retries=AZURE_OPENAI_MAX_RETRIES,
                        ),
                        config=LLMConfig(
                            api_key=self.api_key,
                            model=self.azure_openai_deployment_name,  # Use deployment name for Azure
                            small_model=self.small_model,
                            temperature=self.temperature,
                        ),
                    )
                except Exception as e:
                    logger.error(f'Failed to create Azure managed identity token provider: {str(e)}')
                    raise ValueError(f'Azure managed identity authentication failed: {str(e)}')
            elif self.api_key:
                # Use API key for authentication
                return AzureOpenAILLMClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
                        timeout=AZURE_OPENAI_TIMEOUT,
                        max_retries=AZURE_OPENAI_MAX_RETRIES,
                    ),
                    config=LLMConfig(
                        api_key=self.api_key,
                        model=self.azure_openai_deployment_name,  # Use deployment name for Azure
                        small_model=self.small_model,
                        temperature=self.temperature,
                    ),
                )
            else:
                raise ValueError(
                    'API key must be set when using Azure OpenAI API (set AZURE_OPENAI_API_KEY or OPENAI_API_KEY)')

        if not self.api_key:
            raise ValueError('OPENAI_API_KEY must be set when using OpenAI API')

        llm_client_config = LLMConfig(
            api_key=self.api_key, model=self.model, small_model=self.small_model
        )

        # Set temperature
        llm_client_config.temperature = self.temperature

        return OpenAIClient(config=llm_client_config)


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client.

    Centralizes all embedding-related configuration parameters.
    """

    model: str = DEFAULT_EMBEDDER_MODEL
    api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiEmbedderConfig':
        """Create embedder configuration from environment variables."""

        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('EMBEDDER_MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_EMBEDDER_MODEL

        # Use the same Azure OpenAI configuration as LLM for embeddings
        azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', None)
        azure_openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION', None)
        azure_openai_use_managed_identity = (
                os.environ.get('AZURE_OPENAI_USE_MANAGED_IDENTITY', 'false').lower() == 'true'
        )

        # Validate and fix Azure OpenAI endpoint format if provided  
        if azure_openai_endpoint:
            if not azure_openai_endpoint.endswith('/'):
                azure_openai_endpoint += '/'
                logger.debug(f'Auto-corrected Azure OpenAI endpoint for embeddings: {azure_openai_endpoint}')
            if '.openai.azure.com' not in azure_openai_endpoint:
                logger.warning(f'Azure OpenAI endpoint may have incorrect format: {azure_openai_endpoint}')

        if azure_openai_endpoint is not None:
            # Setup for Azure OpenAI API
            # Log if empty deployment name was provided
            azure_openai_deployment_name = os.environ.get(
                'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', None
            )
            if azure_openai_deployment_name is None:
                logger.error('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set')

                raise ValueError(
                    'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set'
                )

            if not azure_openai_use_managed_identity:
                # Use the same API key as LLM configuration
                api_key = os.environ.get('AZURE_OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY', None)
                if not api_key:
                    logger.error('No API key found in AZURE_OPENAI_API_KEY or OPENAI_API_KEY')
            else:
                # Managed identity
                api_key = None

            return cls(
                model=model,  # Include model for Azure OpenAI embeddings
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_openai_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
            )
        else:
            return cls(
                model=model,
                api_key=os.environ.get('OPENAI_API_KEY'),
            )

    def create_client(self) -> EmbedderClient | None:
        if self.azure_openai_endpoint is not None:
            # Validate required Azure OpenAI parameters for embeddings
            if not self.azure_openai_deployment_name:
                logger.error('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is required when using Azure OpenAI embeddings')
                raise ValueError(
                    'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is required when using Azure OpenAI embeddings')
            if not self.azure_openai_api_version:
                logger.error('AZURE_OPENAI_API_VERSION is required when using Azure OpenAI embeddings')
                raise ValueError('AZURE_OPENAI_API_VERSION is required when using Azure OpenAI embeddings')
            # Azure OpenAI API setup
            if self.azure_openai_use_managed_identity:
                # Use managed identity for authentication
                try:
                    token_provider = create_azure_credential_token_provider()
                    return AzureOpenAIEmbedderClient(
                        azure_client=AsyncAzureOpenAI(
                            azure_endpoint=self.azure_openai_endpoint,
                            azure_deployment=self.azure_openai_deployment_name,
                            api_version=self.azure_openai_api_version,
                            azure_ad_token_provider=token_provider,
                            timeout=AZURE_OPENAI_TIMEOUT,
                            max_retries=AZURE_OPENAI_MAX_RETRIES,
                        ),
                        model=self.azure_openai_deployment_name,  # Use deployment name for Azure
                    )
                except Exception as e:
                    logger.error(f'Failed to create Azure managed identity token provider for embeddings: {str(e)}')
                    raise ValueError(f'Azure managed identity authentication failed for embeddings: {str(e)}')
            elif self.api_key:
                # Use API key for authentication
                return AzureOpenAIEmbedderClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
                        timeout=AZURE_OPENAI_TIMEOUT,
                        max_retries=AZURE_OPENAI_MAX_RETRIES,
                    ),
                    model=self.azure_openai_deployment_name,  # Use deployment name for Azure
                )
            else:
                logger.error(
                    'API key must be set when using Azure OpenAI API (set AZURE_OPENAI_API_KEY or OPENAI_API_KEY)')
                return None
        else:
            # OpenAI API setup
            if not self.api_key:
                return None

            embedder_config = OpenAIEmbedderConfig(api_key=self.api_key, embedding_model=self.model)

            return OpenAIEmbedder(config=embedder_config)


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j database connection."""

    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'password'

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create Neo4j configuration from environment variables."""
        return cls(
            uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.environ.get('NEO4J_USER', 'neo4j'),
            password=os.environ.get('NEO4J_PASSWORD', 'password'),
        )


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client.

    Centralizes all configuration parameters for the Graphiti client.
    """

    llm: GraphitiLLMConfig = Field(default_factory=GraphitiLLMConfig)
    embedder: GraphitiEmbedderConfig = Field(default_factory=GraphitiEmbedderConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    group_id: str | None = None
    use_custom_entities: bool = False
    destroy_graph: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiConfig':
        """Create a configuration instance from environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_env(),
            embedder=GraphitiEmbedderConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiConfig':
        """Create configuration from CLI arguments, falling back to environment variables."""
        # Start with environment configuration
        config = cls.from_env()

        # Apply CLI overrides
        if args.group_id:
            config.group_id = args.group_id
        else:
            config.group_id = 'default'

        config.use_custom_entities = args.use_custom_entities
        config.destroy_graph = args.destroy_graph

        # Update LLM config using CLI args
        config.llm = GraphitiLLMConfig.from_cli_and_env(args)

        return config


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = 'sse'  # Default to SSE transport

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> 'MCPConfig':
        """Create MCP configuration from CLI arguments."""
        return cls(transport=args.transport)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create global config instance - will be properly initialized later
config = GraphitiConfig()

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to 
capture relationships between concepts, entities, and information. The system organizes data as episodes 
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic, 
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including 
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid 
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations. 
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality. 
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid 
API keys are provided for any language model operations.
"""

# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS
)

# Initialize Graphiti client and initialization status
graphiti_client: Graphiti | None = None
initialization_complete: bool = False
initialization_lock = asyncio.Lock()


async def initialize_graphiti():
    """Initialize the Graphiti client with the configured settings."""
    global graphiti_client, config, initialization_complete

    try:
        # Create LLM client if possible
        llm_client = config.llm.create_client()
        if not llm_client and config.use_custom_entities:
            # If custom entities are enabled, we must have an LLM client
            raise ValueError('OPENAI_API_KEY must be set when custom entities are enabled')

        # Validate Neo4j configuration
        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

        embedder_client = config.embedder.create_client()

        # Initialize Graphiti client
        graphiti_client = Graphiti(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            llm_client=llm_client,
            embedder=embedder_client,
            max_coroutines=SEMAPHORE_LIMIT,
        )

        # Destroy graph if requested
        if config.destroy_graph:
            logger.info('Destroying graph...')
            await clear_data(graphiti_client.driver)

        # Initialize the graph database with Graphiti's indices
        await graphiti_client.build_indices_and_constraints()
        logger.info('Graphiti client initialized successfully')

        # Log configuration details for transparency
        if llm_client:
            logger.info(f'Using OpenAI model: {config.llm.model}')
            logger.info(f'Using temperature: {config.llm.temperature}')
        else:
            logger.info('No LLM client configured - entity extraction will be limited')

        logger.info(f'Using group_id: {config.group_id}')
        logger.info(
            f'Custom entity extraction: {"enabled" if config.use_custom_entities else "disabled"}'
        )
        logger.info(f'Using concurrency limit: {SEMAPHORE_LIMIT}')

        # Mark initialization as complete
        initialization_complete = True
        logger.info('Graphiti initialization completed successfully')

    except Exception as e:
        logger.error(f'Failed to initialize Graphiti: {str(e)}')
        initialization_complete = False
        raise


async def ensure_initialization():
    """Ensure Graphiti is fully initialized before processing requests."""
    global initialization_complete, initialization_lock

    if initialization_complete:
        return

    async with initialization_lock:
        if not initialization_complete:
            logger.warning("Request received before initialization complete, waiting...")
            # Wait longer for initialization to complete
            max_wait_time = 30  # 30 seconds maximum wait
            wait_interval = 0.5  # Check every 500ms
            total_waited = 0

            while not initialization_complete and total_waited < max_wait_time:
                await asyncio.sleep(wait_interval)
                total_waited += wait_interval

            if not initialization_complete:
                raise RuntimeError(f"Graphiti initialization not complete after {max_wait_time} seconds")


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    result = edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
        },
    )
    result.get('attributes', {}).pop('fact_embedding', None)
    return result


def handle_sse_errors(func):
    """Decorator to handle SSE connection errors gracefully."""

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except anyio.ClosedResourceError:
            logger.debug("SSE connection closed during tool execution")
            # Return appropriate error response based on function return type
            return ErrorResponse(error="Connection closed")
        except Exception as e:
            # Re-raise other exceptions normally
            raise e

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# Dictionary to store queues for each group_id
# Each queue is a list of tasks to be processed sequentially
episode_queues: dict[str, asyncio.Queue] = {}
# Dictionary to track if a worker is running for each group_id
queue_workers: dict[str, bool] = {}


async def process_episode_queue(group_id: str):
    """Process episodes for a specific group_id sequentially.

    This function runs as a long-lived task that processes episodes
    from the queue one at a time.
    """
    global queue_workers

    logger.info(f'Starting episode queue worker for group_id: {group_id}')
    queue_workers[group_id] = True

    try:
        while True:
            # Get the next episode processing function from the queue
            # This will wait if the queue is empty
            process_func = await episode_queues[group_id].get()

            try:
                # Process the episode
                await process_func()
            except Exception as e:
                logger.error(f'Error processing queued episode for group_id {group_id}: {str(e)}')
            finally:
                # Mark the task as done regardless of success/failure
                episode_queues[group_id].task_done()
    except asyncio.CancelledError:
        logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
    except Exception as e:
        logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
    finally:
        queue_workers[group_id] = False
        logger.info(f'Stopped episode queue worker for group_id: {group_id}')


@mcp.tool()
async def add_memory(
        name: str,
        episode_body: str,
        group_id: str | None = None,
        source: str = 'text',
        source_description: str = '',
        uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode to persist to memory. When source='json', this must be a
                           properly escaped JSON string, not a raw Python dictionary. The JSON data will be
                           automatically processed to extract entities and relationships.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

    Examples:
        # Adding plain text content
        add_memory(
            name="Company News",
            episode_body="Acme Corp announced a new product line today.",
            source="text",
            source_description="news article",
            group_id="some_arbitrary_string"
        )

        # Adding structured JSON data
        # NOTE: episode_body must be a properly escaped JSON string. Note the triple backslashes
        add_memory(
            name="Customer Profile",
            episode_body="{\\\"company\\\": {\\\"name\\\": \\\"Acme Technologies\\\"}, \\\"products\\\": [{\\\"id\\\": \\\"P001\\\", \\\"name\\\": \\\"CloudSync\\\"}, {\\\"id\\\": \\\"P002\\\", \\\"name\\\": \\\"DataMiner\\\"}]}",
            source="json",
            source_description="CRM data"
        )

        # Adding message-style content
        add_memory(
            name="Customer Conversation",
            episode_body="user: What's your return policy?\nassistant: You can return items within 30 days.",
            source="message",
            source_description="chat transcript",
            group_id="some_arbitrary_string"
        )

    Notes:
        When using source='json':
        - The JSON must be a properly escaped string, not a raw Python dictionary
        - The JSON will be automatically processed to extract entities and relationships
        - Complex nested structures are supported (arrays, nested objects, mixed data types), but keep nesting to a minimum
        - Entities will be created from appropriate JSON properties
        - Relationships between entities will be established based on the JSON structure
    """
    global graphiti_client, episode_queues, queue_workers

    try:
        await ensure_initialization()
    except RuntimeError as e:
        return ErrorResponse(error=str(e))

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Map string source to EpisodeType enum
        source_type = EpisodeType.text
        if source.lower() == 'message':
            source_type = EpisodeType.message
        elif source.lower() == 'json':
            source_type = EpisodeType.json

        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        # Cast group_id to str to satisfy type checker
        # The Graphiti client expects a str for group_id, not Optional[str]
        group_id_str = str(effective_group_id) if effective_group_id is not None else ''

        # We've already checked that graphiti_client is not None above
        # This assert statement helps type checkers understand that graphiti_client is defined
        assert graphiti_client is not None, 'graphiti_client should not be None here'

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Define the episode processing function
        async def process_episode():
            max_retries = 5
            retry_delay = 10  # seconds

            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Processing queued episode '{name}' for group_id: {group_id_str} (attempt {attempt + 1}/{max_retries})")
                    # Use all entity types if use_custom_entities is enabled, otherwise use empty dict
                    entity_types = ENTITY_TYPES if config.use_custom_entities else {}

                    # Add timeout for the add_episode operation - increased timeout
                    await asyncio.wait_for(
                        client.add_episode(
                            name=name,
                            episode_body=episode_body,
                            source=source_type,
                            source_description=source_description,
                            group_id=group_id_str,  # Using the string version of group_id
                            uuid=uuid,
                            reference_time=datetime.now(timezone.utc),
                            entity_types=entity_types,
                        ),
                        timeout=600.0  # 10 minutes timeout for the entire operation
                    )
                    logger.info(f"Episode '{name}' added successfully")
                    return  # Success, exit the retry loop

                except asyncio.TimeoutError:
                    error_msg = f"Timeout processing episode '{name}' on attempt {attempt + 1}"
                    logger.warning(error_msg)
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 60)  # Exponential backoff with cap
                    else:
                        logger.error(f"Failed to process episode '{name}' after {max_retries} attempts: {error_msg}")

                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Error processing episode '{name}' on attempt {attempt + 1}: {error_msg}")

                    # Enhanced error categorization
                    is_timeout_related = any(keyword in error_msg.lower() for keyword in [
                        "timeout", "timed out", "connection timeout", "read timeout",
                        "request timeout", "server timeout"
                    ])
                    is_rate_limit = any(keyword in error_msg.lower() for keyword in [
                        "rate limit", "too many requests", "429", "quota exceeded"
                    ])
                    is_network_error = any(keyword in error_msg.lower() for keyword in [
                        "connection", "network", "dns", "ssl", "certificate"
                    ])
                    is_deployment_error = any(keyword in error_msg.lower() for keyword in [
                        "404", "resource not found", "deployment", "not found"
                    ])
                    is_auth_error = any(keyword in error_msg.lower() for keyword in [
                        "401", "access denied", "invalid subscription key", "permission denied"
                    ])

                    if attempt < max_retries - 1:
                        if is_timeout_related or is_rate_limit or is_network_error:
                            # Longer delay for rate limits
                            delay = retry_delay * 2 if is_rate_limit else retry_delay
                            logger.info(
                                f"Retrying in {delay} seconds... (Error type: {'rate_limit' if is_rate_limit else 'timeout/network'})")
                            await asyncio.sleep(delay)
                            retry_delay = min(retry_delay * 1.5, 60)  # Exponential backoff with cap
                        elif is_deployment_error:
                            # Azure deployment errors are not retryable
                            logger.error(f"Azure OpenAI deployment error processing episode '{name}': {error_msg}")
                            logger.error("Please check your Azure OpenAI deployment configuration:")
                            logger.error(f"  - Endpoint: {config.llm.azure_openai_endpoint}")
                            logger.error(f"  - Deployment name: {config.llm.azure_openai_deployment_name}")
                            logger.error(f"  - API version: {config.llm.azure_openai_api_version}")
                            logger.error(
                                "Ensure the deployment exists and is properly configured in Azure OpenAI Studio")
                            break
                        elif is_auth_error:
                            # Authentication errors are not retryable
                            logger.error(f"Azure OpenAI authentication error processing episode '{name}': {error_msg}")
                            logger.error("Please check your Azure OpenAI authentication configuration:")
                            logger.error(f"  - Endpoint: {config.llm.azure_openai_endpoint}")
                            logger.error(f"  - Using managed identity: {config.llm.azure_openai_use_managed_identity}")
                            if not config.llm.azure_openai_use_managed_identity:
                                api_key_configured = "✅ Set" if config.llm.api_key else "❌ Not set"
                                logger.error(f"  - API key: {api_key_configured}")
                                if config.llm.api_key:
                                    logger.error(f"  - API key length: {len(config.llm.api_key)} characters")
                            logger.error("Solutions:")
                            logger.error("  1. Verify API key is correct and active")
                            logger.error("  2. Check if endpoint region matches your subscription")
                            logger.error("  3. Ensure subscription has access to Azure OpenAI")
                            logger.error("  4. Test with curl command to verify credentials")
                            break
                        else:
                            # For other non-retryable errors, fail immediately
                            logger.error(f"Non-retryable error processing episode '{name}': {error_msg}")
                            break
                    else:
                        logger.error(f"Failed to process episode '{name}' after {max_retries} attempts: {error_msg}")

        # Initialize queue for this group_id if it doesn't exist
        if group_id_str not in episode_queues:
            episode_queues[group_id_str] = asyncio.Queue()

        # Add the episode processing function to the queue
        await episode_queues[group_id_str].put(process_episode)

        # Start a worker for this queue if one isn't already running
        if not queue_workers.get(group_id_str, False):
            asyncio.create_task(process_episode_queue(group_id_str))

        # Return immediately with a success message
        return SuccessResponse(
            message=f"Episode '{name}' queued for processing (position: {episode_queues[group_id_str].qsize()})"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode task: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode task: {error_msg}')


def validate_and_clean_input(value, param_name, expected_type=None, min_value=None, max_value=None):
    """Validate and clean input parameters.
    
    Args:
        value: The value to validate
        param_name: Name of the parameter for error messages
        expected_type: Expected type (e.g., str, int, list)
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
    
    Returns:
        Cleaned and validated value
    
    Raises:
        ValueError: If validation fails
    """
    if value is None:
        return None

    # Type validation and conversion
    if expected_type:
        if expected_type == str and not isinstance(value, str):
            if hasattr(value, '__str__'):
                value = str(value)
            else:
                raise ValueError(f"{param_name} must be a string")
        elif expected_type == int:
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    raise ValueError(f"{param_name} must be an integer")
            elif not isinstance(value, int):
                raise ValueError(f"{param_name} must be an integer")
        elif expected_type == list and not isinstance(value, list):
            raise ValueError(f"{param_name} must be a list")

    # Range validation for numeric types
    if isinstance(value, (int, float)):
        if min_value is not None and value < min_value:
            raise ValueError(f"{param_name} must be >= {min_value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"{param_name} must be <= {max_value}")

    # String cleaning
    if isinstance(value, str):
        value = value.strip()
        if not value and param_name != 'entity':  # entity can be empty string
            return None

    return value


def create_error_response(error_msg: str, details: dict = None) -> ErrorResponse:
    """Create a standardized error response with optional details.
    
    Args:
        error_msg: The main error message
        details: Optional dictionary of additional error details
    
    Returns:
        ErrorResponse with formatted error message
    """
    if details:
        detailed_msg = f"{error_msg}. Details: {details}"
    else:
        detailed_msg = error_msg

    logger.error(f"API Error: {detailed_msg}")
    return ErrorResponse(error=detailed_msg)


def parse_mcp_parameters(args, kwargs, extra_kwargs, param_defaults):
    """Enhanced parameter parsing with better error handling and fallback logic.
    
    Args:
        args: Raw args parameter (could be string, dict, or None)
        kwargs: Raw kwargs parameter (could be string, dict, or None)
        extra_kwargs: Additional keyword arguments
        param_defaults: Dictionary of default values for parameters
    
    Returns:
        Dictionary of parsed parameters
    """
    import json

    # Start with defaults
    parsed_params = param_defaults.copy()

    # Helper function to safely update parameters
    def safe_update_params(source_dict, context=""):
        for key, value in source_dict.items():
            if key in parsed_params:
                try:
                    # Validate and clean the value based on parameter type
                    if key == 'query':
                        cleaned_value = validate_and_clean_input(value, key, str)
                    elif key in ['max_nodes', 'max_facts']:
                        cleaned_value = validate_and_clean_input(value, key, int, min_value=1, max_value=100)
                    elif key == 'group_ids':
                        cleaned_value = validate_and_clean_input(value, key, list)
                    elif key in ['center_node_uuid', 'entity']:
                        cleaned_value = validate_and_clean_input(value, key, str)
                    else:
                        cleaned_value = value

                    # Only update if the cleaned value is valid
                    if cleaned_value is not None and (cleaned_value != '' or key in ['entity']):
                        parsed_params[key] = cleaned_value
                        logger.debug(f"Updated {key} from {context}: {cleaned_value}")

                except ValueError as e:
                    logger.warning(f"Validation error for {key} from {context}: {e}")
                    # Continue with default value
                    continue

    # Handle args parameter
    if args is not None:
        if isinstance(args, str):
            # Try JSON parsing first
            try:
                args_dict = json.loads(args)
                safe_update_params(args_dict, "args JSON")
            except (json.JSONDecodeError, TypeError, AttributeError):
                # Fallback: treat as query string if it's a simple string
                if args.strip():
                    logger.info(f"Treating args as query string: {args}")
                    try:
                        cleaned_query = validate_and_clean_input(args, 'query', str)
                        if cleaned_query:
                            parsed_params['query'] = cleaned_query
                    except ValueError as e:
                        logger.warning(f"Query validation error: {e}")
                else:
                    logger.warning(f"Failed to parse args as JSON and string is empty: {args}")
        elif isinstance(args, dict):
            safe_update_params(args, "args dict")
        else:
            logger.warning(f"Unexpected args type: {type(args)}, value: {args}")

    # Handle kwargs parameter
    if kwargs is not None:
        if isinstance(kwargs, str):
            try:
                kwargs_dict = json.loads(kwargs)
                safe_update_params(kwargs_dict, "kwargs JSON")
            except (json.JSONDecodeError, TypeError, AttributeError):
                logger.warning(f"Failed to parse kwargs as JSON: {kwargs}")
        elif isinstance(kwargs, dict):
            safe_update_params(kwargs, "kwargs dict")
        else:
            logger.warning(f"Unexpected kwargs type: {type(kwargs)}, value: {kwargs}")

    # Handle extra kwargs
    if extra_kwargs:
        safe_update_params(extra_kwargs, "extra_kwargs")

    return parsed_params


@mcp.tool()
@handle_sse_errors
async def search_memory_nodes(
        query: str = None,
        group_ids: list[str] | None = None,
        max_nodes: int = 10,
        center_node_uuid: str | None = None,
        entity: str = '',  # cursor seems to break with None
        # Support for Cursor's MCP wrapper format
        args: str = None,
        kwargs: str = None,
        **extra_kwargs
) -> NodeSearchResponse | ErrorResponse:
    """Search the graph memory for relevant node summaries.
    These contain a summary of all of a node's relationships with other nodes.

    Note: entity is a single entity type to filter results (permitted: "Preference", "Procedure").

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_nodes: Maximum number of nodes to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
        entity: Optional single entity type to filter results (permitted: "Preference", "Procedure")
    """
    global graphiti_client

    # Enhanced parameter parsing with fallback logic
    param_defaults = {
        'query': query,
        'group_ids': group_ids,
        'max_nodes': max_nodes,
        'center_node_uuid': center_node_uuid,
        'entity': entity
    }

    try:
        parsed_params = parse_mcp_parameters(args, kwargs, extra_kwargs, param_defaults)
        query = parsed_params['query']
        group_ids = parsed_params['group_ids']
        max_nodes = parsed_params['max_nodes']
        center_node_uuid = parsed_params['center_node_uuid']
        entity = parsed_params['entity']
    except Exception as e:
        logger.error(f"Error parsing parameters: {e}")
        # Continue with original parameters as fallback
        pass

    # Validate and clean required parameter
    if query is None or (isinstance(query, str) and not query.strip()):
        return ErrorResponse(error='query parameter is required and cannot be empty')

    try:
        await ensure_initialization()
    except RuntimeError as e:
        return ErrorResponse(error=str(e))

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        # Configure the search
        if center_node_uuid is not None:
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
        else:
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        filters = SearchFilters()
        if entity != '':
            filters.node_labels = [entity]

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Perform the search using the _search method with timeout
        try:
            search_results = await asyncio.wait_for(
                client._search(
                    query=query,
                    config=search_config,
                    group_ids=effective_group_ids,
                    center_node_uuid=center_node_uuid,
                    search_filter=filters,
                ),
                timeout=60.0  # 60 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f'Node search timed out for query: {query}')
            return ErrorResponse(error='Search operation timed out')

        if not search_results.nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        # Format the node results
        formatted_nodes: list[NodeResult] = [
            {
                'uuid': node.uuid,
                'name': node.name,
                'summary': node.summary if hasattr(node, 'summary') else '',
                'labels': node.labels if hasattr(node, 'labels') else [],
                'group_id': node.group_id,
                'created_at': node.created_at.isoformat(),
                'attributes': node.attributes if hasattr(node, 'attributes') else {},
            }
            for node in search_results.nodes
        ]

        return NodeSearchResponse(message='Nodes retrieved successfully', nodes=formatted_nodes)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')


@mcp.tool()
@handle_sse_errors
async def search_memory_facts(
        query: str = None,
        group_ids: list[str] | None = None,
        max_facts: int = 10,
        center_node_uuid: str | None = None,
        # Support for Cursor's MCP wrapper format
        args: str = None,
        kwargs: str = None,
        **extra_kwargs
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    global graphiti_client

    # Enhanced parameter parsing with fallback logic
    param_defaults = {
        'query': query,
        'group_ids': group_ids,
        'max_facts': max_facts,
        'center_node_uuid': center_node_uuid
    }

    try:
        parsed_params = parse_mcp_parameters(args, kwargs, extra_kwargs, param_defaults)
        query = parsed_params['query']
        group_ids = parsed_params['group_ids']
        max_facts = parsed_params['max_facts']
        center_node_uuid = parsed_params['center_node_uuid']
    except Exception as e:
        logger.error(f"Error parsing parameters: {e}")
        # Continue with original parameters as fallback
        pass

    # Validate and clean required parameter
    if query is None or (isinstance(query, str) and not query.strip()):
        return ErrorResponse(error='query parameter is required and cannot be empty')

    try:
        await ensure_initialization()
    except RuntimeError as e:
        return ErrorResponse(error=str(e))

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Additional validation for max_facts parameter
        try:
            max_facts = validate_and_clean_input(max_facts, 'max_facts', int, min_value=1, max_value=100)
        except ValueError as e:
            return ErrorResponse(error=f'Invalid max_facts parameter: {str(e)}')

        if max_facts is None or max_facts <= 0:
            max_facts = 10  # Default value
            logger.warning("Invalid max_facts, using default value: 10")

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        try:
            relevant_edges = await asyncio.wait_for(
                client.search(
                    group_ids=effective_group_ids,
                    query=query,
                    num_results=max_facts,
                    center_node_uuid=center_node_uuid,
                ),
                timeout=60.0  # 60 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f'Facts search timed out for query: {query}')
            return ErrorResponse(error='Facts search operation timed out')

        if not relevant_edges:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return FactSearchResponse(message='Facts retrieved successfully', facts=facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching facts: {error_msg}')
        return ErrorResponse(error=f'Error searching facts: {error_msg}')


@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
    """
    global graphiti_client

    try:
        await ensure_initialization()
    except RuntimeError as e:
        return ErrorResponse(error=str(e))

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')


@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Args:
        uuid: UUID of the episode to delete
    """
    global graphiti_client

    try:
        await ensure_initialization()
    except RuntimeError as e:
        return ErrorResponse(error=str(e))

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the episodic node by UUID - EpisodicNode is already imported at the top
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')


@mcp.tool()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    global graphiti_client

    try:
        await ensure_initialization()
    except RuntimeError as e:
        return ErrorResponse(error=str(e))

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error getting entity edge: {error_msg}')


@mcp.tool()
async def get_episodes(
        group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent memory episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from. If not provided, uses the default group_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    global graphiti_client

    try:
        await ensure_initialization()
    except RuntimeError as e:
        return ErrorResponse(error=str(e))

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        if not isinstance(effective_group_id, str):
            return ErrorResponse(error='Group ID must be a string')

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        episodes = await client.retrieve_episodes(
            group_ids=[effective_group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
        )

        if not episodes:
            return EpisodeSearchResponse(
                message=f'No episodes found for group {effective_group_id}', episodes=[]
            )

        # Use Pydantic's model_dump method for EpisodicNode serialization
        formatted_episodes = [
            # Use mode='json' to handle datetime serialization
            episode.model_dump(mode='json')
            for episode in episodes
        ]

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')


@mcp.tool()
async def clear_graph() -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph memory and rebuild indices."""
    global graphiti_client

    try:
        await ensure_initialization()
    except RuntimeError as e:
        return ErrorResponse(error=str(e))

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # clear_data is already imported at the top
        await clear_data(client.driver)
        await client.build_indices_and_constraints()
        return SuccessResponse(message='Graph cleared successfully and indices rebuilt')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')


@mcp.resource('http://graphiti/status')
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    global graphiti_client

    try:
        await ensure_initialization()
    except RuntimeError as e:
        return StatusResponse(status='error', message=str(e))

    if graphiti_client is None:
        return StatusResponse(status='error', message='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Test database connection
        await client.driver.client.verify_connectivity()  # type: ignore

        return StatusResponse(
            status='ok', message='Graphiti MCP server is running and connected to Neo4j'
        )
    except anyio.ClosedResourceError:
        logger.debug("SSE connection closed during status check")
        return StatusResponse(status='error', message='Connection closed')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking Neo4j connection: {error_msg}')
        return StatusResponse(
            status='error',
            message=f'Graphiti MCP server is running but Neo4j connection failed: {error_msg}',
        )


@mcp.resource('http://graphiti/healthcheck')
async def healthcheck() -> HealthCheckResponse:
    """Health check endpoint for service monitoring and container orchestration."""
    global graphiti_client

    # Get current timestamp
    timestamp = datetime.now(timezone.utc).isoformat()

    # Get version information
    version = get_version()

    # Initialize service status dict
    services = {}

    try:
        # Check if basic initialization is complete
        if not initialization_complete:
            services['initialization'] = 'initializing'
            return HealthCheckResponse(
                status='starting',
                timestamp=timestamp,
                version=version,
                services=services
            )

        services['initialization'] = 'ready'

        # Check Graphiti client status
        if graphiti_client is None:
            services['graphiti'] = 'unavailable'
            services['neo4j'] = 'unknown'
        else:
            services['graphiti'] = 'ready'

            # Test Neo4j connection (with timeout to avoid hanging)
            try:
                client = cast(Graphiti, graphiti_client)
                await asyncio.wait_for(
                    client.driver.client.verify_connectivity(),  # type: ignore
                    timeout=5.0
                )
                services['neo4j'] = 'healthy'
            except asyncio.TimeoutError:
                services['neo4j'] = 'timeout'
            except Exception as e:
                logger.debug(f'Neo4j connection check failed: {e}')
                services['neo4j'] = 'unhealthy'

        # Determine overall status
        if all(status in ['ready', 'healthy'] for status in services.values()):
            overall_status = 'healthy'
        elif any(status in ['unavailable', 'unhealthy'] for status in services.values()):
            overall_status = 'unhealthy'
        else:
            overall_status = 'degraded'

        return HealthCheckResponse(
            status=overall_status,
            timestamp=timestamp,
            version=version,
            services=services
        )

    except Exception as e:
        logger.error(f'Health check failed: {e}')
        services['healthcheck'] = 'error'
        return HealthCheckResponse(
            status='unhealthy',
            timestamp=timestamp,
            version=version,
            services=services
        )


# Add root path handler to reduce 404 noise
# Note: FastMCP doesn't expose direct app access, so we'll handle this differently
# The 404 errors are normal for health checks and don't affect functionality


async def initialize_server() -> MCPConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config, initialization_lock

    async with initialization_lock:
        parser = argparse.ArgumentParser(
            description='Run the Graphiti MCP server with optional LLM client'
        )
        parser.add_argument(
            '--group-id',
            help='Namespace for the graph. This is an arbitrary string used to organize related data. '
                 'If not provided, a random UUID will be generated.',
        )
        parser.add_argument(
            '--transport',
            choices=['sse', 'stdio'],
            default='sse',
            help='Transport to use for communication with the client. (default: sse)',
        )
        parser.add_argument(
            '--model', help=f'Model name to use with the LLM client. (default: {DEFAULT_LLM_MODEL})'
        )
        parser.add_argument(
            '--small-model',
            help=f'Small model name to use with the LLM client. (default: {SMALL_LLM_MODEL})',
        )
        parser.add_argument(
            '--temperature',
            type=float,
            help='Temperature setting for the LLM (0.0-2.0). Lower values make output more deterministic. (default: 0.7)',
        )
        parser.add_argument('--destroy-graph', action='store_true', help='Destroy all Graphiti graphs')
        parser.add_argument(
            '--use-custom-entities',
            action='store_true',
            help='Enable entity extraction using the predefined ENTITY_TYPES',
        )
        parser.add_argument(
            '--host',
            default=os.environ.get('MCP_SERVER_HOST'),
            help='Host to bind the MCP server to (default: MCP_SERVER_HOST environment variable)',
        )

        args = parser.parse_args()

        # Build configuration from CLI arguments and environment variables
        config = GraphitiConfig.from_cli_and_env(args)

        # Log the group ID configuration
        if args.group_id:
            logger.info(f'Using provided group_id: {config.group_id}')
        else:
            logger.info(f'Generated random group_id: {config.group_id}')

        # Log entity extraction configuration
        if config.use_custom_entities:
            logger.info('Entity extraction enabled using predefined ENTITY_TYPES')
        else:
            logger.info('Entity extraction disabled (no custom entities will be used)')

        # Initialize Graphiti
        await initialize_graphiti()

        if args.host:
            logger.info(f'Setting MCP server host to: {args.host}')
            # Set MCP server host from CLI or env
            mcp.settings.host = args.host

        # Set port to match docker-compose configuration
        mcp.settings.port = int(os.environ.get('MCP_SERVER_PORT', '8000'))

        # Return MCP configuration
        return MCPConfig.from_cli(args)


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with stdio transport for MCP in the same event loop
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'sse':
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        await mcp.run_sse_async()


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise


if __name__ == '__main__':
    main()
