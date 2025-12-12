"""Stirrup - Artificial Analysis 的轻量级AI代理框架，最初为评估测试而构建，简单易用且易于扩展。

这是一个现代化的AI代理开发框架，提供了构建智能代理所需的核心功能：
- 工具调用和执行
- 上下文管理和自动摘要
- 多模态内容支持（图像、视频、音频）
- 灵活的LLM客户端集成
- 代码执行环境（本地、Docker、E2B）

使用示例:
    from stirrup import Agent, DEFAULT_TOOLS
    from stirrup.clients.chat_completions_client import ChatCompletionsClient
    from stirrup.tools.mcp import MCPToolProvider

    # 为你的LLM提供商创建客户端
    # ChatCompletionsClient支持OpenAI兼容的API（OpenAI、OpenRouter、Deepseek等）
    client = ChatCompletionsClient(model="gpt-5")

    # 使用默认工具的简单用法
    # 默认工具包括：代码执行、网页获取、网页搜索
    agent = Agent(
        client=client,
        name="assistant",
        system_prompt="You are a helpful assistant.",
    )

    # 使用session上下文管理器执行任务
    # session会自动处理工具生命周期、日志记录和文件输出
    async with agent.session(output_dir="./output") as session:
        finish_params, history, metadata = await session.run("Your task here")
        print(finish_params.reason)

    # 使用MCP扩展默认工具
    # MCP（Model Context Protocol）允许集成外部工具服务器
    agent = Agent(
        client=client,
        name="assistant",
        tools=[*DEFAULT_TOOLS, MCPToolProvider.from_config("mcp.json")],
    )
"""

from stirrup import tools
from stirrup.core.agent import Agent
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import (
    Addable,
    AssistantMessage,
    AudioContentBlock,
    ChatMessage,
    ImageContentBlock,
    LLMClient,
    SubAgentMetadata,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolProvider,
    ToolResult,
    ToolUseCountMetadata,
    UserMessage,
    VideoContentBlock,
    aggregate_metadata,
)

__all__ = [
    "Addable",
    "Agent",
    "AssistantMessage",
    "AudioContentBlock",
    "ChatMessage",
    "ContextOverflowError",
    "ImageContentBlock",
    "LLMClient",
    "SubAgentMetadata",
    "SystemMessage",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "ToolProvider",
    "ToolResult",
    "ToolUseCountMetadata",
    "UserMessage",
    "VideoContentBlock",
    "aggregate_metadata",
    "tools",
]
