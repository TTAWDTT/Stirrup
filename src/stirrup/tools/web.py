"""网页工具：用于获取网页和搜索网络。

该模块提供web_fetch和web_search工具，通过WebToolProvider类管理共享的HTTP客户端生命周期。

主要功能：
- web_fetch: 获取网页并提取主要内容为Markdown格式
- web_search: 使用Brave Search API搜索网络

使用示例：
    from stirrup.clients.chat_completions_client import ChatCompletionsClient

    # 作为Agent中DEFAULT_TOOLS的一部分使用
    client = ChatCompletionsClient(model="gpt-5")
    agent = Agent(
        client=client,
        name="assistant",
        tools=DEFAULT_TOOLS,  # 包含WebToolProvider
    )

    # 独立使用
    async with WebToolProvider() as provider:
        tools = provider.get_tools()
"""

import os
from html import escape
from types import TracebackType
from typing import Annotated, Any

import httpx
import trafilatura
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from stirrup.core.models import Tool, ToolProvider, ToolResult
from stirrup.utils.text import truncate_msg

__all__ = ["WebToolProvider"]

# 常量配置
MAX_LENGTH_WEB_FETCH_HTML = 40000  # 网页获取结果的最大长度（字符数）
MAX_LENGTH_WEB_SEARCH_RESULTS = 40000  # 搜索结果的最大长度（字符数）
DEFAULT_WEBFETCH_HEADERS = {  # 默认HTTP请求头，模拟浏览器行为
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}
WEB_FETCH_TIMEOUT = 60 * 3  # 网页获取超时时间：3分钟（180秒）
WEB_SEARCH_TIMEOUT = 60 * 3  # 网页搜索超时时间：3分钟（180秒）


# =============================================================================
# 网页获取工具
# =============================================================================


class FetchWebPageParams(BaseModel):
    """网页获取工具的参数模型。"""

    url: Annotated[str, Field(description="要获取和提取的网页的完整HTTP或HTTPS URL地址")]


class WebFetchMetadata(BaseModel):
    """网页获取工具的元数据，跟踪已获取的URL。
    
    实现Addable协议，支持跨多次获取的聚合。
    可用于统计访问了哪些页面、访问次数等。
    """

    num_uses: int = 1  # 工具使用次数
    pages_fetched: list[str] = Field(default_factory=list)  # 已获取的页面URL列表

    def __add__(self, other: "WebFetchMetadata") -> "WebFetchMetadata":
        """组合多次网页获取的元数据。"""
        return WebFetchMetadata(
            num_uses=self.num_uses + other.num_uses,
            pages_fetched=self.pages_fetched + other.pages_fetched,
        )


def _get_fetch_web_page_tool(client: httpx.AsyncClient | None = None) -> Tool[FetchWebPageParams, WebFetchMetadata]:
    """创建网页获取工具，将主要内容提取为Markdown格式。
    
    该工具使用trafilatura库智能提取网页的主要内容，过滤掉导航、广告等无关信息。
    提取的内容以Markdown格式返回，便于LLM理解和处理。
    
    Args:
        client: 可选的共享httpx.AsyncClient用于连接池化，提高性能
    
    Returns:
        配置好的Tool对象，可获取网页并提取干净的Markdown内容
    """

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _fetch(url: str, http_client: httpx.AsyncClient) -> httpx.Response:
        """执行HTTP GET请求，网络错误时自动重试。
        
        使用指数退避策略重试，最多3次尝试。
        这提高了网络不稳定时的可靠性。
        """
        response = await http_client.get(url, headers=DEFAULT_WEBFETCH_HEADERS)
        response.raise_for_status()
        return response

    async def fetch_web_page_executor(params: FetchWebPageParams) -> ToolResult[WebFetchMetadata]:
        """获取网页并使用trafilatura提取主要内容为Markdown。
        
        工作流程：
        1. 发送HTTP GET请求获取网页HTML
        2. 使用trafilatura智能提取主要内容
        3. 转换为Markdown格式
        4. 截断过长内容以适应token限制
        5. 返回XML格式的结果，包含URL和内容/错误
        """
        try:
            # 使用提供的客户端或创建临时客户端（向后兼容）
            if client is not None:
                response = await _fetch(params.url, client)
            else:
                async with httpx.AsyncClient(
                    headers=DEFAULT_WEBFETCH_HEADERS,
                    follow_redirects=True,
                    timeout=WEB_FETCH_TIMEOUT,
                ) as temp_client:
                    response = await _fetch(params.url, temp_client)

            # 使用trafilatura提取主要内容并转换为Markdown
            body_md = trafilatura.extract(response.text, output_format="markdown") or ""
            return ToolResult(
                content=f"<web_fetch><url>{params.url}</url><body>"
                f"{truncate_msg(body_md, MAX_LENGTH_WEB_FETCH_HTML)}</body></web_fetch>",
                metadata=WebFetchMetadata(pages_fetched=[params.url]),
            )
        except httpx.HTTPError as exc:
            # HTTP错误（404、500等）也返回结构化响应
            return ToolResult(
                content=f"<web_fetch><url>{params.url}</url><error>"
                f"{truncate_msg(str(exc), MAX_LENGTH_WEB_FETCH_HTML)}</error></web_fetch>",
                metadata=WebFetchMetadata(pages_fetched=[params.url]),
            )

    return Tool[FetchWebPageParams, WebFetchMetadata](
        name="fetch_web_page",
        description="从网页获取并提取主要内容为Markdown格式。返回正文文本或错误信息，格式为XML。",
        parameters=FetchWebPageParams,
        executor=fetch_web_page_executor,  # ty: ignore[invalid-argument-type]
    )


# =============================================================================
# 网页搜索工具
# =============================================================================


class WebSearchParams(BaseModel):
    """网页搜索工具的参数模型。"""

    query: Annotated[
        str, Field(description="Brave Search的自然语言搜索查询（语法类似Google搜索）")
    ]


class WebSearchMetadata(BaseModel):
    """网页搜索工具的元数据，跟踪搜索结果。
    
    实现Addable协议，支持跨多次搜索的聚合。
    用于统计搜索次数和返回的结果总数。
    """

    num_uses: int = 1  # 搜索次数
    pages_returned: int = 0  # 返回的页面数量

    def __add__(self, other: "WebSearchMetadata") -> "WebSearchMetadata":
        """组合多次搜索的元数据。"""
        return WebSearchMetadata(
            num_uses=self.num_uses + other.num_uses,
            pages_returned=self.pages_returned + other.pages_returned,
        )


def _get_websearch_tool(
    brave_api_key: str | None, client: httpx.AsyncClient | None = None
) -> Tool[WebSearchParams, WebSearchMetadata]:
    """创建使用Brave Search API的网页搜索工具。
    
    Brave Search提供快速、隐私友好的网页搜索服务。
    需要API密钥才能使用（可从Brave Search API网站获取）。
    
    Args:
        brave_api_key: Brave Search API密钥，或None则使用BRAVE_API_KEY环境变量
        client: 可选的共享httpx.AsyncClient用于连接池化
    
    Returns:
        配置好的Tool对象，可搜索网络并返回前5个结果为XML格式
    
    Raises:
        RuntimeError: 如果未提供API密钥且环境变量中也未找到
    """
    if brave_api_key is None:
        brave_api_key = os.getenv("BRAVE_API_KEY")

    if brave_api_key is None:
        raise RuntimeError("未提供Brave Search API密钥。")

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=4, min=1, max=3),
        reraise=True,
    )
    async def _search(query: str, http_client: httpx.AsyncClient) -> dict:
        """执行Brave Search API请求，网络错误时自动重试。
        
        使用指数退避策略，最多3次尝试。
        """
        response = await http_client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": brave_api_key,
                "Accept": "application/json",
            },
            params={"q": query, "count": 5},  # 返回前5个搜索结果
        )
        response.raise_for_status()
        return response.json()

    async def websearch_executor(params: WebSearchParams) -> ToolResult[WebSearchMetadata]:
        """执行网页搜索并将结果格式化为XML，包含标题、URL和描述。
        
        返回格式：
        <results>
          <result>
            <title>页面标题</title>
            <url>页面URL</url>
            <description>页面描述</description>
          </result>
          ...
        </results>
        """
        # 使用提供的客户端或创建临时客户端
        if client is not None:
            data = await _search(params.query, client)
        else:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as temp_client:
                data = await _search(params.query, temp_client)

        results = data.get("web", {}).get("results", [])
        # 构建XML格式的搜索结果，转义特殊字符防止XML注入
        results_xml = (
            "<results>\n"
            + "\n".join(
                (
                    "<result>"
                    f"\n<title>{escape(result.get('title', '') or '')}</title>"
                    f"\n<url>{escape(result.get('url', '') or '')}</url>"
                    f"\n<description>{escape(result.get('description', '') or '')}</description>"
                    "\n</result>"
                )
                for result in results
            )
            + "\n</results>"
        )

        return ToolResult(
            content=truncate_msg(results_xml, MAX_LENGTH_WEB_SEARCH_RESULTS),
            metadata=WebSearchMetadata(pages_returned=len(results)),
        )

    return Tool[WebSearchParams, WebSearchMetadata](
        name="web_search",
        description="使用Brave Search API搜索网络。返回前5个结果，包含标题、URL和描述，格式为XML。",
        parameters=WebSearchParams,
        executor=websearch_executor,  # ty: ignore[invalid-argument-type]
    )


# =============================================================================
# WebToolProvider - 网页工具提供者
# =============================================================================


class WebToolProvider(ToolProvider):
    """提供网页工具（web_fetch、web_search），带有托管的HTTP客户端生命周期。
    
    WebToolProvider实现Tool生命周期协议（has_lifecycle=True），
    因此可以直接在Agent的tools列表中使用。它在__aenter__时创建httpx.AsyncClient
    并返回网页工具。
    
    特性：
    - 自动HTTP客户端管理（连接池、超时控制）
    - 可选的Brave Search集成（需要API密钥）
    - 自动重试机制处理网络错误
    - 智能内容提取（使用trafilatura）
    
    在Agent中作为Tool使用（推荐）：
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(
            client=client,
            name="assistant",
            tools=[LocalCodeExecToolProvider(), WebToolProvider(), CALCULATOR_TOOL],
        )

        async with agent.session(output_dir="./output") as session:
            await session.run("搜索网络并获取页面")

    独立使用：
        async with WebToolProvider() as provider:
            tools = provider.get_tools()
    """

    def __init__(
        self,
        *,
        timeout: float = 60 * 3,
        brave_api_key: str | None = None,
    ) -> None:
        """初始化WebToolProvider。
        
        Args:
            timeout: HTTP超时时间（秒）。默认180秒（3分钟）。
                    适用于慢速网站或大文件下载。
            brave_api_key: Brave Search API密钥，用于web_search工具。
                          如果为None，使用BRAVE_API_KEY环境变量。
                          如果未提供API密钥，网页搜索将不可用（仅web_fetch可用）。
        """
        self._timeout = timeout
        self._brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> list[Tool[Any, Any]]:
        """进入异步上下文：创建HTTP客户端并返回网页工具。
        
        创建配置好的httpx.AsyncClient，启用：
        - 指定的超时时间
        - 自动跟随重定向
        - 连接池化（提高性能）
        
        Returns:
            Tool对象列表（web_fetch，以及web_search如果API密钥可用）
        """
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
        )
        await self._client.__aenter__()
        return self.get_tools()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """退出异步上下文：关闭HTTP客户端。
        
        确保所有连接正确关闭，释放系统资源。
        """
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    def get_tools(self) -> list[Tool[Any, Any]]:
        """获取配置了托管HTTP客户端的网页工具。
        
        Returns:
            包含web_fetch工具的列表，以及web_search工具（如果API密钥可用）
        
        Raises:
            RuntimeError: 如果在进入上下文之前调用
        """
        if self._client is None:
            raise RuntimeError("WebToolProvider未启动。请先使用'async with'。")

        tools: list[Tool[Any, Any]] = [_get_fetch_web_page_tool(self._client)]

        # 仅在API密钥可用时添加web_search
        if self._brave_api_key:
            tools.append(_get_websearch_tool(self._brave_api_key, self._client))

        return tools
