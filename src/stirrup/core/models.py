import mimetypes
import warnings
from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import Awaitable, Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from io import BytesIO
from math import isinf, isnan, sqrt
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import Annotated, Any, ClassVar, Literal, Protocol, Self, overload, runtime_checkable

import filetype
from moviepy import AudioFileClip, VideoFileClip
from moviepy.video.fx import Resize
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from stirrup.constants import RESOLUTION_1MP, RESOLUTION_480P

__all__ = [
    "Addable",
    "AssistantMessage",
    "AudioContentBlock",
    "BinaryContentBlock",
    "ChatMessage",
    "Content",
    "ContentBlock",
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
]


def downscale_image(w: int, h: int, max_pixels: int | None = 1_000_000) -> tuple[int, int]:
    """缩小图像尺寸以适应最大像素数限制，同时保持宽高比。
    
    该函数根据指定的最大像素数自动计算缩放后的图像尺寸。它会：
    1. 保持原始图像的宽高比不变
    2. 确保返回的尺寸为偶数（某些视频编码器需要）
    3. 保证最小尺寸为2x2像素
    
    Args:
        w: 原始图像宽度（像素）
        h: 原始图像高度（像素）
        max_pixels: 允许的最大像素总数，None表示不限制
        
    Returns:
        缩放后的(宽度, 高度)元组，值均为偶数且不小于2
    """
    s = 1.0 if max_pixels is None or w * h <= max_pixels else sqrt(max_pixels / (w * h))
    nw, nh = int(w * s) // 2 * 2, int(h * s) // 2 * 2
    return max(nw, 2), max(nh, 2)


# 内容块相关类型定义
class BinaryContentBlock(BaseModel, ABC):
    """二进制内容块基类：用于处理图像、视频、音频等二进制内容，包含MIME类型验证。
    
    这是所有二进制内容类型的抽象基类，提供了：
    1. 自动MIME类型检测（基于文件头部特征）
    2. 文件扩展名推断
    3. 内容有效性验证
    
    子类必须实现_probe()方法来执行格式特定的验证检查。
    """

    data: bytes  # 二进制数据内容
    allowed_mime_types: ClassVar[set[str]]  # 该内容类型允许的MIME类型集合

    @property
    def mime_type(self) -> str:
        """基于数据头部检测MIME类型。
        
        使用filetype库分析二进制数据的魔术字节（magic bytes）来确定文件类型。
        这比依赖文件扩展名更可靠，因为它检查实际的文件内容。
        
        Returns:
            检测到的MIME类型字符串（例如："image/png", "video/mp4"）
            
        Raises:
            ValueError: 如果无法识别文件类型
        """
        match: filetype.Type = filetype.guess(self.data)
        if match is None:
            raise ValueError(f"Unsupported file type {self.data!r}")
        return match.mime

    @property
    def extension(self) -> str:
        """获取内容的文件扩展名（例如：'png', 'mp4', 'mp3'），不包含前导点。
        
        基于检测到的MIME类型推断文件扩展名。这对于保存文件或生成文件名很有用。
        
        Returns:
            文件扩展名字符串，不包含'.'前缀
            
        Raises:
            ValueError: 如果MIME类型无法映射到已知的文件扩展名
        """
        _extension = mimetypes.guess_extension(self.mime_type)
        if _extension is None:
            raise ValueError(f"Unsupported mime_type {self.mime_type!r}")
        return _extension[1:]

    @model_validator(mode="after")
    def _check_mime(self) -> Self:
        """验证MIME类型是否在允许列表中，并检查内容是否可读。
        
        这是一个Pydantic验证器，在对象创建后自动执行：
        1. 检查检测到的MIME类型是否在allowed_mime_types集合中
        2. 调用_probe()方法验证内容格式是否正确
        
        Returns:
            验证通过的对象实例
            
        Raises:
            ValueError: 如果MIME类型不被支持或内容无法解析
        """
        if self.allowed_mime_types and self.mime_type not in self.allowed_mime_types:
            raise ValueError("Unsupported mime_type {self.mime_type!r}; allowed: {allowed}")
        self._probe()  # 轻量级损坏检查；不执行重度处理
        return self

    @abstractmethod
    def _probe(self) -> None:
        """验证内容是否可以打开和读取；子类实现特定格式的检查。
        
        这是一个抽象方法，每个子类必须实现其特定的验证逻辑。
        例如：
        - ImageContentBlock使用PIL验证图像
        - VideoContentBlock使用moviepy验证视频
        - AudioContentBlock使用moviepy验证音频
        
        Raises:
            应在内容无效时抛出异常
        """


class ImageContentBlock(BinaryContentBlock):
    """图像内容块：支持PNG、JPEG、WebP、PSD等格式，带有自动降采样功能。
    
    该类处理各种图像格式，并提供以下功能：
    1. 自动格式检测和验证
    2. 图像降采样以控制像素数（节省API成本和带宽）
    3. 转换为base64编码的data URL用于LLM输入
    
    支持的格式包括：JPEG、PNG、GIF、BMP、TIFF、PSD
    """

    kind: Literal["image_content_block"] = "image_content_block"
    allowed_mime_types: ClassVar[set[str]] = {
        "image/jpeg",  # JPEG格式 - 有损压缩，适合照片
        "image/png",  # PNG格式 - 无损压缩，支持透明度
        "image/gif",  # GIF格式 - 支持动画和透明度
        "image/bmp",  # BMP格式 - Windows位图格式
        "image/tiff",  # TIFF格式 - 高质量图像格式
        "image/vnd.adobe.photoshop",  # PSD格式 - Adobe Photoshop文件
    }

    def _probe(self) -> None:
        """使用PIL尝试打开和验证图像数据的有效性。
        
        这个方法会：
        1. 将二进制数据加载到PIL Image对象中
        2. 调用verify()方法检查图像完整性
        3. 如果图像损坏或格式无效，会抛出异常
        """
        with Image.open(BytesIO(self.data)) as im:
            im.verify()

    def to_base64_url(self, max_pixels: int | None = RESOLUTION_1MP) -> str:
        """将图像转换为base64数据URL，可选择调整大小到最大像素数。
        
        该方法执行以下操作：
        1. 如果图像超过max_pixels限制，则降采样到指定大小
        2. 转换为RGB模式（如果不是）以确保兼容性
        3. 保存为PNG格式（无损且广泛支持）
        4. 编码为base64并生成data URL格式
        
        Args:
            max_pixels: 最大像素数限制，None表示不限制大小
            
        Returns:
            格式为 "data:image/png;base64,..." 的base64编码图像URL
        """
        img: Image.Image = Image.open(BytesIO(self.data))
        if max_pixels is not None and img.width * img.height > max_pixels:
            tw, th = downscale_image(img.width, img.height, max_pixels)
            img.thumbnail((tw, th), Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{b64encode(buf.getvalue()).decode()}"


class VideoContentBlock(BinaryContentBlock):
    """视频内容块：支持MP4等格式，带有自动转码和分辨率降采样功能。
    
    该类处理各种视频格式，提供以下核心功能：
    1. 多种视频格式的自动检测和验证
    2. 视频转码为标准MP4格式（H.264编码）
    3. 分辨率降采样以控制文件大小和处理成本
    4. 帧率调整和音频处理
    
    支持的输入格式：AVI、MP4、MOV、MKV、WMV、FLV、MPEG、WebM、GIF（动画）
    输出格式：标准MP4（H.264视频 + AAC音频）
    """

    kind: Literal["video_content_block"] = "video_content_block"
    allowed_mime_types: ClassVar[set[str]] = {
        "video/x-msvideo",  # AVI格式 - 微软视频格式
        "video/mp4",  # MP4格式 - 最广泛支持的现代视频格式
        "video/quicktime",  # MOV格式 - Apple QuickTime格式
        "video/x-matroska",  # MKV格式 - 开源容器格式
        "video/x-ms-wmv",  # WMV格式 - Windows Media Video
        "video/x-flv",  # FLV格式 - Flash Video
        "video/mpeg",  # MPEG格式 - 传统MPEG视频
        "video/webm",  # WebM格式 - Web优化格式
        "video/gif",  # GIF格式 - 动画GIF
    }

    def _probe(self) -> None:
        """通过尝试将其打开为VideoFileClip来验证视频数据的有效性。
        
        该方法会：
        1. 将二进制数据写入临时文件（moviepy需要文件路径）
        2. 使用moviepy加载视频以验证格式
        3. 检查视频是否可以成功打开和解析
        4. 抑制moviepy的警告信息以保持日志清洁
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="moviepy.*")
        with NamedTemporaryFile(suffix=".bin") as f:
            f.write(self.data)
            f.flush()
            clip = VideoFileClip(f.name)
            clip.close()

    def to_base64_url(self, max_pixels: int | None = RESOLUTION_480P, fps: int | None = None) -> str:
        """将视频转码为MP4并返回base64数据URL。
        
        执行完整的视频处理流程：
        1. 如果分辨率超过max_pixels，降采样视频尺寸
        2. 应用H.264视频编码（广泛兼容）
        3. 如果原视频有音频，使用AAC编码处理音频
        4. 可选择调整帧率以进一步减小文件大小
        5. 编码为base64格式的data URL
        
        Args:
            max_pixels: 最大像素数限制（默认480p），用于控制视频分辨率
            fps: 可选的目标帧率，None表示保持原始帧率
            
        Returns:
            格式为 "data:video/mp4;base64,..." 的base64编码视频URL
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="moviepy.*")
            with NamedTemporaryFile(suffix=".mp4") as fin, NamedTemporaryFile(suffix=".mp4") as fout:
                fin.write(self.data)
                fin.flush()
                clip = VideoFileClip(fin.name)
                tw, th = downscale_image(int(clip.w), int(clip.h), max_pixels)
                clip = clip.with_effects([Resize(new_size=(tw, th))])

                clip.write_videofile(
                    fout.name,
                    codec="libx264",
                    fps=fps,
                    audio=clip.audio is not None,
                    audio_codec="aac",
                    preset="veryfast",
                    logger=None,
                )
                clip.close()
                return f"data:video/mp4;base64,{b64encode(fout.read()).decode()}"


class AudioContentBlock(BinaryContentBlock):
    """音频内容块：支持MPEG、WAV、AAC等常见音频格式。
    
    该类处理各种音频格式，提供以下功能：
    1. 多种音频格式的自动检测和验证
    2. 音频转码为标准MP3格式
    3. 比特率控制以平衡质量和文件大小
    4. 转换为base64编码用于LLM输入
    
    支持的格式：AAC、FLAC、MP3、M4A、MPEG、OGG、PCM、WAV、WebM等
    输出格式：MP3（libmp3lame编码）
    """

    kind: Literal["audio_content_block"] = "audio_content_block"
    allowed_mime_types: ClassVar[set[str]] = {
        "audio/x-aac",  # AAC格式 - 高级音频编码
        "audio/flac",  # FLAC格式 - 无损音频压缩
        "audio/mp3",  # MP3格式 - 最常用的音频格式
        "audio/m4a",  # M4A格式 - Apple音频格式
        "audio/mpeg",  # MPEG音频
        "audio/mpga",  # MPEG音频（另一种MIME类型）
        "audio/mp4",  # MP4音频
        "audio/ogg",  # OGG格式 - 开源音频格式
        "audio/pcm",  # PCM格式 - 未压缩音频
        "audio/wav",  # WAV格式 - Windows波形音频
        "audio/webm",  # WebM音频
        "audio/x-wav",  # WAV格式（另一种MIME类型）
        "audio/aac",  # AAC格式（另一种MIME类型）
    }

    def _probe(self) -> None:
        """通过尝试将其打开为AudioFileClip来验证音频数据的有效性。
        
        验证过程：
        1. 将音频数据写入临时文件
        2. 使用moviepy的AudioFileClip加载音频
        3. 验证音频是否可以成功解析
        4. 抑制警告信息以保持输出清洁
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="moviepy.*")
        with NamedTemporaryFile(suffix=".bin") as fin:
            fin.write(self.data)
            fin.flush()
            clip = AudioFileClip(fin.name)
            clip.close()

    def to_base64_url(self, bitrate: str = "192k") -> str:
        """将音频转码为MP3并返回base64数据URL。
        
        音频处理流程：
        1. 使用libmp3lame编码器转码为MP3格式
        2. 应用指定的比特率（默认192kbps，良好的质量/大小平衡）
        3. 编码为base64格式的data URL
        
        Args:
            bitrate: 目标音频比特率（如"128k"、"192k"、"320k"）
                    更高的比特率 = 更好的质量但文件更大
            
        Returns:
            格式为 "data:audio/mpeg;base64,..." 的base64编码音频URL
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="moviepy.*")
            with NamedTemporaryFile(suffix=".bin") as fin, NamedTemporaryFile(suffix=".mp3") as fout:
                fin.write(self.data)
                fin.flush()
                clip = AudioFileClip(fin.name)
                clip.write_audiofile(fout.name, codec="libmp3lame", bitrate=bitrate, logger=None)
                clip.close()
                return f"data:audio/mpeg;base64,{b64encode(fout.read()).decode()}"


type ContentBlock = ImageContentBlock | VideoContentBlock | AudioContentBlock | str
"""内容块类型联合：图像、视频、音频或纯文本的联合类型。

这个类型定义允许消息内容包含多种媒体类型，实现真正的多模态对话。
"""

type Content = list[ContentBlock] | str
"""消息内容类型：可以是纯字符串，也可以是混合内容块的列表。

示例：
- 纯文本："Hello, world!"
- 混合内容：["这是一张图片:", ImageContentBlock(...), "这是视频:", VideoContentBlock(...)]
"""


# 元数据协议和聚合
@runtime_checkable
class Addable(Protocol):
    """可加性协议：定义支持通过__add__进行聚合的类型。
    
    实现此协议的类型可以使用+运算符进行组合，这对于聚合多次工具调用的元数据非常有用。
    例如：TokenUsage、ToolUseCountMetadata等都实现了此协议。
    """

    def __add__(self, other: Self) -> Self: ...


def _aggregate_list[T: Addable](metadata_list: list[T]) -> T | None:
    """使用__add__方法聚合元数据列表。
    
    该函数将列表中的所有元数据对象组合成单个聚合对象。
    这对于统计工具使用情况、token消耗等非常有用。
    
    Args:
        metadata_list: 要聚合的元数据对象列表
        
    Returns:
        聚合后的元数据对象，如果列表为空则返回None
    """
    if not metadata_list:
        return None
    aggregated = metadata_list[0]
    for m in metadata_list[1:]:
        aggregated = aggregated + m
    return aggregated


def to_json_serializable(value: object) -> object:
    # None and JSON primitives
    if value is None or isinstance(value, str | int | bool):
        return value

    # Floats need special handling for nan/inf
    if isinstance(value, float):
        if isnan(value) or isinf(value):
            raise ValueError(f"Cannot serialize {value} to JSON")
        return value

    # Pydantic models
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    # Common non-serializable types
    if isinstance(value, datetime | date | time):
        return value.isoformat()

    if isinstance(value, timedelta):
        return value.total_seconds()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, dict):
        return {k: to_json_serializable(v) for k, v in value.items()}

    if isinstance(value, list | tuple | set | frozenset):
        return [to_json_serializable(v) for v in value]

    # We have not implemented other cases (e.g. Bytes, Enum, etc.)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON: {value!r}")


@overload
def aggregate_metadata(
    metadata_dict: dict[str, list[Any]], prefix: str = "", return_json_serializable: Literal[True] = True
) -> object: ...


@overload
def aggregate_metadata(
    metadata_dict: dict[str, list[Any]], prefix: str = "", return_json_serializable: Literal[False] = ...
) -> dict: ...


def _collect_all_token_usage(result: dict) -> "TokenUsage":
    """Recursively collect all token_usage from a flattened aggregate_metadata result.

    Args:
        result: The flattened dict from aggregate_metadata (before JSON serialization)

    Returns:
        Combined TokenUsage from all entries (direct and nested sub-agents)
    """
    total = TokenUsage()

    for key, value in result.items():
        if key == "token_usage" and isinstance(value, TokenUsage):
            # Direct token_usage at this level
            total = total + value
        elif isinstance(value, dict):
            # This could be a sub-agent's tool dict - check for nested token_usage
            nested_token_usage = value.get("token_usage")
            if isinstance(nested_token_usage, TokenUsage):
                total = total + nested_token_usage

    return total


def aggregate_metadata(
    metadata_dict: dict[str, list[Any]], prefix: str = "", return_json_serializable: bool = True
) -> dict | object:
    """Aggregate metadata lists and flatten sub-agents into a single-level dict with hierarchical keys.

    For entries with nested run_metadata (e.g., SubAgentMetadata), flattens sub-agents using dot notation.
    Each sub-agent's value is a dict mapping its direct tool names to their aggregated metadata
    (excluding nested sub-agent data, which gets its own top-level key).

    At the root level, token_usage is rolled up to include all sub-agent token usage.

    Args:
        metadata_dict: Dict mapping names (tools or agents) to lists of metadata instances
        prefix: Key prefix for nested calls (used internally for recursion)

    Returns:
        Flat dict with dot-notation keys for sub-agents.
        Example: {
            "token_usage": <combined from all agents>,
            "web_browsing_sub_agent": {"web_search": <aggregated>, "token_usage": <aggregated>},
            "web_browsing_sub_agent.web_fetch_sub_agent": {"fetch_web_page": <aggregated>, "token_usage": <aggregated>}
        }
    """
    result: dict = {}

    # First pass: aggregate all entries in this level
    aggregated_level: dict = {}
    for name, metadata_list in metadata_dict.items():
        if not metadata_list:
            continue
        aggregated_level[name] = _aggregate_list(metadata_list)

    # Second pass: separate nested sub-agents from direct tools, and recurse
    direct_tools: dict = {}
    for name, aggregated in aggregated_level.items():
        if hasattr(aggregated, "run_metadata") and isinstance(aggregated.run_metadata, dict):
            # This is a sub-agent - recurse into it
            full_key = f"{prefix}.{name}" if prefix else name
            nested = aggregate_metadata(aggregated.run_metadata, prefix=full_key, return_json_serializable=False)
            result.update(nested)
        else:
            # This is a direct tool/metadata - keep it at this level
            direct_tools[name] = aggregated

    # Store direct tools under the current prefix
    if prefix:
        result[prefix] = direct_tools
    else:
        # At root level, merge direct tools into result
        result.update(direct_tools)

    # At root level, roll up all token_usage from sub-agents
    if not prefix:
        total_token_usage = _collect_all_token_usage(result)
        if total_token_usage.total > 0:
            result["token_usage"] = [total_token_usage]

    if return_json_serializable:
        # Convert all Pydantic models to JSON-serializable dicts
        return to_json_serializable(result)
    return result


# Messages
class TokenUsage(BaseModel):
    """Token counts for LLM usage (input, output, reasoning tokens)."""

    input: int = 0
    output: int = 0
    reasoning: int = 0

    @property
    def total(self) -> int:
        """Total token count across input, output, and reasoning."""
        return self.input + self.output + self.reasoning

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects together, summing each field independently."""
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            reasoning=self.reasoning + other.reasoning,
        )


class ToolUseCountMetadata(BaseModel):
    """Generic metadata tracking tool usage count.

    Implements Addable protocol for aggregation. Use this for tools that only need
    to track how many times they were called.
    """

    num_uses: int = 1

    def __add__(self, other: "ToolUseCountMetadata") -> "ToolUseCountMetadata":
        return ToolUseCountMetadata(num_uses=self.num_uses + other.num_uses)


class ToolResult[M](BaseModel):
    """Result from a tool executor with optional metadata.

    Generic over metadata type M. M should implement Addable protocol for aggregation support,
    but this is not enforced at the class level due to Pydantic schema generation limitations.
    """

    content: Content
    metadata: M | None = None


class Tool[P: BaseModel, M](BaseModel):
    """Tool definition with name, description, parameter schema, and executor function.

    Generic over:
        P: Parameter model type (must be a Pydantic BaseModel, or None for parameterless tools)
        M: Metadata type (should implement Addable for aggregation; use None for tools without metadata)

    Tools are simple, stateless callables. For tools requiring lifecycle management
    (setup/teardown, resource pooling), use a ToolProvider instead.

    Example with parameters:
        ```python
        class CalcParams(BaseModel):
            expression: str

        calc_tool = Tool[CalcParams, None](
            name="calc",
            description="Evaluate math",
            parameters=CalcParams,
            executor=lambda p: ToolResult(content=str(eval(p.expression))),
        )
        ```

    Example without parameters:
        ```python
        time_tool = Tool[None, None](
            name="time",
            description="Get current time",
            executor=lambda _: ToolResult(content=datetime.now().isoformat()),
        )
        ```
    """

    name: str
    description: str
    parameters: type[P] | None = None
    executor: Callable[[P], ToolResult[M] | Awaitable[ToolResult[M]]]


class ToolProvider(ABC):
    """Abstract base class for tool providers with lifecycle management.

    ToolProviders manage resources (HTTP clients, sandboxes, server connections)
    and return Tool instances when entering their async context. They implement
    the async context manager protocol.

    Use ToolProvider for:
    - Tools requiring setup/teardown (connections, temp directories)
    - Tools that return multiple Tool instances (e.g., MCP servers)
    - Tools with shared state across calls (e.g., HTTP client pooling)

    Example:
        class MyToolProvider(ToolProvider):
            async def __aenter__(self) -> Tool | list[Tool]:
                # Setup resources and return tool(s)
                return self._create_tool()

            # __aexit__ is optional - default is no-op

    Agent automatically manages ToolProvider lifecycle via its session() context.
    """

    @abstractmethod
    async def __aenter__(self) -> "Tool | list[Tool]":
        """Enter async context: setup resources and return tool(s).

        Returns:
            A single Tool instance, or a list of Tool instances for providers
            that expose multiple tools (e.g., MCP servers).
        """
        ...

    async def __aexit__(  # noqa: B027
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context: cleanup resources. Default: no-op."""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol defining the interface for LLM client implementations.

    Any LLM client must implement this protocol to work with the Agent class.
    Provides text generation with tool support and model capability inspection.
    """

    @abstractmethod
    async def generate(self, messages: list["ChatMessage"], tools: dict[str, Tool]) -> "AssistantMessage": ...

    @property
    def model_slug(self) -> str: ...

    @property
    def max_tokens(self) -> int: ...


class ToolCall(BaseModel):
    """Represents a tool invocation request from the LLM.

    Attributes:
        name: Name of the tool to invoke
        arguments: JSON string containing tool parameters
        tool_call_id: Unique identifier for tracking this tool call and its result
    """

    name: str
    arguments: str
    tool_call_id: str | None = None


class SystemMessage(BaseModel):
    """System-level instructions and context for the LLM."""

    role: Literal["system"] = "system"
    content: Content


class UserMessage(BaseModel):
    """User input message to the LLM."""

    role: Literal["user"] = "user"
    content: Content


class Reasoning(BaseModel):
    """Extended thinking/reasoning content from models that support chain-of-thought reasoning."""

    signature: str | None = None
    content: str


class AssistantMessage(BaseModel):
    """LLM response message with optional tool calls and token usage tracking."""

    role: Literal["assistant"] = "assistant"
    reasoning: Reasoning | None = None
    content: Content
    tool_calls: Annotated[list[ToolCall], Field(default_factory=list)]
    token_usage: Annotated[TokenUsage, Field(default_factory=TokenUsage)]


class ToolMessage(BaseModel):
    """Tool execution result returned to the LLM."""

    role: Literal["tool"] = "tool"
    content: Content
    tool_call_id: str | None = None
    name: str | None = None
    args_was_valid: bool = True


type ChatMessage = Annotated[SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")]
"""Discriminated union of all message types, automatically parsed based on role field."""


class SubAgentMetadata(BaseModel):
    """Metadata from sub-agent execution including token usage, message history, and child run metadata.

    Implements Addable protocol to support aggregation across multiple subagent calls.
    """

    message_history: list[list[ChatMessage]]
    run_metadata: Annotated[dict[str, list[Any]], Field(default_factory=dict)]

    def __add__(self, other: "SubAgentMetadata") -> "SubAgentMetadata":
        """Combine metadata from multiple subagent calls."""
        # Concatenate message histories
        combined_history = self.message_history + other.message_history
        # Merge run metadata (concatenate lists per key)
        combined_meta: dict[str, list[Any]] = dict(self.run_metadata)
        for key, metadata_list in other.run_metadata.items():
            if key in combined_meta:
                combined_meta[key] = combined_meta[key] + metadata_list
            else:
                combined_meta[key] = list(metadata_list)
        return SubAgentMetadata(
            message_history=combined_history,
            run_metadata=combined_meta,
        )
