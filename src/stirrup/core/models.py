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
    """将Python对象转换为JSON可序列化的格式。
    
    该函数处理各种Python类型，将它们转换为可以安全序列化为JSON的格式。
    这对于保存元数据、日志记录和API响应非常重要。
    
    处理的类型包括：
    - 基本类型：None、str、int、bool
    - 浮点数：处理NaN和Infinity的特殊情况
    - Pydantic模型：使用model_dump转换
    - 日期/时间：转换为ISO格式字符串
    - 集合类型：dict、list、tuple、set等
    
    Args:
        value: 要序列化的Python对象
        
    Returns:
        JSON可序列化的对象（dict、list、str、int、float、bool或None）
        
    Raises:
        ValueError: 如果遇到NaN或Infinity浮点值
        TypeError: 如果遇到不支持的类型
    """
    # None和JSON基本类型可以直接使用
    if value is None or isinstance(value, str | int | bool):
        return value

    # 浮点数需要特殊处理NaN和无穷大
    if isinstance(value, float):
        if isnan(value) or isinf(value):
            raise ValueError(f"Cannot serialize {value} to JSON")
        return value

    # Pydantic模型转换为字典
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    # 常见的不可序列化类型转换
    if isinstance(value, datetime | date | time):
        return value.isoformat()  # 转换为ISO 8601格式字符串

    if isinstance(value, timedelta):
        return value.total_seconds()  # 转换为秒数

    if isinstance(value, Decimal):
        return float(value)  # 高精度数字转换为浮点数

    # 递归处理字典
    if isinstance(value, dict):
        return {k: to_json_serializable(v) for k, v in value.items()}

    # 递归处理列表和其他序列类型
    if isinstance(value, list | tuple | set | frozenset):
        return [to_json_serializable(v) for v in value]

    # 未实现其他情况（如Bytes、Enum等）
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
    """递归收集扁平化aggregate_metadata结果中的所有token使用量。
    
    该函数遍历聚合元数据字典，查找并累加所有的token使用量，
    包括直接的token_usage和嵌套子代理的token_usage。
    
    Args:
        result: aggregate_metadata返回的扁平化字典（JSON序列化之前）
    
    Returns:
        组合所有条目（直接和嵌套子代理）的TokenUsage总和
    """
    total = TokenUsage()

    for key, value in result.items():
        if key == "token_usage" and isinstance(value, TokenUsage):
            # 当前层级的直接token使用量
            total = total + value
        elif isinstance(value, dict):
            # 这可能是子代理的工具字典 - 检查嵌套的token_usage
            nested_token_usage = value.get("token_usage")
            if isinstance(nested_token_usage, TokenUsage):
                total = total + nested_token_usage

    return total


def aggregate_metadata(
    metadata_dict: dict[str, list[Any]], prefix: str = "", return_json_serializable: bool = True
) -> dict | object:
    """聚合元数据列表并将子代理扁平化为带有层次键的单级字典。
    
    对于具有嵌套run_metadata的条目（如SubAgentMetadata），使用点表示法扁平化子代理。
    每个子代理的值是一个字典，将其直接工具名称映射到聚合的元数据
    （不包括嵌套的子代理数据，它们有自己的顶级键）。
    
    在根级别，token_usage被汇总以包括所有子代理的token使用量。
    
    Args:
        metadata_dict: 将名称（工具或代理）映射到元数据实例列表的字典
        prefix: 嵌套调用的键前缀（内部递归使用）
        return_json_serializable: 是否将结果转换为JSON可序列化格式
    
    Returns:
        扁平化字典，使用点表示法表示子代理。
        示例: {
            "token_usage": <所有代理的组合>,
            "web_browsing_sub_agent": {"web_search": <聚合>, "token_usage": <聚合>},
            "web_browsing_sub_agent.web_fetch_sub_agent": {"fetch_web_page": <聚合>, "token_usage": <聚合>}
        }
    """

    result: dict = {}

    # 第一遍：聚合当前层级的所有条目
    aggregated_level: dict = {}
    for name, metadata_list in metadata_dict.items():
        if not metadata_list:
            continue
        aggregated_level[name] = _aggregate_list(metadata_list)

    # 第二遍：将嵌套的子代理与直接工具分开，并递归处理
    direct_tools: dict = {}
    for name, aggregated in aggregated_level.items():
        if hasattr(aggregated, "run_metadata") and isinstance(aggregated.run_metadata, dict):
            # 这是一个子代理 - 递归进入它
            full_key = f"{prefix}.{name}" if prefix else name
            nested = aggregate_metadata(aggregated.run_metadata, prefix=full_key, return_json_serializable=False)
            result.update(nested)
        else:
            # 这是直接工具/元数据 - 保留在当前层级
            direct_tools[name] = aggregated

    # 将直接工具存储在当前前缀下
    if prefix:
        result[prefix] = direct_tools
    else:
        # 在根层级，将直接工具合并到result中
        result.update(direct_tools)

    # 在根层级，汇总所有子代理的token_usage
    if not prefix:
        total_token_usage = _collect_all_token_usage(result)
        if total_token_usage.total > 0:
            result["token_usage"] = [total_token_usage]

    if return_json_serializable:
        # 将所有Pydantic模型转换为JSON可序列化的字典
        return to_json_serializable(result)
    return result


# 消息相关类
class TokenUsage(BaseModel):
    """LLM使用的token计数（输入、输出、推理tokens）。
    
    跟踪LLM API调用中的token消耗，用于：
    1. 成本计算（不同类型的token可能有不同价格）
    2. 上下文窗口管理（确保不超过模型限制）
    3. 性能分析（了解哪里消耗了最多tokens）
    """

    input: int = 0  # 输入tokens数量（提示和上下文）
    output: int = 0  # 输出tokens数量（模型生成的响应）
    reasoning: int = 0  # 推理tokens数量（某些模型如o1系列使用）

    @property
    def total(self) -> int:
        """跨输入、输出和推理的总token计数。"""
        return self.input + self.output + self.reasoning

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """将两个TokenUsage对象相加，独立地对每个字段求和。
        
        这允许轻松累积多次API调用的token使用量。
        """
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            reasoning=self.reasoning + other.reasoning,
        )


class ToolUseCountMetadata(BaseModel):
    """通用元数据，用于跟踪工具使用次数。
    
    实现Addable协议以支持聚合。对于只需要跟踪调用次数的工具，使用此类。
    许多简单工具使用此元数据类型而不是定义自定义元数据。
    """

    num_uses: int = 1  # 工具被使用的次数

    def __add__(self, other: "ToolUseCountMetadata") -> "ToolUseCountMetadata":
        """组合两次工具使用的元数据，累加使用次数。"""
        return ToolUseCountMetadata(num_uses=self.num_uses + other.num_uses)


class ToolResult[M](BaseModel):
    """工具执行器的结果，包含可选的元数据。
    
    泛型类型参数M表示元数据类型。M应该实现Addable协议以支持聚合，
    但由于Pydantic模式生成的限制，这不在类级别强制执行。
    
    工具执行器返回此类型，包含：
    - content: 工具执行的实际输出（可以是文本、图像等）
    - metadata: 关于执行的可选元数据（如使用次数、性能指标等）
    """

    content: Content  # 工具的输出内容
    metadata: M | None = None  # 可选的元数据


class Tool[P: BaseModel, M](BaseModel):
    """工具定义：包含名称、描述、参数模式和执行函数。
    
    泛型类型参数：
        P: 参数模型类型（必须是Pydantic BaseModel，或None表示无参数工具）
        M: 元数据类型（应实现Addable以支持聚合；对于没有元数据的工具使用None）
    
    工具是简单的、无状态的可调用对象。对于需要生命周期管理的工具
    （设置/清理、资源池化），请改用ToolProvider。
    
    带参数的示例：
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
    
    无参数的示例：
        ```python
        time_tool = Tool[None, None](
            name="time",
            description="Get current time",
            executor=lambda _: ToolResult(content=datetime.now().isoformat()),
        )
        ```
    """

    name: str  # 工具名称（必须唯一）
    description: str  # 工具描述（LLM看到此描述以决定何时使用该工具）
    parameters: type[P] | None = None  # 参数模式（Pydantic模型类）
    executor: Callable[[P], ToolResult[M] | Awaitable[ToolResult[M]]]  # 执行函数（同步或异步）


class ToolProvider(ABC):
    """工具提供者抽象基类：带有生命周期管理。
    
    ToolProvider管理资源（HTTP客户端、沙箱、服务器连接）并在进入其异步上下文时
    返回Tool实例。它们实现异步上下文管理器协议。
    
    使用ToolProvider的场景：
    - 需要设置/清理的工具（连接、临时目录）
    - 返回多个Tool实例的工具（例如MCP服务器）
    - 跨调用共享状态的工具（例如HTTP客户端池）
    
    示例：
        class MyToolProvider(ToolProvider):
            async def __aenter__(self) -> Tool | list[Tool]:
                # 设置资源并返回工具
                return self._create_tool()

            # __aexit__是可选的 - 默认为无操作
    
    Agent通过其session()上下文自动管理ToolProvider生命周期。
    """

    @abstractmethod
    async def __aenter__(self) -> "Tool | list[Tool]":
        """进入异步上下文：设置资源并返回工具。
        
        Returns:
            单个Tool实例，或对于暴露多个工具的提供者（例如MCP服务器）
            返回Tool实例列表。
        """
        ...

    async def __aexit__(  # noqa: B027
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """退出异步上下文：清理资源。默认为无操作。
        
        子类可以重写此方法来实现自定义清理逻辑，例如：
        - 关闭网络连接
        - 删除临时文件
        - 释放系统资源
        """


@runtime_checkable
class LLMClient(Protocol):
    """定义LLM客户端实现的接口协议。
    
    任何LLM客户端都必须实现此协议才能与Agent类一起工作。
    提供带工具支持的文本生成和模型能力检查。
    
    实现此协议的示例：
    - ChatCompletionsClient（OpenAI兼容API）
    - LiteLLMClient（多提供商支持）
    """

    @abstractmethod
    async def generate(self, messages: list["ChatMessage"], tools: dict[str, Tool]) -> "AssistantMessage":
        """生成LLM响应。
        
        Args:
            messages: 对话历史消息列表
            tools: 可用工具字典（名称 -> Tool对象）
            
        Returns:
            LLM生成的助手消息，可能包含工具调用
        """
        ...

    @property
    def model_slug(self) -> str:
        """模型标识符字符串（例如："gpt-4"、"claude-3-opus"）。"""
        ...

    @property
    def max_tokens(self) -> int:
        """模型的最大上下文窗口大小（以tokens计）。"""
        ...


class ToolCall(BaseModel):
    """表示来自LLM的工具调用请求。
    
    当LLM决定使用工具时，它会生成一个或多个ToolCall对象。
    每个ToolCall包含工具名称、参数和唯一ID用于跟踪。
    
    Attributes:
        name: 要调用的工具名称
        arguments: 包含工具参数的JSON字符串
        tool_call_id: 用于跟踪此工具调用及其结果的唯一标识符
    """

    name: str  # 工具名称
    arguments: str  # JSON格式的参数
    tool_call_id: str | None = None  # 唯一调用ID


class SystemMessage(BaseModel):
    """系统级指令和上下文消息。
    
    系统消息通常包含对LLM的高级指令、角色定义、约束条件等。
    它设定了对话的基调和规则。
    """

    role: Literal["system"] = "system"
    content: Content  # 系统指令内容


class UserMessage(BaseModel):
    """用户输入消息。
    
    表示来自用户的输入，可以是纯文本，也可以包含图像、视频等多模态内容。
    """

    role: Literal["user"] = "user"
    content: Content  # 用户输入内容


class Reasoning(BaseModel):
    """扩展思考/推理内容，来自支持思维链推理的模型。
    
    某些模型（如OpenAI的o1系列）会生成推理过程，显示它们如何得出答案。
    此类捕获该推理内容和可选的签名。
    """

    signature: str | None = None  # 可选的推理签名
    content: str  # 推理过程内容


class AssistantMessage(BaseModel):
    """LLM响应消息，包含可选的工具调用和token使用跟踪。
    
    这是LLM生成的响应，可能包含：
    - 文本内容（回答、解释等）
    - 工具调用请求（如果LLM决定使用工具）
    - 推理过程（对于支持的模型）
    - Token使用统计
    """

    role: Literal["assistant"] = "assistant"
    reasoning: Reasoning | None = None  # 可选的推理过程
    content: Content  # 响应内容
    tool_calls: Annotated[list[ToolCall], Field(default_factory=list)]  # 工具调用列表
    token_usage: Annotated[TokenUsage, Field(default_factory=TokenUsage)]  # Token使用统计


class ToolMessage(BaseModel):
    """工具执行结果，返回给LLM。
    
    在工具执行后，结果被包装在ToolMessage中并添加到对话历史，
    让LLM可以看到工具执行的结果并据此继续对话。
    """

    role: Literal["tool"] = "tool"
    content: Content  # 工具执行结果
    tool_call_id: str | None = None  # 对应的工具调用ID
    name: str | None = None  # 工具名称
    args_was_valid: bool = True  # 参数是否有效


type ChatMessage = Annotated[SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")]
"""聊天消息类型的判别联合，根据role字段自动解析。

Pydantic会根据role字段自动选择正确的消息类型：
- "system" -> SystemMessage
- "user" -> UserMessage  
- "assistant" -> AssistantMessage
- "tool" -> ToolMessage
"""


class SubAgentMetadata(BaseModel):
    """子代理执行的元数据，包括token使用、消息历史和子运行元数据。
    
    当使用子代理（通过Agent.to_tool()创建）时，此元数据捕获子代理的完整执行信息。
    实现Addable协议以支持跨多次子代理调用的聚合。
    """

    message_history: list[list[ChatMessage]]  # 子代理的消息历史
    run_metadata: Annotated[dict[str, list[Any]], Field(default_factory=dict)]  # 子代理的运行元数据

    def __add__(self, other: "SubAgentMetadata") -> "SubAgentMetadata":
        """组合来自多次子代理调用的元数据。
        
        将两个SubAgentMetadata实例合并：
        1. 串联消息历史（保持执行顺序）
        2. 合并运行元数据（每个键的列表拼接）
        
        这允许跟踪同一子代理的多次调用的累积信息。
        """
        # 串联消息历史
        combined_history = self.message_history + other.message_history
        # 合并运行元数据（每个键拼接列表）
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
