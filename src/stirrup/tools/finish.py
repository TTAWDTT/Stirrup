from typing import Annotated

from pydantic import BaseModel, Field

from stirrup.constants import FINISH_TOOL_NAME
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata


class FinishParams(BaseModel):
    """完成参数模型：用于说明任务完成的原因或无法继续的理由。
    
    该模型定义了代理完成任务时需要提供的信息结构，包括完成原因和相关文件路径。
    这些信息将被保存并返回给调用者，用于记录任务执行结果。
    """

    reason: Annotated[str, Field(description="任务完成的原因说明。应详细描述任务是如何完成的，或者如果无法完成，说明遇到的问题和原因。")]
    paths: Annotated[
        list[str], Field(description="任务执行过程中创建或修改的文件路径列表。仅包含文件路径，不要包含目录。这些文件将被保存到输出目录中。")
    ]


# 简单完成工具：这是代理用来标记任务完成的标准工具
# 当代理判断任务已完成或无法继续进行时，会调用此工具来结束执行循环
# 注意：调用finish工具需要占用一个单独的回合，因此代理需要在接近max_turns之前调用
SIMPLE_FINISH_TOOL: Tool[FinishParams, ToolUseCountMetadata] = Tool[FinishParams, ToolUseCountMetadata](
    name=FINISH_TOOL_NAME,
    description="发出任务完成信号并提供完成原因。当任务已完成或无法继续进行时使用此工具。注意：你需要一个单独的回合来完成任务，所以要提前规划好。",
    parameters=FinishParams,
    executor=lambda params: ToolResult(content=params.reason, metadata=ToolUseCountMetadata()),
)
