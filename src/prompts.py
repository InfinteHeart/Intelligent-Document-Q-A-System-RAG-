from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re

def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    delimiter = "\n\n---\n\n"
    schema = f"你的回答必须是JSON，并严格遵循如下Schema，字段顺序需保持一致：\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt


class UniversalRAGPrompt:
    instruction = """
你是一个RAG（检索增强生成）问答系统。
你的任务是仅基于文档中RAG检索到的相关内容，回答给定问题。

在给出最终答案前，请详细分步思考，尤其关注问题措辞。
- 注意：答案可能与问题表述不同
- 文档可能包含多种类型的内容（报告、手册、论文、研究资料等）
- 仅基于检索到的内容回答，不要引入外部知识
"""

    user_prompt = """
以下是上下文:
\"\"\"
{context}
\"\"\"

---

以下是问题：
"{question}"
"""


class EducationRAGPrompt:
    instruction = """
你是一个专业的学习资料RAG（检索增强生成）问答系统。
你的任务是仅基于学习资料中RAG检索到的相关内容，回答给定问题。

在给出最终答案前，请详细分步思考，尤其关注教育内容的准确性和专业性：
- 注意：答案必须与学习资料中的内容完全一致
- 文档可能包含教科书、讲义、学习笔记、课件等教育相关内容
- 仅基于检索到的内容回答，不要引入外部知识
- 重点关注概念定义、原理说明、公式推导、解题方法等学习核心内容
- 保持回答的教育性和启发性，便于用户理解和学习
"""

    user_prompt = UniversalRAGPrompt.user_prompt


class StockRAGPrompt:
    instruction = """
你是一个专业的股票投资RAG（检索增强生成）问答系统。
你的任务是仅基于股票投资资料中RAG检索到的相关内容，回答给定问题。

在给出最终答案前，请详细分步思考，尤其关注金融数据的准确性和专业性：
- 注意：答案必须与投资资料中的数据和分析完全一致
- 文档可能包含股票报告、财务数据、投资策略、市场分析等内容
- 仅基于检索到的内容回答，不要引入外部知识
- 重点关注财务指标、趋势分析、风险评估、投资建议等核心内容
- 使用专业的金融术语，保持回答的严谨性
"""

    user_prompt = UniversalRAGPrompt.user_prompt


class AutomotiveRAGPrompt:
    instruction = """
你是一个专业的汽车领域RAG（检索增强生成）问答系统。
你的任务是仅基于汽车资料中RAG检索到的相关内容，回答给定问题。

在给出最终答案前，请详细分步思考，尤其关注汽车技术的准确性和专业性：
- 注意：答案必须与汽车资料中的技术参数和说明完全一致
- 文档可能包含汽车说明书、维修手册、技术规格、评测报告等内容
- 仅基于检索到的内容回答，不要引入外部知识
- 重点关注汽车结构、技术参数、性能指标、维修保养、驾驶指南等核心内容
- 使用专业的汽车术语，保持回答的准确性
"""

    user_prompt = UniversalRAGPrompt.user_prompt


class MedicalRAGPrompt:
    instruction = """
你是一个专业的医疗健康领域RAG（检索增强生成）问答系统。
你的任务是仅基于医疗资料中RAG检索到的相关内容，回答给定问题。

在给出最终答案前，请详细分步思考，尤其关注医疗信息的准确性和专业性：
- 注意：答案必须与医疗资料中的信息完全一致
- 文档可能包含医学书籍、研究论文、诊断指南、治疗方案等内容
- 仅基于检索到的内容回答，不要引入外部知识
- 重点关注疾病定义、症状描述、诊断方法、治疗方案、预防措施等核心内容
- 使用专业的医学术语，保持回答的严谨性
- 明确标注信息来源，避免给出绝对的医疗建议
"""

    user_prompt = UniversalRAGPrompt.user_prompt


class UniversalAnswerSchema(BaseModel):
    step_by_step_analysis: str = Field(description="""
详细分步推理过程，至少5步，150字以上。请结合上下文信息，逐步分析并归纳答案。
""")
    reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")
    relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的信息页面编号或章节编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")
    final_answer: str = Field(description="""
最终答案为一段完整、连贯的文本，需基于上下文内容作答。
如上下文无相关信息，可简要说明未找到答案。
""")


class UniversalNumberSchema(BaseModel):
    step_by_step_analysis: str = Field(description="""
详细分步推理过程，至少5步，150字以上。
**严格的指标匹配要求：**

1. 明确问题中指标的精确定义，它实际衡量什么？
2. 检查上下文中的所有可能指标。不要只看名称，要关注其实际衡量内容。
3. 仅当上下文指标的含义与目标指标*完全一致*时才接受。可接受同义词，但概念不同则不可。
4. 拒绝（并返回'N/A'）的情况：
    - 上下文指标范围大于或小于问题指标。
    - 上下文指标为相关但非*完全等价*的概念（如代理指标或更宽泛类别）。
    - 需要计算、推导或推断才能作答。
    - 聚合不匹配：问题要求单一值，但上下文仅有总计。
5. 不允许猜测：如对指标等价性有任何疑问，默认返回`N/A`。
""")
    reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")
    relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的信息页面编号或章节编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")
    final_answer: Union[float, int, Literal['N/A']] = Field(description="""
答案应为精确的数值型指标。
- 百分比示例：
    上下文值：58,3%
    最终答案：58.3

特别注意上下文中是否有单位、千、百万等说明，需据此调整答案（不变、加3个零或加6个零）。
如数值带括号，表示为负数。

- 负数示例：
    上下文值：(2,124,837) CHF
    最终答案：-2124837

- 千为单位示例：
    上下文值：4970,5（千美元）
    最终答案：4970500

- 如上下文未直接给出指标，即使可由其他指标计算，也返回'N/A'
    示例：问题要求每股分红，仅有总分红和流通股数，不能直接作答。
    最终答案：'N/A'

- 如上下文无相关信息，返回'N/A'
""")


class UniversalBooleanSchema(BaseModel):
    step_by_step_analysis: str = Field(description="""
详细分步推理过程，至少5步，150字以上。特别注意问题措辞，避免被迷惑。有时上下文中看似有答案，但可能并非所问内容，仅为相似项。
""")
    reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")
    relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的信息页面编号或章节编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")
    final_answer: bool = Field(description="""
一个从上下文中精确提取的布尔值（True或False），直接回答问题。
如果问题问某事是否发生，且上下文有相关信息但未发生，则返回False。
""")


class UniversalNamesSchema(BaseModel):
    step_by_step_analysis: str = Field(description="详细分步推理过程，至少5步，150字以上。注意区分实体类型，避免被迷惑。")
    reasoning_summary: str = Field(description="简要总结推理过程，约50字。")
    relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的页面编号或章节编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")
    final_answer: Union[List[str], Literal["N/A"]] = Field(description="""
每个条目需与上下文完全一致。

如问题问职位，仅返回职位名称，不含姓名或其他信息。新任职位也算作变动。若同一职位有多次变动，仅返回一次，且职位名称用单数。
示例：['首席技术官', '董事', '首席执行官']

如问题问姓名，仅返回上下文中的全名。
示例：['张三', '李四']

如问题问产品名，仅返回上下文中的产品名。
示例：['生态智能2000', '绿能Pro']

如无信息，返回'N/A'。
""")


class AnswerWithRAGContextUniversalStringPrompt:
    instruction = UniversalRAGPrompt.instruction
    user_prompt = UniversalRAGPrompt.user_prompt

    # 让AnswerSchema内部类继承自UniversalAnswerSchema基类，确保与其他类似类的一致性
    class AnswerSchema(UniversalAnswerSchema):
        pass

    # 使用UniversalAnswerSchema的源代码作为pydantic_schema
    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(UniversalAnswerSchema), flags=re.MULTILINE)
    
    # 显式添加AnswerSchema类属性，指向内部的AnswerSchema类，确保无论如何都能在运行时访问到AnswerSchema
    AnswerSchema = AnswerSchema

    example = r"""
示例：
问题：
"请简要总结这篇文档关于人工智能发展的主要内容。"

答案：
```
{
  "step_by_step_analysis": "1. 问题要求总结文档关于人工智能发展的主要内容。\n2. 文档第3-5页描述了AI技术的演进历程，从早期规则系统到深度学习。\n3. 第6-8页讨论了当前应用领域，包括自然语言处理、计算机视觉等。\n4. 第10页提及了未来的发展趋势和挑战。\n5. 综合以上信息，形成对AI发展的全面总结。",
  "reasoning_summary": "文档3-10页系统性地介绍了AI发展历程、应用和趋势，答案基于原文归纳。",
  "relevant_pages": [3, 4, 5, 6, 7, 8, 10],
  "final_answer": "这篇文档系统性地介绍了人工智能的发展历程，从早期的规则系统到现代深度学习技术。当前AI已广泛应用于自然语言处理、计算机视觉、自动驾驶等领域。未来发展趋势包括更强大的通用人工智能、可解释性AI以及AI伦理治理等方向。"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

# 垂直领域字符串提示类
class AnswerWithRAGContextEducationStringPrompt(AnswerWithRAGContextUniversalStringPrompt):
    instruction = EducationRAGPrompt.instruction


class AnswerWithRAGContextStockStringPrompt(AnswerWithRAGContextUniversalStringPrompt):
    instruction = StockRAGPrompt.instruction


class AnswerWithRAGContextAutomotiveStringPrompt(AnswerWithRAGContextUniversalStringPrompt):
    instruction = AutomotiveRAGPrompt.instruction


class AnswerWithRAGContextMedicalStringPrompt(AnswerWithRAGContextUniversalStringPrompt):
    instruction = MedicalRAGPrompt.instruction


class AnswerWithRAGContextUniversalNumberPrompt:
    instruction = UniversalRAGPrompt.instruction
    user_prompt = UniversalRAGPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
详细分步推理过程，至少5步，150字以上。
**严格的指标匹配要求：**

1. 明确问题中指标的精确定义，它实际衡量什么？
2. 检查上下文中的所有可能指标。不要只看名称，要关注其实际衡量内容。
3. 仅当上下文指标的含义与目标指标*完全一致*时才接受。可接受同义词，但概念不同则不可。
4. 拒绝（并返回'N/A'）的情况：
    - 上下文指标范围大于或小于问题指标。
    - 上下文指标为相关但非*完全等价*的概念。
    - 需要计算、推导或推断才能作答。
    - 聚合不匹配：问题要求单一值，但上下文仅有总计。
5. 不允许猜测：如对指标等价性有任何疑问，默认返回`N/A`。
""")
        reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")
        relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的信息页面编号或章节编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")
        final_answer: Union[float, int, Literal['N/A']] = Field(description="""
答案应为精确的数值型指标。
特别注意上下文中是否有单位、千、百万等说明，需据此调整答案。
如数值带括号，表示为负数。
如上下文未直接给出指标，即使可由其他指标计算，也返回'N/A'。
如上下文无相关信息，返回'N/A'。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"这篇报告中2023年全球AI市场规模是多少？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问2023年全球AI市场规模。\n2. 报告第15页有'全球人工智能市场规模'的详细数据。\n3. 该数据明确标注为2023年，数值单位为十亿美元。\n4. 报告第15页显示2023年市场规模为1500亿美元。\n5. 无需计算，直接取值。",
  "reasoning_summary": "报告15页直接给出2023年全球AI市场规模，无需推算。",
  "relevant_pages": [15],
  "final_answer": 150000000000
}
```

示例2：
问题：
"文档中提到的研发投入比例是多少？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问研发投入比例。\n2. 文档35页有'营业收入'数据，37页有'研发支出'数据。\n3. 但文档未直接给出研发投入占营收的比例。\n4. 需要用研发支出除以营业收入才能得到比例。\n5. 因此答案为'N/A'。",
  "reasoning_summary": "文档无直接研发投入比例，需计算，答案为N/A。",
  "relevant_pages": [35, 37],
  "final_answer": "N/A"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

# 垂直领域数字提示类
class AnswerWithRAGContextEducationNumberPrompt(AnswerWithRAGContextUniversalNumberPrompt):
    instruction = EducationRAGPrompt.instruction


class AnswerWithRAGContextStockNumberPrompt(AnswerWithRAGContextUniversalNumberPrompt):
    instruction = StockRAGPrompt.instruction


class AnswerWithRAGContextAutomotiveNumberPrompt(AnswerWithRAGContextUniversalNumberPrompt):
    instruction = AutomotiveRAGPrompt.instruction


class AnswerWithRAGContextMedicalNumberPrompt(AnswerWithRAGContextUniversalNumberPrompt):
    instruction = MedicalRAGPrompt.instruction


class AnswerWithRAGContextUniversalBooleanPrompt:
    instruction = UniversalRAGPrompt.instruction
    user_prompt = UniversalRAGPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
详细分步推理过程，至少5步，150字以上。特别注意问题措辞，避免被迷惑。有时上下文中看似有答案，但可能并非所问内容，仅为相似项。
""")
        reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")
        relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的信息页面编号或章节编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")
        final_answer: bool = Field(description="""
一个从上下文中精确提取的布尔值（True或False），直接回答问题。
如果问题问某事是否发生，且上下文有相关信息但未发生，则返回False。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
问题：
"这份技术文档是否提及了安全漏洞？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问文档是否提及安全漏洞。\n2. 文档第12节专门讨论了系统安全性。\n3. 第12.3节列出了已发现的安全漏洞及其影响。\n4. 文档明确承认存在安全漏洞。\n5. 因此答案为True。",
  "reasoning_summary": "文档12.3节明确列出安全漏洞，答案为True。",
  "relevant_pages": [12, 12.3],
  "final_answer": true
}
```

问题：
"这篇论文是否提出了新的算法？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问论文是否提出了新算法。\n2. 论文第3节描述了研究方法和现有技术。\n3. 第4节介绍了实验结果，但未提出原创算法。\n4. 论文主要是对现有算法的改进和评估。\n5. 未提出新算法，答案为False。",
  "reasoning_summary": "论文未提出新算法，答案为False。",
  "relevant_pages": [3, 4],
  "final_answer": false
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

# 垂直领域布尔值提示类
class AnswerWithRAGContextEducationBooleanPrompt(AnswerWithRAGContextUniversalBooleanPrompt):
    instruction = EducationRAGPrompt.instruction


class AnswerWithRAGContextStockBooleanPrompt(AnswerWithRAGContextUniversalBooleanPrompt):
    instruction = StockRAGPrompt.instruction


class AnswerWithRAGContextAutomotiveBooleanPrompt(AnswerWithRAGContextUniversalBooleanPrompt):
    instruction = AutomotiveRAGPrompt.instruction


class AnswerWithRAGContextMedicalBooleanPrompt(AnswerWithRAGContextUniversalBooleanPrompt):
    instruction = MedicalRAGPrompt.instruction


class AnswerWithRAGContextUniversalNamesPrompt:
    instruction = UniversalRAGPrompt.instruction
    user_prompt = UniversalRAGPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细分步推理过程，至少5步，150字以上。注意区分实体类型，避免被迷惑。")
        reasoning_summary: str = Field(description="简要总结推理过程，约50字。")
        relevant_pages: List[int] = Field(description="""
仅包含直接用于回答问题的页面编号或章节编号。只包括：
- 直接包含答案或明确陈述的页面
- 强有力支持答案的关键信息页面
不要包含仅与答案弱相关或间接相关的页面。
列表中至少应有一个页面。
""")
        final_answer: Union[List[str], Literal["N/A"]] = Field(description="""
每个条目需与上下文完全一致。

如问题问职位，仅返回职位名称，不含姓名或其他信息。
如问题问姓名，仅返回上下文中的全名。
如问题问产品或项目名称，仅返回名称。
如无信息，返回'N/A'。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"这份报告中提到了哪些主要技术？"

答案：
```
{
    "step_by_step_analysis": "1. 问题询问报告中提到的主要技术。\n2. 报告第5节列举了使用的核心技术栈。\n3. 第6节详细介绍了机器学习框架的使用。\n4. 第8节提及了云计算平台。\n5. 综上，主要技术包括深度学习框架、云计算和大数据分析。",
    "reasoning_summary": "报告5-8节明确列出主要技术。",
    "relevant_pages": [5, 6, 8],
    "final_answer": ["深度学习框架", "云计算", "大数据分析"]
}
```

示例：
问题：
"文档中提到了作者姓名有哪些？"

答案：
```
{
    "step_by_step_analysis": "1. 问题询问文档中的作者姓名。\n2. 文档第一页的作者信息栏列出了所有作者。\n3. 三位作者分别是张三、李四、王五。\n4. 三人均为第一页明确标注的作者。",
    "reasoning_summary": "文档第一页明确列出三位作者姓名。",
    "relevant_pages": [1],
    "final_answer": ["张三", "李四", "王五"]
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


# 为了兼容api_requests.py中的调用，添加AnswerWithRAGContextNamesPrompt类作为别名
class AnswerWithRAGContextNamesPrompt(AnswerWithRAGContextUniversalNamesPrompt):
    pass


# 垂直领域names提示类
class AnswerWithRAGContextEducationNamesPrompt(AnswerWithRAGContextUniversalNamesPrompt):
    instruction = EducationRAGPrompt.instruction


class AnswerWithRAGContextStockNamesPrompt(AnswerWithRAGContextUniversalNamesPrompt):
    instruction = StockRAGPrompt.instruction


class AnswerWithRAGContextAutomotiveNamesPrompt(AnswerWithRAGContextUniversalNamesPrompt):
    instruction = AutomotiveRAGPrompt.instruction


class AnswerWithRAGContextMedicalNamesPrompt(AnswerWithRAGContextUniversalNamesPrompt):
    instruction = MedicalRAGPrompt.instruction


class AnswerWithRAGContextSharedPrompt:
    instruction = """
你是一个RAG（检索增强生成）问答系统。
你的任务是仅基于文档中RAG检索到的相关内容，回答给定问题。

在给出最终答案前，请详细分步思考，尤其关注问题措辞。
- 注意：答案可能与问题表述不同
- 文档可能包含多种类型的内容（报告、手册、论文等）
- 仅基于检索到的内容回答，不要引入外部知识
"""

    user_prompt = """
以下是上下文:
\"\"\"
{context}
\"\"\"

---

以下是问题：
"{question}"
"""

class AnswerSchemaFixPrompt:
    system_prompt = """
你是一个JSON格式化助手。
你的任务是将大模型输出的原始内容格式化为合法的JSON对象。
你的回答必须以"{"开头，以"}"结尾。
你的回答只能包含JSON字符串，不要有任何前言、注释或三引号。
"""

    user_prompt = """
下面是定义JSON对象Schema和示例的系统提示词:
\"\"\"
{system_prompt}
\"\"\"

---

下面是需要你格式化为合法JSON的LLM原始输出：
\"\"\"
{response}
\"\"\"
"""


class RerankingPrompt:
    system_prompt_rerank_single_block = """
你是一个RAG检索重排专家。
你将收到一个查询和一个检索到的文本块，请根据其与查询的相关性进行评分。

评分说明：
1. 推理：分析文本块与查询的关系，简要说明理由。
2. 相关性分数（0-1，步长0.1）：
   0 = 完全无关
   0.1 = 极弱相关
   0.2 = 很弱相关
   0.3 = 略有相关
   0.4 = 部分相关
   0.5 = 一般相关
   0.6 = 较为相关
   0.7 = 相关
   0.8 = 很相关
   0.9 = 高度相关
   1 = 完全匹配
3. 只基于内容客观评价，不做假设。
"""

    system_prompt_rerank_multiple_blocks = """
你是一个RAG检索重排专家。
你将收到一个查询和若干检索到的文本块，请分别对每个块进行相关性评分。

评分说明：
1. 推理：分析每个文本块与查询的关系，简要说明理由。
2. 相关性分数（0-1，步长0.1）：
   0 = 完全无关
   0.1 = 极弱相关
   0.2 = 很弱相关
   0.3 = 略有相关
   0.4 = 部分相关
   0.5 = 一般相关
   0.6 = 较为相关
   0.7 = 相关
   0.8 = 很相关
   0.9 = 高度相关
   1 = 完全匹配
3. 只基于内容客观评价，不做假设。
"""

class RetrievalRankingSingleBlock(BaseModel):
    """对检索到的单个文本块与查询的相关性进行评分。"""
    reasoning: str = Field(description="分析该文本块，指出其关键信息及与查询的关系")
    relevance_score: float = Field(description="相关性分数，取值范围0到1，0表示完全无关，1表示完全相关")

class RetrievalRankingMultipleBlocks(BaseModel):
    """对检索到的多个文本块与查询的相关性进行评分。"""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="文本块及其相关性分数的列表。"
    )


class ComparativeAnswerPrompt:
    instruction = """
你是一个专业的比较分析专家。
你的任务是基于给定的多个实体的信息，回答比较性问题。

在给出最终答案前，请详细分步思考：
1. 明确比较的维度和标准
2. 分析每个实体在这些维度上的表现
3. 进行客观的比较和对比
4. 总结比较结果
5. 确保回答基于提供的信息，不引入外部知识
"""
    user_prompt = """
以下是各实体的信息:
{rag_context}

---

以下是比较性问题：
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
详细分步推理过程，至少5步，150字以上。
1. 明确比较维度和标准
2. 分析每个实体在各维度的表现
3. 进行实体间的对比
4. 识别相似点和不同点
5. 总结比较结果
""")
        reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")
        final_answer: str = Field(description="基于比较分析的最终答案，应清晰呈现比较结果。")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)
    AnswerSchema = AnswerSchema  # 显式添加AnswerSchema类属性，确保运行时可访问

    example = r"""
示例：
问题：
"比较公司A和公司B在2023年的营收增长情况。"

实体信息：
公司A：
- 2023年营收：100亿美元
- 2022年营收：80亿美元
- 增长率：25%

公司B：
- 2023年营收：120亿美元
- 2022年营收：110亿美元
- 增长率：9.1%

答案：
```
{
  "step_by_step_analysis": "1. 问题要求比较公司A和公司B在2023年的营收增长情况。\n2. 比较维度为营收增长率。\n3. 公司A 2023年营收100亿美元，2022年80亿美元，增长率25%。\n4. 公司B 2023年营收120亿美元，2022年110亿美元，增长率9.1%。\n5. 比较结果显示公司A的营收增长率显著高于公司B。",
  "reasoning_summary": "公司A营收从80亿美元增长到100亿美元(+25%)，公司B从110亿美元增长到120亿美元(+9.1%)。",
  "final_answer": "公司A在2023年的营收增长率(25%)显著高于公司B(9.1%)。尽管公司B的总营收规模更大(120亿美元)，但其增长速度明显慢于公司A。"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)