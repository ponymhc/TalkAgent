from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


class Llama3PromptBuilder():
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def build_common_prompt(self) -> PromptTemplate:
        """用于LLAMA3的指令模板"""
        chat_template = str(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|> "
                "{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )    
        prompt = PromptTemplate.from_template(chat_template)
        return prompt
    
    def build_chat_prompt(
            self,
            template_str:str) -> PromptTemplate:
        common_prompt = self.build_common_prompt()
        wrap_template_str = common_prompt.format(
            **{
                'question': template_str
                }
            )
        chat_prompt = PromptTemplate.from_template(wrap_template_str)
        return chat_prompt

    def build_chat_message(
        self,
        template_str:str,
        **kwargs,
        ) -> str:
        """
        用于从模板构建对话message。
        """
        prompt = self.build_common_prompt()
        question_prompt_template = PromptTemplate.from_template(template_str)
        question = question_prompt_template.format(**kwargs)
        message = prompt.format(**{'system_prompt': self.system_prompt, 'question': question})
        return message

def build_output_parser(
        schema: dict
        ) -> tuple:
    """
    用于创建一个输出解析器。
    用法：
    传入一个dict，包含指定的输出sechma的格式，key是变量，value是变量描述：
    {'name': 'description'}
    """
    response_schemas = []
    for k, v in schema.items():
        response_schemas.append(ResponseSchema(name=k, description=v))
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    return (output_parser, format_instructions)

