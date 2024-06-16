import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import parseaddr, formataddr
from src.utils import build_output_parser, Llama3PromptBuilder
from langchain.agents import Tool
import json
import re

# format_email_prompt_template = """
# 从下面的文本中提取出收件人邮箱地址，邮件主题以及邮件内容。

# 注意：如果文本中未明确提及邮件主题，则根据邮件内容总结出合适的主题。

# 文本: 
# {question}
# 输出应该是遵守以下模式的代码片段，包括开头和结尾的"```json"和"```":
# ```json
# {{
# 	"contact": string  // 收件人姓名,
#     "subject": string // 邮件主题,
#     "content": string // 邮件内容,
# }}
# ```"""

format_email_prompt_template = """Extract the contact, the subject of the email, and the content of the email from the text below.

Note: If the subject of the email is not explicitly mentioned in the text, summarize an appropriate subject based on the content of the email.

Text:
{question}
The output must be a code snippet that adheres to the following pattern, including the opening and closing "```json" and "```":
```json
{{
    "contact": string  // must a name of contact,
    "subject": string // subject of the email,
    "content": string // content of the email,
}}
```

do not include other content except this json!!!
"""


class EmailTool():
    def __init__(self, llm, args):
        self.llm = llm
        self.email_config = 'tools/email_config.json'
        self.output_parser, self.format_instructions = build_output_parser({
                                                                            "contact":'收件人',
                                                                            "subject":'邮件主题',
                                                                            "content":'邮件内容',
                                                                            })
        self.prompt_builder = Llama3PromptBuilder('你是一个智能助手，可以帮助用户发送邮件。')
        self.sender_addr, self.secret_key = self._load_email_config()
        self.format_email_prompt = self.prompt_builder.build_chat_prompt(format_email_prompt_template)
        self.contact_list = json.load(open('./tools/contact_list.json','r'))
        self.chain = (
			self.format_email_prompt
			| self.llm
			| self.output_parser
			| self.send_message
		)

    def _load_email_config(self):
        email_config = json.load(open(self.email_config, 'r'))
        email_addr = email_config['email_addr']
        secret_key = email_config['secret_key']
        return email_addr, secret_key

    def _format_addr(self, s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))
    
    def send_message(self, schema):
        try:
            smtpObj = smtplib.SMTP('smtp.qq.com',587)
            smtpObj.ehlo()
            smtpObj.starttls()

            smtpObj.login(self.sender_addr, self.secret_key)
            from_addr = self.sender_addr
            for contact in self.contact_list:
                if contact in schema['contact']:
                    recipient_name = contact
            to_addr = self.contact_list[recipient_name]

            sender_name = from_addr.split('@')[0]
            recipient_addr_name = to_addr.split('@')[0]

            message = MIMEText(schema['content'], 'plain', 'utf-8')
            message['From'] = self._format_addr(f'{sender_name} {from_addr}')
            message['To'] =  self._format_addr(f'{recipient_addr_name} {to_addr}')

            message['Subject'] = Header(schema['subject'], 'utf-8')

            smtpObj.sendmail(from_addr, [to_addr], message.as_string())

            smtpObj.quit()
            return {
                'status': 'successful',
                'final_answer': f'已成功将邮件发送至{recipient_name}的邮箱，如果需要发送其他邮件，请随时吩咐。'
            }
        except:
            return {
                'status': 'failed',
                'final_answer': '邮件发送失败，请重新尝试。'
            }
    
    def tool_wrapper(self):
        tool = Tool(
            name='Email',
            func=self.chain.invoke,
            description="""
用于发送邮件信息的Action，接受以下三个Action Inputs: contact、content、subject。
"""
        )
        return tool