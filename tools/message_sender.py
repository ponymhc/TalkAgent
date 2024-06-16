import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import parseaddr, formataddr
from src.utils import build_output_parser, Llama3PromptBuilder
from langchain.agents import Tool
import json

# format_email_prompt_template = """
# 从下面的文本中提取出收件人邮箱地址，邮件主题以及邮件内容。

# 注意：如果文本中未明确提及邮件主题，则根据邮件内容总结出合适的主题。

# 文本: 
# {question}
# 输出应该是遵守以下模式的代码片段，包括开头和结尾的"```json"和"```":
# ```json
# {{
# 	"receiver_addr": string  // 收件人邮箱地址,
#     "subject": string // 邮件主题,
#     "content": string // 邮件内容,
# }}
# ```"""

format_email_prompt_template = """Extract the recipient, the subject of the email, and the content of the email from the text below.

Note: If the subject of the email is not explicitly mentioned in the text, summarize an appropriate subject based on the content of the email.

Text:
{question}
The output should be a code snippet that adheres to the following pattern, including the opening and closing "```json" and "```":
```json
{{
    "recipient": string  // recipient,
    "subject": string // subject of the email,
    "content": string // content of the email,
}}
```
"""


class EmailTool:
    def __init__(self, llm, args):
        self.llm = llm
        self.email_config = 'tools/email_config.json'
        self.output_parser, self.format_instructions = build_output_parser({
                                                                            "recipient":'收件人',
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
            recipient_name = schema['recipient']
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
                'success': True,
                'final_answer': '邮件发送成功。'
            }
        except:
            return {
                'success': False,
                'final_answer': '邮件发送失败，请重新尝试'
            }
    
    def tool_wrapper(self):
        tool = Tool(
            name='Email',
            func=self.chain.invoke,
            description="""
用来发送邮件的Action，接受以下三个Action Inputs: 收件人(recipient)、邮件内容(content)、邮件主题(subject)。
"""
        )
        return tool