import json
import requests
from src.utils import build_output_parser, Llama3PromptBuilder
from langchain.agents import Tool

format_city_prompt_template = """
从下面的文本中提取出想要查询的天气的城市的名称是什么。
文本: 
{question}
输出应该是遵守以下模式的代码片段，包括开头和结尾的"```json"和"```":
```json
{{
	"city": string  // 城市名称
}}
```"""

class WeatherTool():
	def __init__(self, llm, args):
		self.llm = llm
		self.output_parser, self.format_instructions = build_output_parser({'city': '城市名称'})
		self.prompt_builder = Llama3PromptBuilder('你是一个智能助手，可以根据城市名称查询天气信息和天气预报。')
		self.format_city_prompt = self.prompt_builder.build_chat_prompt(format_city_prompt_template)
		self.chain = (
			self.format_city_prompt
			| self.llm
			| self.output_parser
			| self.get_weather_info
		)
		self.cities = json.load(open('tools/city_code.json', 'rb'))

	def get_weather_info(self, schema):
		url = 'http://t.weather.sojson.com/api/weather/city/'
		city = schema['city']
		city = city.replace('市','').replace('区','')
		city_code = self.cities.get(city)
		if city_code is None:
			return {}
		try:
			response = requests.get(url + city_code, timeout=(0.1,0.1))
			d = response.json()
			if(d['status'] == 200):
				return {
					'final_answer': f"""{city}今天的温度是{d['data']['wendu']}度，
					湿度是{d['data']['shidu']}，
					二点五微米颗粒物为{d['data']['pm25']}，
					十微米颗粒物为{d['data']['pm10']}，
					空气质量为{d['data']['quality']}"""
					}
				
			else:
				return {}
		except requests.exceptions.Timeout:
			print("请求超时，请稍后重试。")
			return {}

	def tool_wrapper(self):
		tool = Tool(
            name='Weather',
            func=self.chain.invoke,
            description='用来查询城市的天气相关信息。'
        )
		return tool