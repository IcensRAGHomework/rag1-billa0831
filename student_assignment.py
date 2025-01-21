import json
import traceback
import json

import requests

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from openai.types.chat.completion_create_params import ResponseFormat
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_core.tools import tool
import calendarific


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

# Calendarific API的配置
calendarific_api_key = 'JKoxdMesIZE3ZjOdoqpT6Ay7425lr1BO'
calendarific_base_url = 'https://api.calendarific.com/v2/holidays'

# def generate_hw01(question):
#     llm = AzureChatOpenAI(
#             model=gpt_config['model_name'],
#             deployment_name=gpt_config['deployment_name'],
#             openai_api_key=gpt_config['api_key'],
#             openai_api_version=gpt_config['api_version'],
#             azure_endpoint=gpt_config['api_base'],
#             temperature=gpt_config['temperature']
#     )

#     prompt_template = """
#     please answer question
#     將輸出格式化為json, 包含以下鍵
#     Result
#     name
#     date

#     問題：{question}
#     """
#     # 創建 PromptTemplate 實例
#     prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

#     # 將問題格式化為提示模板
#     formatted_question = prompt.format(question=question)
    
#     # 使用 HumanMessage 來包裝問題
#     message = HumanMessage(content=formatted_question)

    
#     # 呼叫模型的 invoke 方法來生成回答
#     response = llm.invoke([message])
#     ans = response.content.replace("```json", "")
#     ans = ans.replace("```", "")
#     return ans

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    prompt_template = """
    please answer question
    將輸出格式化為json, 包含以下鍵
    Result
    name
    date

    問題：{question}
    """
    # 創建 PromptTemplate 實例
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

    # 將問題格式化為提示模板
    formatted_question = prompt.format(question=question)
    
    # 使用 HumanMessage 來包裝問題
    message = HumanMessage(content=formatted_question)

    # 呼叫模型的 invoke 方法來生成回答
    response = llm.invoke([message])
    parser = JsonOutputParser()
    try:
        parsed_result = parser.parse(response.content)
    except Exception:
        return {"json parse error"}

    return parsed_result
# 定義一個函數來調用Calendarific API
@tool
def get_holidays(conutry, year, month, language) -> str:
    """ get_holidays

    Args:
        conutry: TThe country parameter must be in the iso-3166 format as specified in the document here. To view a list of countries and regions we support, visit our list of supported countries.
        year: The year you want to return the holidays. We currently support both historical and future years until 2049. The year must be specified as a number eg, 2019
        month: Limits the number of holidays to a particular month. Must be passed as the numeric value of the month [1..12].
        language: Returns the name of the holiday in the official language of the country if available. This defaults to english. This must be passed as the 2-letter ISO639 Language Code. An example is to return all the names of france holidays in french you can just add the parameter like this: fr
    """
    # url = f"{calendarific_base_url}?api_key={calendarific_api_key}&country={country}&year={year}"
    url = 'https://calendarific.com/api/v2/holidays?'
    # parameters = {
    # # Required
    # 'country': 'tw',
    # 'year':    2024,
    # 'api_key': calendarific_api_key,
    # }

    parameters = {  
            "country": conutry,
            "year": year,
            "month": month,
            "language": language,
            "api_key": calendarific_api_key
        }
    response = requests.get(url, parameters)
    if response.status_code == 200:
        data = response.json()

        holidays = data.get("response", {}).get("holidays", [])
        result = {
            "Result": [
                {"date": holiday["date"]["iso"], "name": holiday["name"]}
                for holiday in holidays
            ]
        }
        return json.dumps(result)
    else:
        return {"error": "Could not fetch holidays"}
    
def fetch_holidays_if_valid(ai_msg):
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            if (tool_call["name"].lower() == "get_holidays"):
                return get_holidays.invoke(tool_call["args"])

    return ai_msg
    
def generate_hw02(question):

    # 初始化 LangChain Agent
    tools = [get_holidays]
    # calapi = calendarific.v2('JKoxdMesIZE3ZjOdoqpT6Ay7425lr1BO')
    
    # holidays = calapi.holidays(parameters)
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    llm_with_tools = llm.bind_tools(tools)
    
    # # holidays_llm = json_llm.bind(holidays)
  
    prompt_template = """
    please answer question
    將輸出格式化為json,  {{\"Result\": [{{\"date\": \"YYYY-MM-DD\", \"name\": \"紀念日名稱\"}}]}}

    問題：{question}
    """
    # 創建 PromptTemplate 實例
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

    # 將問題格式化為提示模板
    formatted_question = prompt.format(question=question)

    # 使用 HumanMessage 來包裝問題
    message = HumanMessage(content=formatted_question)
    # llm = prompt | llm
    chain = llm_with_tools | fetch_holidays_if_valid
    result = chain.invoke([message])
    print(result)
    return result
 
    
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

# 測試
question = "2024年台灣10月紀念日有哪些?"
# hw1
result = generate_hw01(question)
print(result)

# hw2
# result = generate_hw02_mine(question)
# print(result)
