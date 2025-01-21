import json
import traceback
import json

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

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
    date
    name

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
    ans = response.content.replace("```json", "")
    ans = ans.replace("```", "")
    return ans
    
    # # 返回模型回應
    # data = json.loads(ans)
    # return data['Result']
    
def generate_hw02(question):
    pass
    
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
result = generate_hw01(question)
print(result)