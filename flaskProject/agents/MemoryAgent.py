from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
import sys
sys.path.append(r"/root/data/flaskProject/tools")
from langchain.memory import ChatMessageHistory
history = ChatMessageHistory()

from alibabacloud_imageprocess20200320.client import Client
from alibabacloud_imageprocess20200320.models import DetectSkinDiseaseAdvanceRequest
from alibabacloud_tea_util.models import RuntimeOptions
from alibabacloud_tea_openapi.models import Config
from SearchDetailsOfSkinDisease import searchSkins
configApi = Config(
  access_key_id="xxxxxxxx",
  access_key_secret="xxxxxxxx",
  endpoint='xxxxxxxx',
  # 访问的域名对应的region
  region_id='cn-shanghai'
)
runtime_option = RuntimeOptions()
# pip install tavily-python
from tavily import TavilyClient
tavily = TavilyClient(api_key="填入你的apikey")

desc = (
    "When the similarity of skin diseases is the highest.use this tool when given the URL of an image that you'd like to be ."
    "described. .A Q&A on skin conditions will be returned"

    

)
from typing import Optional
class ImageCaptionTool(BaseTool):
    name = "skin captioner"
    description = desc
    
    def _run(self,url:str,question: Optional[str] = None):
        img = open(url, 'rb')
        print("url为")
        print(url)
        print("读取文件成功")
        print("用户问题是")
        print(question)
        requestApi = DetectSkinDiseaseAdvanceRequest()
        requestApi.url_object = img
        requestApi.org_id = " "
        requestApi.org_name = " "
        client = Client(configApi)
        response = client.detect_skin_disease_advance(requestApi, runtime_option)
        print(response.body.data)
        # print(type(response.body.data))
        dict_data = eval(str(response.body.data))
    
        img.close()
        input_="皮肤病诊断结果为"+str(dict_data["Results"])+"尽可能多的搜索返回与概率最高的皮肤病的生病原因，预防措施，治疗方式等信息。"
        print(input_)
        caption=searchSkins(input_)
        print("caption为:")
        print(caption)
        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")



from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.prompts import PromptTemplate

from LLM import llm
from CypherQA import cypher_qa

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm,
        return_direct=True
    ),
    Tool.from_function(
        name="Cypher QA",
        description="When asked about medical and disease-related questions, such as what medicine to take when sick, how to recover quickly, preventive measures, etc., call this methodCypher QA is a tool focused on medical knowledge, answering users' medical questions using the Cypher query language. Users can ask various medical-related questions, such as medication recommendations or disease diagnostic criteria. The tool generates Cypher queries based on user questions and searches through medical knowledge databases to find answers. Cypher QA aims to provide accurate and detailed medical knowledge queries and answers.",
        func = cypher_qa,
        return_direct=True
    ),ImageCaptionTool()

]

agent_prompt = PromptTemplate.from_template("""
You are a medical expert providing information about medical knowledge.
Be as helpful as possible and return as much information as possible.

请你在处理过程中不要翻译我的问题，比如我是用中文问的：成人呼吸窘迫综合征应该挂什么号。你接下来就应该直接传递成人呼吸窘迫综合征应该挂什么号，而不是翻译成英语
你千万不要把我的输入翻译成英语，千万不要！！！你千万不要修改我的问题！！！原封不动的传递下去！！！
Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.
你返回的结果一定要是中文，不能返回英文，只能返回中文！！！
你一定要用工具，用下面的工具之一
TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: 这里输入用户输入的问题
Observation: the result of the action
```
Make the most of the knowledge in Observation, return as much information as possible, and be sure to make the most of it


When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!


history:{history}
New input: {input}
{agent_scratchpad}
""")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )
from JudgeLLM import Judgechain,summurychain
def generate_response(prompt):
    history.add_user_message(prompt)
    # print(prompt)
    print("history是", history.messages)
    flag=-1
    response = agent_executor.invoke({"input": prompt,"history": history.messages})
    try:
        imedi=""
        if isinstance(response['output'], str):
            imedi=response['output']

        # return response['output']
        else:
            imedi=response['output'].get('result')
        flag=Judgechain.invoke({"input":imedi})
    except Exception as e:
        flag="0";
    
    if flag=="0":
        response = tavily._search(query=prompt, search_depth="advanced", max_results=3)
        context = [{"url": obj["url"], "content": obj["content"], "title": obj["title"], "score": obj["score"]} for obj
                   in response["results"]]
        direct_result = ""
        input_=""
        for obj in context:
            direct_result += f"URL: {obj['url']}\tTitle: {obj['title']}\tScore: {obj['score']}\n"
            input_ += f"URL: {obj['content']}\n"
        input_prompt="question是:"+prompt+"搜到的资料为:"+str(context)
        final=summurychain.invoke({"input": input_prompt})
        history.add_ai_message(final)
        return final
    else :
        history.add_ai_message(imedi)
        return imedi
    
    

