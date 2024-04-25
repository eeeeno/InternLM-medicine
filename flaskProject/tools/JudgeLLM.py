from LLM import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
strr="""
你的作用是帮我判断输入的意思
以下情况返回0：当意思表达为不清楚，意思与“无法为您提供,很抱歉，我不知道，我不清楚，我不太懂，我无法找到相关信息”意思接近
以下情况返回1：明确给出了答案时。
注意：不需要输出其他内容，只需要根据语义输出0或者1.
只有下面两种可能的输出
输出格式:0
输出格式:1
"""

prompt5str="""
你的作用是回答用户问题。我将传输给你用户的问题和历史聊天记录。请你根据这些信息回答用户问题。你只能回答医疗方面的问题，当是其他问题，请你回复:抱歉，我只能回答与医疗相关的问题，无法对外貌进行评价。如果您有任何医疗方面的问题或需要皮肤病检测，请随时告诉我，我会尽力帮助您。如果您有皮肤病照片需要诊断，欢迎上传给我。谢谢！
如果问题是你是谁，你用下面的答案输出：我是张艳婷老师班上,第三组打造的书生浦语医疗问答大模型，我内部是InternLM大模型，我集成了众多工具，比如皮肤病向量知识库、neo4j医疗数据库、huggingface模型，谷歌搜索API、阿里云API.可以回答与任何医疗有关的问题，同时我还支持上传皮肤病照片帮您诊断皮肤病！但您一次只能上传一张照片。

"""


prompt = ChatPromptTemplate.from_messages([
    ("system", strr),
    ("user", "{input}")
])
prompt4 =ChatPromptTemplate.from_messages([
    ("system", "你的作用是帮我判断用户输入的问题。如果是与健康医疗，挂号，吃药，症状，治疗方法等任何与健康医疗有关的问题就返回1，如果是其他问题就返回0。只要是与医疗问答有关的内容就返回1.你不用解释，只需要返回0或者1"),
    ("user", "{input}")
])
prompt5 =ChatPromptTemplate.from_messages([
    ("system", prompt5str),
    ("user", "{input}")
])

prompt2 =ChatPromptTemplate.from_messages([
    ("system", "你是一名极其专业的医生,请你回答用户问题，根据你的知识以及下面网站中搜索到的所有知识综合回复用户问题.网站title用《》括起来。网站链接用<>括起来，返回的格式来说，返回所有参考网站链接并放在最后。返回的网站示例如下: <https://www.uptodate.com/contents/zh-Hans/peptic-ulcer-disease-clinical-manifestations-and-diagnosis>"),
    ("user", "{input}")
])
prompt3 =ChatPromptTemplate.from_messages([
    ("system", "我将输入一段图片的描述,请你帮我把这个描述翻译成中文并插入到下面的输出格式里面，替换掉里面的（）.输出格式：抱歉，根据小医的理解，您的图片好像是（）小医只允许接受皮肤照片用于皮肤病检测，您可以再次上传照片.请保证照片里大部分均为您的皮肤照片"),
    ("user", "{input}")
])
from LLm import LLm
output_parser = StrOutputParser()
Judgechain = prompt | llm | output_parser
summurychain = prompt2 | LLm | output_parser
Translatechain = prompt3 | llm | output_parser
JudgeProblemchain = prompt4 | LLm | output_parser
Answerchain = prompt5 | llm | output_parser
# print(summurychain.invoke({"input": "[{'url': 'https://www.thepaper.cn/newsDetail_forward_25495160', 'content': '6类常用抗病毒流感药物用法用量一览表. 1、普通感冒vs流感. 普通感冒主要是由致病力较弱的鼻病毒、副流感病毒、呼吸道合胞病毒等引起，常年可发病，传染性弱，一般上呼吸道症状明显而全身症状较轻。. 流感多在冬春发病，季节性流感主要由甲型H1N1、H3N2和 ...', 'title': '6类常用抗病毒流感药物用法用量一览表 - 澎湃新闻', 'score': 0.9215}, {'url': 'http://gi.dxy.cn/article/847853', 'content': '近期甲流（甲型流感）高发，用药助手 列出了一份清单，里面包括了 3 大类药物：. 一、流感治疗药品. 1、抗病毒药： 只有流感可用，对流感有效，对新冠感染、普通感冒无效。 2、对症处理药物： 流感 、新冠和普通感冒，症状相似，症状严重时都可以用；也就是如果之前你应对新冠剩下一些药物 ...', 'title': '应对甲流，需要备什么药？（患者版） - 丁香园', 'score': 0.91733}, {'url': 'http://infect.dxy.cn/article/543059', 'content': '大部分流感患者可自愈，为何还要用抗流感药物？ 什么情况下推荐使用抗流感药物治疗？ ... 奥司他韦的最佳给药时间是在流感症状出现的 2 天之内（即 48 小时内），但在发病 4 天后（即 96 小时后）使用也会有一定的疗效。 ... 后两者的益处应该才是我们使用奥 ...', 'title': '大部分流感可自愈，为何还需要抗流感药物？ - 丁香园', 'score': 0.90811}]"}))