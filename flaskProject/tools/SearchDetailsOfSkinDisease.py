

from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os

# 定义 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

# 向量数据库持久化路径
persist_directory = 'data_skin/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings
)
from LLM import llm
from langchain.prompts import PromptTemplate

# 我们所构造的 Prompt 模板
template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
请你一定要给出具体的治疗方法，具体的专业名词,回答一定不要笼统
尽可能多的搜索返回与概率最高的皮肤病的生病原因，预防措施，治疗方式等信息
问题: {question}

可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答你不知道。
一定要输出用户问题中患某种皮肤病的概率值，输出的的模板以下面文字开头：根据您上传的照片，小医调用了阿里云的皮肤病诊断API，您最可能
    """

# 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
# 检索问答链回答效果
# question = "银屑病的治疗方式有哪些"

def searchSkins(question):
    result = qa_chain({"query": question})
    return result["result"]


# print("检索问答链回答 question 的结果：")
# print()
