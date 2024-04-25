from langchain.chains import GraphCypherQAChain
# tag::import-prompt-template[]
from langchain.prompts.prompt import PromptTemplate

from CypherQA import graph
from LLM import llm
# from solutions.graph import graph



cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.
问题{question}如果是英文，请你先转换为中文再根据中文问题从{schema}中创建cypher语句
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure you alias both entities and relationships
properly. Do not run any queries that would add to or delete from
the database. 
If you need to divide numbers, make sure to
filter the denominator to be non zero.

Examples:
#小明得了成人呼吸窘迫综合征，他不能吃什么
MATCH (d:Disease) WHERE d.name = '成人呼吸窘迫综合征'
MATCH (d)-[:no_eat]->(f:Food)
RETURN f.name


#请问膈下脓肿应该吃什么药呢？
MATCH (bs:Disease) WHERE bs.name = '膈下脓肿'
MATCH (bs)-[:recommand_drug]->(s:Drug)
RETURN s.name


#请问膈下脓肿一般有哪些症状
MATCH (bs:Disease) WHERE bs.name = '膈下脓肿'
MATCH (bs)-[:has_symptom]->(s:Symptom)
RETURN s.name


String category values:
节点Disease有以下属性：cause（某个疾病的生病原因），desc（某个疾病的描述），prevent（某个疾病的预防方式）,cure_department(某个病应该去什么部门治疗，也就是挂什么号），
当查询的问题与（疾病信息、挂什么号，预防方式，生病原因）有关时利用cypher语句查询Disease的属性
比如：大楼病综合征的预防措施有哪些
MATCH (d:Disease) WHERE d.name = '大楼病综合征'
RETURN d.prevent
比如：成人呼吸窘迫综合征应该挂什么号
MATCH (d:Disease) WHERE d.name = '大楼病综合征'
RETURN d.cure_department

Schema:
{schema}
The question is:
{question}
"""
# cypher_generation_prompt=PromptTemplate.from_template(cypher_generation_template)
cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)
# ...

qa_generation_template = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a users natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.你回答的最终答案应该是以中文返回

Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

If the information is not empty, you must provide an answer using the
results. If the question involves a time duration, assume the query
results are in units of days unless otherwise specified.

Never say you don't have the right information if there is data in
the query results. Always use the data in the query results.

Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

# from langchain_openai import ChatOpenAI

cypher_qa = GraphCypherQAChain.from_llm(

    cypher_llm=llm,
    qa_llm=llm,
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
return_intermediate_steps=True
)
# a=cypher_qa.invoke("肺炎球菌肺炎不能吃哪些食物呢？")

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "请你修改下面cypher语句,RETURN的结果里一定要有关系！能够保证RETURN的对象有三种比如：RETURN 节点1，关系，节点2..要求1：返回的cypher语句没有换行符分割.要求2：返回的cypher语句的return有三个结果， 你一定要记住一定要返回关系，一定一定要返回关系，一定返回关系。关系在cypher语句中用[r:关系名称]表示，并且返回这个r。返回结果的示例为：MATCH (d:Disease)-[r:no_eat]->(n) WHERE d.name = '成人呼吸窘迫综合征' RETURN d, r, n;."),
   ("user", "{input}")
])
from langchain_core.output_parsers import StrOutputParser
from LLm import LLm
output_parser = StrOutputParser()
chain = prompt | LLm | output_parser
# str=chain.invoke({"input": a['intermediate_steps']})

# print(str)

def generateCypher(str):
    first=cypher_qa.invoke(str)
    mediate=first['intermediate_steps']
    print("中间过程为")
    print(mediate)
    final=first['result']

    result=chain.invoke({"input": mediate})

    result_list = [str, result,final]
    return result_list

# print(generateCypher("肺炎球菌肺炎应该做哪些检查？"))
# 肺炎球菌肺炎应该做哪些检查？