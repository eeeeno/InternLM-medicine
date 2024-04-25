from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

from LLM import llm
from langchain.prompts.prompt import PromptTemplate
graph = Neo4jGraph(
    url="bolt://localhost:7687", username="neo4j", password="xxxxxx"
)


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

qa_generation_template = """
如果Neo4j Cypher query 的结果为[]或空,则你直接返回下面四个字：我不知道
如果Neo4j Cypher query的结果不为空，在回答最后加入 数据来源于：《neo4j数据库》
如果Neo4j Cypher query 的结果为[]或空,则你直接返回下面四个字：我不知道
如果Neo4j Cypher query的结果不为空，在回答最后加入 数据来源于：《neo4j数据库》
如果Neo4j Cypher query 的结果为[]或空,则你直接返回下面四个字：我不知道
如果Neo4j Cypher query的结果不为空，在回答最后加入 数据来源于：《neo4j数据库》
If the provided information is empty, say you don't know the answer.
Empty information looks like this: []
If the provided information is empty, say you don't know the answer.
Empty information looks like this: []
You are an assistant that takes the results
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

from LLm import LLm

cypher_qa = GraphCypherQAChain.from_llm(
cypher_llm=llm,
    qa_llm=LLm,
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)



# chain = GraphCypherQAChain.from_llm(
#     llm, graph=grapha, verbose=True
# ).invoke({"input": prompt})肺炎球菌肺炎应该吃什么药
# cypher_qa.invoke({"query": "流感应该吃什么药?"})
# print(cypher_qa.invoke({"query": "流感应该吃什么药?"}))
print(cypher_qa.invoke({"query": "流行性感冒应该吃什么药?"}))