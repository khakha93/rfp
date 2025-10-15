from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import AgentStreamParser
import seaborn as sns
import os
import pandas as pd


dir_path = r"C:\kha\project\rfp\data(sample)"
f_names = [os.path.join(dir_path,f) for f in os.listdir(dir_path)]
df_list = [pd.read_csv(f, sep="\t") for f in f_names]


sns.set_style("white")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(model="gpt-4o-mini", temperature=0),
    df_list[:4],
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
)

stream_parser = AgentStreamParser()


def ask(query):
    # 질의에 대한 답변을 출력합니다.
    response = agent.stream({"input": query})

    for step in response:
        stream_parser.process_agent_steps(step)


ask("몇 개의 행이 있어?")


