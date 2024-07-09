from dotenv import load_dotenv
import os
import streamlit as st # type: ignore

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent, create_openai_functions_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.agent_toolkits import create_sql_agent
from langchain import hub
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.tools.tavily_search import TavilySearchResults

import time
import llm_logic
import init_db
from langchain.globals import set_verbose

set_verbose(True)
load_dotenv()

st.title('🦜🔗 Tavily to search trustable infos 🚀')
st.divider()     

tool = TavilySearchResults()
st.session_state.model = "gpt-4o"

if prompt := st.chat_input("What is your question ?"):
    with st.chat_message("Human"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        instructions = """You are an assistant that give to the user exact answer to the question"""
        base_prompt = hub.pull("langchain-ai/openai-functions-template")
        prompt_partial = base_prompt.partial(instructions=instructions)
        llm = ChatOpenAI(model=st.session_state.model, temperature=0)
        tavily_tool = TavilySearchResults()
        tools = [tavily_tool]
        agent = create_openai_functions_agent(llm, tools, prompt_partial)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
        )
        response = agent_executor.invoke({"input": prompt}, {"callbacks": [st_callback]})

        st.write(response['output'])
