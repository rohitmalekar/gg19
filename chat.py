import streamlit as st
import os

from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
# from langsmith import Client
from langchain.agents.agent_types import AgentType
from langchain.tools import Tool, tool
from trubrics.integrations.streamlit import FeedbackCollector

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

collector = FeedbackCollector(
    project="default",
    email=st.secrets["TRUBRICS_EMAIL"],
    password=st.secrets["TRUBRICS_PWD"],
)

# client = Client()
st.set_page_config(
    page_title="GrantsScope",
    page_icon="🔎",
    # layout="wide",
    initial_sidebar_state="expanded",
)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

st.sidebar.markdown("## Important Links:")
st.sidebar.markdown("- Check out the new [Explorer](https://explorer.gitcoin.co/#/) with improved search")
st.sidebar.markdown("- GG19 Donation and Leaderboard [Dashboard](https://gitcoin-grants-51f2c0c12a8e.herokuapp.com/)")
st.sidebar.markdown("- [Grants Portal](https://grants-portal.gitcoin.co/gitcoin-grants-grantee-portal) for how-to videos, resources, and FAQs")
st.sidebar.markdown("- Refresh your [Gitcoin Passport](https://passport.gitcoin.co/) score")
st.sidebar.markdown("- Review the [GG19 Outline and Strategy](https://gov.gitcoin.co/t/gg19-outline-and-strategy/16682)")
st.sidebar.markdown("- Understand [Gitcoin Vocabulary](https://gov.gitcoin.co/t/vocabulary-gitcoin-grants-programs-rounds-etc/16773)")
st.sidebar.markdown("- About [GrantsScope](http://grantsscope.xyz/)")

st.title('GrantsScope - GG19')
st.markdown('Ask away your questions to learn more about the grantees in GG19 Program Rounds and Climate Solutions Round. Information on other Community and Independent Rounds is coming soon! See useful links in the side bar.')
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.link_button("Explore all projects", "https://explorer.gitcoin.co/#/projects",type="primary")
with col2:
    st.link_button("Support GrantsScope in GG19", "https://explorer.gitcoin.co/#/round/424/0x98720dd1925d34a2453ebc1f91c9d48e7e89ec29/0x98720dd1925d34a2453ebc1f91c9d48e7e89ec29-195",type="secondary")

@st.cache_resource(ttl="1h")

def configure_retriever():
    index = './storage/faiss'
    embeddings = OpenAIEmbeddings()    
    vectorstore = FAISS.load_local(index, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

discoverer = create_retriever_tool(
    configure_retriever(),
    "Grantee_Discovery",
    "Helps search information about grantees in different Rounds for GG19, use this tool to respond to questions about specific grantees and projects across all rounds",
)

tools = [discoverer]

llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo-16k")
#memory = AgentTokenBufferMemory(llm=llm)

message = SystemMessage(
    content=(
        "Do not respond to questions that ask to sort or rank grantees. Do not respond to questions that ask to compare grantees. Similarly, do not respond to questions that ask for advice on which grantee to donate contributions to. Few examples of such questions are (a) Which grantee had the most impact? (b) Who should I donate to? (c) Rank the grantees by impact (d) Compare work of one grantee versus another? For such questions, do not share any grantee information and just say: ""Dear human, I am told not to influence you with my biases for such queries. The burden of choosing the public greats and saving the future of your kind lies on you. Choose well!"""
        "Only use the context provided to respond to the question. Do not use information outside the context."
        "If the answer is not available in the context information given above, respond: Sorry! I don't have an answer for this."
        "Given this information, please answer the following question. When sharing information about a project, share which round they are part of, the website and the Explorer Link. Format the output as a table if the response includes multiple projects."
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt, )

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    #return_intermediate_steps=True,
)

starter_message = "Ask me anything about the grantees in GG19 Rounds (Program Rounds and Climate Solutions Round)!"

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]

if "logged_prompt" not in st.session_state:
    st.session_state.logged_prompt = None

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    #memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("assistant"):

        latest_messages = ""
        # Capture the latest two responses as additional context to the prompt
        if 'messages' in st.session_state and len(st.session_state.messages) >= 4:
            latest_messages = st.session_state.messages[-4:]
        else:
            latest_messages = st.session_state.messages

        # Debug
        #st.markdown("Additional context includes **** ")
        #for msg in latest_messages:
        #    st.markdown(msg)    
        #st.markdown("***")

        #st_callback = StreamlitCallbackHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        
        try:
            response = agent_executor(
                #{"input": prompt},
                {"input": prompt, "history": latest_messages},
                #callbacks=[st_callback],
                callbacks=[stream_handler],
                include_run_info=True,
            )

            st.session_state.messages.append(AIMessage(content=response["output"]))
            #st.markdown(response["output"])
            #memory.save_context({"input": prompt}, response)
            #st.session_state["messages"] = memory.buffer
            run_id = response["__run"].run_id

            st.session_state.logged_prompt = collector.log_prompt(
                config_model={"model": "gpt-3.5-turbo-16k"},
                prompt=prompt,
                generation=response["output"],
            )

        except:
            st.markdown("The dude who made me doesn't have access to models with longer context yet, or, in English, my brain exploded trying to compress all the information needed to answer your question.")
            st.markdown("Please refresh the browser and try asking this a little differently. I will try to remain sane!")
            st.markdown("![Exploding brain meme](https://media.tenor.com/InOgyW0EIEcAAAAC/exploding-brain-mind-blown.gif)")
            

if st.session_state.logged_prompt:
    user_feedback = collector.st_feedback(
        component="GG19",
        feedback_type="thumbs",
        open_feedback_label="[Optional] Provide additional feedback",
        model="gpt-3.5-turbo-16k",
        prompt_id=st.session_state.logged_prompt.id,
    )
