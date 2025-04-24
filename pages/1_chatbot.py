import streamlit as st
import time
import google.generativeai as genai
import os 
import re
import ast
import psycopg2
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings, StorageContext, load_index_from_storage
# from llama_index.core.node_parser import SemanticSplitterNodeParser
from dotenv import load_dotenv
load_dotenv()
# from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_google_genai import GoogleGenerativeAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ['GOOGLE_API_KEY'])


# Set up the HuggingFaceEmbedding class with the required model to use with llamaindex core.
embed_model  = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en")
Settings.embed_model = embed_model
#set up llm 
Settings.llm = Gemini(model='models/gemini-1.5-flash')
# Settings.text_splitter = SemanticSplitterNodeParser(
#                             buffer_size=2, breakpoint_percentile_threshold=95, embed_model=embed_model
#                         )

PERSIST_DIR =  "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

def extract_sql_list(response_text):
    # This pattern extracts a Python-style list from text
    pattern = r"\[.*?\]"

    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        try:
            sql_list = ast.literal_eval(match.group())
            if isinstance(sql_list, list):
                return sql_list
        except Exception as e:
            raise ValueError(f"Failed to parse SQL list from regex match: {e}")
    
    raise ValueError("No valid SQL list found in response.")

def run_queries_from_nl(user_query: str):
    # Step 1: Ask Gemini to convert NL query to PostgreSQL
    genai.configure(api_key="AIzaSyCegBDS6CFQXE0iwf_VnuSbioUnKbQieZQ")
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = f"""
        You are an expert data analyst. The user will give you a natural language question. 
        Return ONLY a valid Python list of PostgreSQL queries as strings to fetch the data needed. 
        Each query should be standalone.
        the table name is : expensetracker
        These are the columns of the table : date	description	amount	transaction_type	category	account_name 
        An example of the values of the table : 01/01/2018	Amazon	11.11	debit	Shopping	Platinum Card
        Natural language query: "{user_query}"
        Remenber to be careful about the case sensitivity and syntax of PostgreSQL
    """
    gemini_response = model.generate_content(prompt)
    
    try:
        sql_queries = extract_sql_list(gemini_response.text.strip()) # Expecting a list of SQL strings
        # print(sql_queries)
        assert isinstance(sql_queries, list)
    except Exception as e:
        raise ValueError(f"Failed to parse Gemini response into SQL list: {e}\nResponse: {gemini_response.text}")
    
    DB_CONFIG = {'dbname': 'expensedb',
                'user': "admin",
                'password': "securepass",
                'host': 'localhost',
                'port': '5432',
                }
    # Step 2: Connect to the database and execute queries
    responses = []
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        for query in sql_queries:
            cursor.execute(query)
            rows = cursor.fetchall()
            responses.append(rows)

        cursor.close()
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Error executing SQL queries: {e}")

    return sql_queries , responses

def tax_queries( query):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response

taxation_tool = Tool(
    name="Taxation",
    func=tax_queries,
    description="Retrieves data from a RAG pipeline with info on the entire subject of taxation. This tool is capable of answering tax-related or law-related queries directly and does not require additional processing."
)

expense_tracker_tool = Tool(
    name="Personal Transaction History",
    func=run_queries_from_nl,
    description="Retrieves financial transaction details from the user's history. This tool can answer specific financial questions directly, such as total spending, category-wise breakdown, and trends. The agent should pass the entire user query to this tool without trying to do any modifications itself. The return value will be a tuple of two lists psql queries and their results, the agent has to interpret and responsed naturally to the user"
)

agent = initialize_agent(
    tools=[taxation_tool, expense_tracker_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Allows reasoning & tool use
    verbose=True,  # Show reasoning process
    handle_parsing_errors=True  # Avoid parsing errors if response format is wrong
)

st.set_page_config(page_title="Tax & Expense Chatbot", layout="wide")

st.title("💬 AI Chat on taxes & your expenses")

# --- Simulate your backend chatbot function (replace this with agent.run) ---
# def agent_response(user_input):
#     # Replace this with your real chatbot
#     return f"**Great question!**\n\nHere's what I found related to: _'{user_input}'_ 🧾\n\nYou might be eligible for certain deductions. Would you like me to check that?"

# --- Session State for Chat Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Collapsible Chat History ---
with st.sidebar:
    st.header("🕘 Chat History")
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Conversation #{len(st.session_state.chat_history)-i+1}", expanded=False):
                st.markdown(f"**You:** {chat['user']}")
                st.markdown(f"**AI:** {chat['bot']}")
    else:
        st.info("No history yet. Start a conversation!")

# --- Display Current Chat ---
for chat in st.session_state.chat_history:
    with st.chat_message("user", avatar="👤"):
        st.markdown(chat["user"])
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(chat["bot"])

# --- User Input ---
user_input = st.chat_input("Ask about taxes, expenses, deductions...")

if user_input:
    # Show user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Typing indicator
    with st.chat_message("assistant", avatar="🤖"):
        typing = st.empty()
        typing.markdown("_Assistant is typing..._")
        time.sleep(1)

        # Get the response (replace this with agent.run)
        response = agent.run(user_input)

        # Stream response word-by-word
        full_text = ""
        bot_output = st.empty()
        for word in response.split():
            full_text += word + " "
            bot_output.markdown(full_text)
            time.sleep(0.03)

    # Save to chat history
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": response
    })

# --- Navigation back to Home ---
# if st.button("🏠 Back to Home"):
#     st.switch_page("Home.py")
