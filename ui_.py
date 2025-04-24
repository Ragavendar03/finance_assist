import pandas as pd
from sqlalchemy import create_engine, text
# import google.generativeai as genai
import psycopg2
import re
import ast
import os 
# from llama_index.llms.gemini import Gemini
# from llama_index.core import Settings, StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.node_parser import SemanticSplitterNodeParser
from dotenv import load_dotenv
load_dotenv()
# from llama_cloud_services import LlamaParse
import tempfile
import streamlit as st
import pandas as pd
import os
# from langchain_core.tools import tool
# from langchain.agents import initialize_agent, AgentType
# from langchain.tools import Tool
# from langchain_google_genai import GoogleGenerativeAI

# import nest_asyncio
# nest_asyncio.apply()

os.environ["STREAMLIT_WATCHDOG_USE_POLLING"] = "true"

os.environ['GOOGLE_API_KEY'] = 'AIzaSyCegBDS6CFQXE0iwf_VnuSbioUnKbQieZQ'
os.environ['LLAMA_CLOUD_API_KEY']='llx-s3cwTyhRCT6TOOVsjXNHpX2qhm7tJcru6UVAbYGCPtzHkaqP'

# llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ['GOOGLE_API_KEY'])


# # Set up the HuggingFaceEmbedding class with the required model to use with llamaindex core.
# embed_model  = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en")
# Settings.embed_model = embed_model
# #set up llm 
# Settings.llm = Gemini(model='models/gemini-1.5-flash')
# Settings.text_splitter = SemanticSplitterNodeParser(
#                             buffer_size=2, breakpoint_percentile_threshold=95, embed_model=embed_model
#                         )
# PERSIST_DIR =  "./llama_parsed_store/storage"
# storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
# index = load_index_from_storage(storage_context)

def push_csv_to_db(df):
    table_name = "expensetracker"                         # Table name in the DB
    db_user = "admin"
    db_password = "securepass"
    db_host = "localhost"
    db_port = "5432"
    db_name = "expensedb"

    # === LOAD CSV ===
    # df = pd.read_csv(uploaded_file)

    # === CREATE DB ENGINE ===
    engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    # df.columns = df.columns.str.lower()
    # === PUSH TO DB ===
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y").dt.date
    df.to_sql(table_name, engine, index=False, if_exists='replace')  # Use 'append' to add without dropping
    
    alter_query = """
                ALTER TABLE expensetracker
                ALTER COLUMN date TYPE DATE
                USING date::DATE;
                """

    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    
    cur = conn.cursor()
    cur.execute(alter_query)
    conn.commit()

    print(f"✅ CSV pushed to PostgreSQL table: {table_name}")
    print(f"Table {table_name} altered successfully.")
def add_data_to_expensedb(data : dict):
    db_user = "admin"
    db_password = "securepass"
    db_host = "localhost"
    db_port = "5432"
    db_name = "expensedb"
    table_name = "expensetracker"

    df = pd.DataFrame(data)
    engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    # === Manual Upsert Logic ===
    with engine.begin() as conn:
        for _, row in df.iterrows():
            st.write(row)
            # Check if the row exists
            check_query = text(f"""
                SELECT 1 FROM {table_name}
                WHERE date = :date
                AND description = :description
                AND amount = :amount
            """)
            result = conn.execute(check_query, {
                "date": row["date"],
                "description": row["description"],
                "amount": row["amount"]
            }).fetchone()

            if result:
                # Update existing row
                update_query = text(f"""
                    UPDATE {table_name}
                    SET transaction_type = :transaction_type,
                        category = :category,
                        account_name = :account_name
                    WHERE date = :date
                    AND description = :description
                    AND amount = :amount
                """)
                conn.execute(update_query, {
                    "transaction_type": row["transaction_type"],
                    "category": row["category"],
                    "account_name": row["account_name"],
                    "date": row["date"],
                    "description": row["description"],
                    "amount": row["amount"]
                })
            else:
                # Insert new row
                insert_query = text(f"""
                    INSERT INTO {table_name} (
                        date, description, amount,
                        transaction_type, category, account_name
                    ) VALUES (
                        :date, :description, :amount,
                        :transaction_type, :category, :account_name
                    )
                """)
                conn.execute(insert_query, {
                    "date": row["date"],
                    "description": row["description"],
                    "amount": row["amount"],
                    "transaction_type": row["transaction_type"],
                    "category": row["category"],
                    "account_name": row["account_name"]
                })

    print("✅ Data updated or inserted successfully.")
def see_the_updated_table():
    db_user = "admin"
    db_password = "securepass"
    db_host = "localhost"
    db_port = "5432"
    db_name = "expensedb"
    table_name = "expensetracker"
    engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    df = pd.read_sql_table(table_name, engine)
    df.tail()


# def extract_sql_list(response_text):
#     # This pattern extracts a Python-style list from text
#     pattern = r"\[.*?\]"

#     match = re.search(pattern, response_text, re.DOTALL)
#     if match:
#         try:
#             sql_list = ast.literal_eval(match.group())
#             if isinstance(sql_list, list):
#                 return sql_list
#         except Exception as e:
#             raise ValueError(f"Failed to parse SQL list from regex match: {e}")
    
#     raise ValueError("No valid SQL list found in response.")

# def run_queries_from_nl(user_query: str):
#     # Step 1: Ask Gemini to convert NL query to PostgreSQL
#     genai.configure(api_key="AIzaSyCegBDS6CFQXE0iwf_VnuSbioUnKbQieZQ")
#     model = genai.GenerativeModel("models/gemini-1.5-flash")
#     prompt = f"""
#         You are an expert data analyst. The user will give you a natural language question. 
#         Return ONLY a valid Python list of PostgreSQL queries as strings to fetch the data needed. 
#         Each query should be standalone.
#         the table name is : expensetracker
#         These are the columns of the table : date	description	amount	transaction_type	category	account_name 
#         An example of the values of the table : 01/01/2018	Amazon	11.11	debit	Shopping	Platinum Card
#         Natural language query: "{user_query}"
#         Remenber to be careful about the case sensitivity and syntax of PostgreSQL
#     """
#     gemini_response = model.generate_content(prompt)
    
#     try:
#         sql_queries = extract_sql_list(gemini_response.text.strip()) # Expecting a list of SQL strings
#         # print(sql_queries)
#         assert isinstance(sql_queries, list)
#     except Exception as e:
#         raise ValueError(f"Failed to parse Gemini response into SQL list: {e}\nResponse: {gemini_response.text}")
    
#     DB_CONFIG = {'dbname': 'expensedb',
#                 'user': "admin",
#                 'password': "securepass",
#                 'host': 'localhost',
#                 'port': '5432',
#                 }
#     # Step 2: Connect to the database and execute queries
#     responses = []
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         cursor = conn.cursor()

#         for query in sql_queries:
#             cursor.execute(query)
#             rows = cursor.fetchall()
#             responses.append(rows)

#         cursor.close()
#         conn.close()
#     except Exception as e:
#         raise RuntimeError(f"Error executing SQL queries: {e}")

#     return sql_queries , responses

# def configure_llama_index():    
#     # Set up the HuggingFaceEmbedding class with the required model to use with llamaindex core.
#     embed_model  = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en")
#     Settings.embed_model = embed_model
#     #set up llm 
#     Settings.llm = Gemini(model='models/gemini-1.5-flash')
#     Settings.text_splitter = SemanticSplitterNodeParser(
#                                 buffer_size=2, breakpoint_percentile_threshold=95, embed_model=embed_model
#                             )

# def load_llama_indexDB(PERSIST_DIR =  "./llama_parsed_store/storage"):
    
#     PERSIST_DIR = PERSIST_DIR

#     if not os.path.exists(PERSIST_DIR):
#         parser = LlamaParse(
#         result_type="markdown"  # "markdown" and "text" are available
#         )
#         file_extractor = {".pdf": parser}
#         # load the documents and create the index
#         documents = SimpleDirectoryReader(input_dir=r"D:\finance_rag\data_pdfs", file_extractor=file_extractor).load_data()
#         index = VectorStoreIndex.from_documents(documents)
#         # store it for later
#         index.storage_context.persist(persist_dir=PERSIST_DIR)
#     else:
#         # load the existing index
#         storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#         index = load_index_from_storage(storage_context)

#     return index 

# def tax_queries( query):
#     query_engine = index.as_query_engine()
#     response = query_engine.query(query)
#     return response

# taxation_tool = Tool(
#     name="Taxation",
#     func=tax_queries,
#     description="Retrieves data from a RAG pipeline with info on the entire subject of taxation. This tool is capable of answering tax-related or law-related queries directly and does not require additional processing."
# )

# expense_tracker_tool = Tool(
#     name="Personal Transaction History",
#     func=run_queries_from_nl,
#     description="Retrieves financial transaction details from the user's history. This tool can answer specific financial questions directly, such as total spending, category-wise breakdown, and trends. The agent should pass the entire user query to this tool without trying to do any modifications itself. The return value will be a tuple of two lists psql queries and their results, the agent has to interpret and responsed naturally to the user"
# )

# agent = initialize_agent(
#     tools=[taxation_tool, expense_tracker_tool],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Allows reasoning & tool use
#     verbose=True,  # Show reasoning process
#     handle_parsing_errors=True  # Avoid parsing errors if response format is wrong
# )

# embed_model  = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en")
# Settings.embed_model = embed_model
#set up llm 
# configure_llama_index()
# load_llama_indexDB()

st.set_page_config(page_title="Expense Tracker", layout="centered")

st.markdown("<h1 style='text-align: center;'>💵Agentic Assistant💵</h1>", unsafe_allow_html=True)
st.markdown("## 📁 Upload Your Expense CSV")

if "expense_df" not in st.session_state:
    st.session_state.expense_df = None

if "user_query" not in st.session_state:
    st.session_state.user_query = ""

uploaded_file = st.file_uploader("Drop your CSV here 👇", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.date
    except:
        pass
    st.session_state.expense_df = df
    push_csv_to_db(df)
    st.write('your data has been pushed to the database')
    st.session_state.uploaded_filename = uploaded_file.name  # Save filename for reuse

if st.session_state.expense_df is not None:
    df = st.session_state.expense_df
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    st.markdown("## ✍ Manually Add a New Expense Entry")

    # with st.form("entry_form", clear_on_submit=True):
    #     columns = df.columns.tolist()
    #     new_data = {}
    #     for col in columns:
    #         col = col.lower()
    #         if "date" in col.lower():
    #             new_data[col] = st.date_input(f"{col}")
    #         elif df[col].dtype in ["float64", "int64"]:
    #             new_data[col] = st.number_input(f"{col}", step=0.01)
    #         else:
    #             new_data[col] = st.text_input(f"{col}")
        
    #         submitted = st.form_submit_button("➕ Add Entry")
    
    with st.form("entry_form", clear_on_submit=True):
        date = st.date_input("Date")
        description = st.text_input("Description")
        amount = st.number_input("Amount", step=0.01)
        transaction_type = st.selectbox("Transaction Type", ["debit", "credit"])
        category = st.text_input("Category")
        account_name = st.text_input("Account Name")
        
        submitted = st.form_submit_button("➕ Add Entry")

    
    if submitted:
        st.write("📦 Data Submitted:")
        new_data = {
            "date": [date.strftime("%Y-%m-%d")],
            "description": [description],
            "amount": [amount],
            "transaction_type": [transaction_type],
            "category": [category],
            "account_name": [account_name]
        }

        st.write("📦 Data Submitted:")
        add_data_to_expensedb(new_data)

    # if submitted:
        # add_data_to_expensedb(new_data)
        new_row = pd.DataFrame(new_data)
        
        for col in new_row.columns:
            if "date" in col.lower():
                new_row[col] = pd.to_datetime(new_row[col]).dt.date

        st.session_state.expense_df = pd.concat([df, new_row], ignore_index=True)

        # Save updated CSV to a temp file

        temp_dir = tempfile.gettempdir()
        csv_path = os.path.join(temp_dir, st.session_state.uploaded_filename)

        st.session_state.expense_df.to_csv(csv_path, index=False)

        st.success(f"✅ New entry added and saved to {st.session_state.uploaded_filename}!")
        st.dataframe(st.session_state.expense_df.tail(), use_container_width=True)

        # Optional: Download updated CSV
        st.download_button("⬇ Download Updated CSV",
                           data=st.session_state.expense_df.to_csv(index=False),
                           file_name=st.session_state.uploaded_filename,
                           mime="text/csv")

    st.markdown("---")
    st.markdown("## 🔍 Ask AI About Your Expenses and taxes")

    # def run_queries_from_nl(user_query):
    #     dummy_query = "SELECT * FROM expensetracker LIMIT 5;"
    #     dummy_result = df.head(5).values.tolist()
    #     return [dummy_query], [dummy_result]

    # Initialize session state
    # if "user_query" not in st.session_state:
    #     st.session_state.user_query = ""
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []

    # # Input box
    # user_query = st.text_input("💬 Ask your question about taxes:")

    # # Handle new input
    # if user_query:
    #     # Only run if this is a *new* question
    #     if user_query != st.session_state.user_query:
    #         st.session_state.user_query = user_query
    #         # Run the agent and get response
    #         response = agent.run(user_query)
    #         # Store the conversation
    #         st.session_state.chat_history.append((user_query, response))

    # # Display conversation history
    # for q, a in st.session_state.chat_history:
    #     st.markdown(f"**🧑 You:** {q}")
    #     st.markdown(f"**🤖 Agent:** {a}")

    if st.button("💬 Go to Chatbot"):
        st.switch_page("pages/1_chatbot.py")

    st.markdown("---")
    if st.button("📊 Go to Dashboard"):
        st.switch_page("pages/2_Dashboard.py")

else:
    st.info("📎 Please upload a CSV file to get started.")

