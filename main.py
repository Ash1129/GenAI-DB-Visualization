### GCP, qdrant

import streamlit as st
import os
import qdrant_client
import pandas as pd
from google.cloud.sql.connector import Connector
import sqlalchemy
from vanna.qdrant import Qdrant_VectorStore
from vertexai.language_models import TextGenerationModel
from vertexai import init as vertexai_init
from vanna.base import VannaBase 

project_id # =  "Enter your GCP_Project_ID"

# Details for QDrant
os.environ['QDRANT_HOST']  # = "Enter your QDrant Host URL"
os.environ['QDRANT_API_KEY'] # = "Enter your QDrant api_key"
 
# Details for GCP connector
INSTANCE_CONNECTION_NAME # = "Enter your GCP_Cloud_SQL_Project_Instance_ID"
print(f"Your instance connection name is: {INSTANCE_CONNECTION_NAME}")
DB_USER  # = "Enter your GCP_Cloud_SQL_DB_UserName"
DB_PASS  # = "Enter your GCP_Cloud_SQL_DB_Pass"
DB_NAME  # = "Enter your GCP_Cloud_SQL_DB_Name"

class ChatBison(VannaBase):
    def __init__(self, config=None):
        if config is None:
            raise ValueError("For ChatBison, config must be provided with project, location, and model details.")
        
        self.project = config.get("project")
        self.location = config.get("location")
        self.model_name = config.get("model_name")
        self.tuned_model_id = config.get("tuned_model_id")
        
        if not self.project or not self.location or not self.model_name or not self.tuned_model_id:
            raise ValueError("Missing necessary configuration for ChatBison.")
        
        vertexai_init(project=self.project, location=self.location)
        self.model = TextGenerationModel.from_pretrained(self.model_name).get_tuned_model(self.tuned_model_id)

        self.parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0.9,
            "top_p": 1
        }
    
    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        if not prompt.strip():  # Check if prompt is empty or contains only whitespace
            return "Please provide a valid prompt."
        
        response = self.model.predict(prompt, **self.parameters)
        return response.text

class MyVanna(Qdrant_VectorStore, ChatBison):
    def __init__(self, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        ChatBison.__init__(self, config=config)


@st.cache_resource(ttl=3600)
def setup_vanna():
    client = qdrant_client.QdrantClient(
        os.getenv('QDRANT_HOST'),
        api_key=os.getenv('QDRANT_API_KEY')
    )  
    
    config = {
        "project": # Enter your vertexai project id
        "location": # Enter your vertexai project location
        "model_name": # Enter your fine-tuned model name
        "tuned_model_id": # Enter your fine-tuned model id
    }

    vn = MyVanna(config={'client': client, **config})

    # Connect to your database

    connector = Connector()

    def getconn():
        conn = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME
        )
        return conn

    # Create the SQLAlchemy engine using the connection from the connector
    engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    def run_sql(sql: str) -> pd.DataFrame:
        df = pd.read_sql_query(sql, engine)
        return df

    vn.run_sql = run_sql
    vn.run_sql_is_set = True
    
    # The information schema query may need some tweaking depending on your database. This is a good starting point.
    df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
    
    # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
    plan = vn.get_training_plan_generic(df_information_schema)
    
    # If you like the plan, then uncomment this and run it to train
    # vn.train(plan=plan)

    return vn


@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    return vn.generate_sql(question=question, allow_llm_to_see_data=True)

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    vn = setup_vanna()
    return vn.is_sql_valid(sql=sql)

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    vn = setup_vanna()
    return vn.run_sql(sql=sql)

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_vanna()
    return vn.should_generate_chart(df=df)

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_vanna()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_vanna()
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    vn = setup_vanna()
    return vn.generate_summary(question=question, df=df)