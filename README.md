# GCP and Qdrant Integration with Streamlit

This project demonstrates how to integrate Google Cloud Platform (GCP), Qdrant, and Streamlit to create a sophisticated AI-driven data exploration and visualization tool. The core functionalities include generating SQL queries, validating them, and creating visualizations based on the data retrieved from a Google Cloud SQL database.

## Prerequisites

Before running the application, ensure you have the following:

1. **Google Cloud Project ID**: You need a Google Cloud project with appropriate permissions.
2. **Qdrant Host URL and API Key**: Qdrant setup with host URL and API key.
3. **Google Cloud SQL Instance**: An instance with a PostgreSQL database.
4. **Vertex AI Model Details**: Information about your Vertex AI model, including project, location, model name, and tuned model ID.

## Installation

1. **Clone the Repository**:

   ```sh
    git clone https://github.com/yourusername/GenAI-DB-Visualization.git
    cd GenAI-DB-Visualization
   ```

2. **Install Dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   You can set these variables in your environment or a `.env` file.
   ```sh
   export QDRANT_HOST="your_qdrant_host_url"
   export QDRANT_API_KEY="your_qdrant_api_key"
   export INSTANCE_CONNECTION_NAME="your_gcp_cloud_sql_project_instance_id"
   export DB_USER="your_gcp_cloud_sql_db_username"
   export DB_PASS="your_gcp_cloud_sql_db_password"
   export DB_NAME="your_gcp_cloud_sql_db_name"
   ```

## Configuration

Modify the configuration parameters for your Vertex AI model in the `setup_vanna` function:

```python
config = {
    "project": "your_vertexai_project_id",
    "location": "your_vertexai_project_location",
    "model_name": "your_fine-tuned_model_name",
    "tuned_model_id": "your_fine-tuned_model_id"
}
```

## Google Cloud Setup

Run the following Google Cloud SDK commands in your terminal to authenticate, set up your project, and enable necessary services:

1. **Authenticate with Google Cloud**:

   ```sh
   gcloud auth login
   ```

2. **Set Your Project**:

   ```sh
   gcloud config set project <project-id>
   ```

3. **Add IAM Policy Binding**:

   ```sh
   gcloud projects add-iam-policy-binding {project_id} \
     --member={current_user} \
     --role="roles/cloudsql.client"
   ```

4. **Enable SQL Admin API**:
   ```sh
   gcloud services enable sqladmin.googleapis.com
   ```

## Running the Application

To run the Streamlit application, use the following command:

```sh
streamlit run app.py
```

## Classes and Functions

### ChatBison Class

The `ChatBison` class inherits from `VannaBase` and initializes a Vertex AI Text Generation Model. It provides methods to format messages and submit prompts to the model.

### MyVanna Class

`MyVanna` class combines functionalities from `Qdrant_VectorStore` and `ChatBison`, initializing both vector storage and text generation capabilities.

### Streamlit Caching Functions

The application uses Streamlit caching to optimize performance for repeated operations:

- **setup_vanna**: Initializes `MyVanna` with Qdrant client and database connection.
- **generate_questions_cached**: Generates sample questions.
- **generate_sql_cached**: Generates SQL queries based on questions.
- **is_sql_valid_cached**: Validates generated SQL queries.
- **run_sql_cached**: Executes SQL queries and returns the result as a DataFrame.
- **should_generate_chart_cached**: Determines if a chart should be generated based on the query.
- **generate_plotly_code_cached**: Generates Plotly code for visualizations.
- **generate_plot_cached**: Executes the Plotly code to create visualizations.
- **generate_followup_cached**: Generates follow-up questions based on previous interactions.
- **generate_summary_cached**: Generates summaries of data.

## Troubleshooting

- Ensure all environment variables are correctly set.
- Verify your Google Cloud SQL instance and database credentials.
- Check the configurations for Vertex AI and Qdrant.

For further assistance, refer to the respective documentation of GCP, Qdrant, and Streamlit.

## License

This project is licensed under the MIT License.
