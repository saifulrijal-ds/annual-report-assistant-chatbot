import os
import logging
from dotenv import load_dotenv
import nest_asyncio

from llama_parse import LlamaParse
from llama_parse.utils import Language
import joblib

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(messages)s",
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()
nest_asyncio.apply()

def load_or_parse_data():
    parsed_data_file = "./data/parsed_data.pkl"

    if os.path.exists(parsed_data_file):
        logging.info("Loading parsed data from file")
        parsed_data = joblib.load(parsed_data_file)
    else:
        logging.info("Parsing data from scratch")
        # Initialize the LlamaParse object
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            language=Language.INDONESIAN,
            result_type="markdown",
            max_timeout=5000,
            verbose=True
        )

        llama_parse_documents = parser.load_data("./data/IR BFIN 2023_IN.pdf")
        joblib.dump(llama_parse_documents, parsed_data_file)
        logging.info("Parsed data saved to file")
        parsed_data = llama_parse_documents

    return parsed_data

def create_vector_database():
    llama_parse_documents = load_or_parse_data()
    logging.info("Parsed data sample: %s", llama_parse_documents[0].text[:300])

    with open("./data/output.md", 'a') as f_out:
        for doc in llama_parse_documents:
            f_out.write(doc.text + "\n")
        logging.info("Parsed data written to output.md")

    markdown_path = "./data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path=markdown_path)
    documents = loader.load()
    logging.info("Loaded %d documents", len(documents))

    text_spliiter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_verlap = 100
    )
    docs = text_spliiter.split_documents(documents)

    logging.info("Generated %d document chunks", len(docs))

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024
    )

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="./chroma_data",
        collection_name="bfi_annual_report"
    )
    logging.info("Vector store created and saved to %s", vector_store.persist_directory)

if __name__ == "__main__":
    create_vector_database()
