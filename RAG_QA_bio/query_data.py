# Importing Libraries

from langchain.chains import RetrievalQA, ConversationalRetrievalChain,LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from create_db import generate_data_store

# Create the vector database (Chroma) for storing embeddings 
vectordb = generate_data_store()

 # Configure retriever to find top 5 similar chunks
retriever = vectordb.as_retriever(search_type = "similarity", search_kwargs={"k": 5})

def main(query):
    llm = HuggingFaceHub(
        repo_id = "mistralai/Mistral-7B-Instruct-v0.1",  # Load the Mistral LLM
        model_kwargs={"temperature":0.2, "max_length":512}, # Adjust generation parameters
        huggingfacehub_api_token = "HF_TOKEN"  # Hugging Face API token
    )

    print('In Search Mode')
    
     # Create a QA prompt for retrieval and answering
    rqa_prompt_template = """Use the following pieces of context to answer the questions at the end.
                        Answer only from the context. If you don't know the answer, say you do not know.
                    {context}
                    Explain in detail.
                    Question: {question}
                    """
    RQA_PROMPT = PromptTemplate(
        template = rqa_prompt_template, input_variables = ["context","question"]
    )
    rqa_chain_type_kwargs = {"prompt": RQA_PROMPT}

    qa = RetrievalQA.from_chain_type(llm,
                                     chain_type="stuff", # 'Stuffing' method for embedding context into the LLM prompt
                                     retriever = retriever, # Use the Chroma-based retriever
                                     chain_type_kwargs=rqa_chain_type_kwargs,
                                     return_source_documents = True, # Retrieve source documents for reference
                                     verbose = False)
                                     
    # Run the question-answering chain
    result = qa({"query": query}) 
    return result

if __name__ == "__main__":
    query = input("Ask a question: ")
    print("looking for results")
    result = main(query)
    print(result['result'])
