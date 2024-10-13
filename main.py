#Finalized
import os
import boto3
import argparse
import streamlit as st
from langchain_aws import ChatBedrock
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import EnsembleRetriever

def format_Tdocs(docs, k = 10):
    return "\n\n".join("Source Of This Document: "+doc[0].metadata['source']+"\nDcoument: "+doc[0].page_content for doc in docs[0:k+1])
def createQuestionsList(text):
    return text.split("\n")
def reciprocal_rank_fusion(results, k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

def Setup(embedmodel, compressionmodel_type = "llm"):
    print('hello, setup starts')
    
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    bedrock=boto3.client(service_name="bedrock-runtime", region_name='ap-south-1')
    llm = ChatBedrock(model_id="meta.llama3-8b-instruct-v1:0",client=bedrock, model_kwargs={"temperature": 0.1, "top_p": 0.9})
    
    model_name = embedmodel
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, show_progress=True, cache_folder="./Embeddings"
    )
    print("Embeddings downloaded")
    
    FAISS_PATH = "./FAISS_data"
    vectorstore = FAISS.load_local(FAISS_PATH, hf_embeddings, allow_dangerous_deserialization = True)
    print("Vector DB Setup")
    
    retriever = vectorstore.as_retriever()
    
    if(compressionmodel_type == 'embed'):
        embeddings_filter = EmbeddingsFilter(embeddings=hf_embeddings, similarity_threshold=0.5)
        compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
    else:
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
    ensemble_retriever = EnsembleRetriever(retrievers=[compression_retriever, retriever], weights=[0.5, 0.5])
    
    print("Retriever setup")
    print("Setup finished")
    return bedrock, llm, ensemble_retriever

def Get_Reponse(user_query, chat_history, bedrock, llm, ensemble_retriever):
    print("Here")
    template = """You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines. Original question: {question}

    For providing these answers, imagine that you're a lead developer in an elevator company and you should rewrite these questions in such a way that it works prefectly with the vector database that the company has.
    """
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    getMoreQuestions = ({"question": RunnablePassthrough()} 
                    | prompt_rag_fusion 
                    | ChatBedrock(model_id="mistral.mixtral-8x7b-instruct-v0:1", client=bedrock, model_kwargs={"temperature": 0.1, "top_p": 0.9})
                    | StrOutputParser()  
                    | createQuestionsList)
    
    retrieval_chain_rag_fusion = getMoreQuestions | ensemble_retriever.map() | reciprocal_rank_fusion
    
    template = """Imagine you are a developer and give me answer to the question based on the context alone.
    Provide the sources of the documentation as well. (documentation url is mentioned is sent to you as well)
    And in the case of providing the pictures, provide the url of the photo. (photo url is mentioned within the documentation)
    
    NOTE: Do not hallucinate. Answer based only on the documentation provided. If something isn't explicitly mentioned within the documentation, do not assume anything unless you have a strong reason to. 
    If incase of you are assuming, mention that fact along with your reason to assume something within the answer. In the case of assuming, draw conclusions based only from the documentation and when doing so, explain your train of thought to conclude as such.

    question : {question}

    documentation: {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion | format_Tdocs, 
         "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain.invoke(user_query)

if __name__ == '__main__':
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if 'state_setup' not in st.session_state:
        parser = argparse.ArgumentParser()
        parser.add_argument("--embedmodel", type=str, default="BAAI/bge-small-en", help="SD for SupportingDoc; RD for RawDocs")
        parser.add_argument("--compressionmodel_type", type=str, default="llm", help="SD for SupportingDoc; RD for RawDocs")
        args = parser.parse_args()
        bedrock, llm, ensemble_retriever = Setup(embedmodel=args.embedmodel, compressionmodel_type=args.compressionmodel_type)
        st.session_state.state_setup = [bedrock, llm, ensemble_retriever]
    if 'stop' not in st.session_state:
        st.session_state['stop'] = False

    st.set_page_config(page_title="RAGSync: Chat")
    st.title('RAGSync Bot')
    
    if st.button('Stop the bot?'):
        st.session_state['stop'] = True
    if st.session_state['stop']:
        st.write('Stopping the app...')
        st.stop()    

    for message in st.session_state.chat_history: 
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message('YBot'):
                st.markdown(message.content)

    user_query = st.chat_input("Your query")
    if user_query is not None and user_query != '':
        st.session_state.chat_history.append(HumanMessage(user_query))
    
        with st.chat_message('Human'):
            st.markdown(user_query)
        
        with st.chat_message("YBot"):
            ai_message = Get_Reponse(user_query, st.session_state.chat_history, st.session_state.state_setup[0],  st.session_state.state_setup[1], st.session_state.state_setup[2])
            st.markdown(ai_message)
            st.session_state.chat_history.append(AIMessage(ai_message))