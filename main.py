import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model='gpt-6-luna')
vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer with direct quotes from the context to ground your answer:
"""
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval without LCEL, with limitations
    """

    # retrieve relevant docs from vector store
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents")
    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL.
    Returns a chain that can be invoked with {"question": "..."}
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs 
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == '__main__':
    print("Retrieving...")

    query = "what is Pinecone in machine learning?"
    print(f"Query: {query}")

    # Raw invocation without RAG
    print("Raw invocation without RAG:")
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(result_raw.content)

    print("\n\n" + "=" * 70)  # Separator for clarity
    print("Implementation without LCEL:")
    result_without_lcel = retrieval_chain_without_lcel(query)
    print(result_without_lcel)

    print("\n\n" + "=" * 70)  # Separator for clarity
    print("Implementation with LCEL:")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print(result_with_lcel)