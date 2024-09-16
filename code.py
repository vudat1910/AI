import streamlit as st
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import TogetherEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# Tải dữ liệu từ tệp CSV
memory = MemorySaver()
loader = CSVLoader(
    file_path='C:/Users/dat.vuphat/Downloads/mobi.csv', 
    encoding='utf-8'
)
data = loader.load()

# Phân chia văn bản
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    add_start_index=True
)
all_splits = text_splitter.split_documents(data)

# Nhập API key
os.environ["TOGETHER_API_KEY"] = 'badc4b39ad1c39ffd1ea5b0522d12ad4e9593ede6b66b764b0b514f5e0cb2330'
os.environ["LANGSMITH_API_KEY"] = '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'

# Tạo embeddings
embeddings = TogetherEmbeddings(model='togethercomputer/m2-bert-80M-8k-retrieval')

# Tạo vectorstore
vectorstore = InMemoryVectorStore.from_documents(
    data, 
    embedding=embeddings
)
retriever = vectorstore.as_retriever(
    search_type='similarity', 
    search_kwargs={"k": 1000}
)

# Khởi tạo LLM
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
)


### Build retriever tool ###
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool]


agent_executor = create_react_agent(llm, tools, checkpointer=memory)
### Contextualize question ###
contextualize_q_system_prompt = (
    "Dựa vào lịch sử trò chuyện và câu hỏi mới nhất của người dùng "
    "có thể tham chiếu ngữ cảnh trong lịch sử, hãy tạo thành một câu hỏi "
    "độc lập bằng tiếng Việt mà có thể hiểu được mà không cần đến lịch sử trò chuyện."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi. "
    "Sử dụng các đoạn văn bản sau đây để trả lời câu hỏi bằng tiếng Việt. "
    "Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết. "
    "Trả lời ngắn gọn trong tối đa ba câu."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    question_answer_chain
)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
    
# Tạo chatbox
st.title('CHATBOT')
session_id = st.session_state.get('session_id', 'default')
message_history = get_session_history(session_id)

# Hiển thị lịch sử chat
for message in message_history.messages:
    if message.role == "human":
        st.chat_message("user").write(message.content)
    else:
        st.chat_message("assistant").write(message.content)

# Nhận câu hỏi từ người dùng
if question := st.chat_input("Hỏi gì đó..."):
    # Ghi lại câu hỏi của người dùng
    message_history.add_message(question)
    st.chat_message("user").write(question)

    # Gọi RAG chain để trả lời
    response = conversational_rag_chain.invoke(
        {"input": question},
        {"configurable": {"session_id": session_id }}  
    )

    # Ghi lại câu trả lời của chatbot
    message_history.add_message(response["answer"])
    st.chat_message("assistant").write(response["answer"])

