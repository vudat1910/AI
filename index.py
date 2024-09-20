import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
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
import numpy as np
import psycopg2
# Tải dữ liệu từ tệp CSV
memory = MemorySaver()
loader_csv = CSVLoader(
    file_path='C:/Users/dat.vuphat/Downloads/mobi.csv', 
    encoding='utf-8'
)
urls = [
    'https://mobigames.vn/',
    'https://mobigame.com.vn/home',
    'https://mobiedu.vn/',
    'https://www.mobifone.vn/dich-vu-di-dong/dich-vu/cliptv-ClipTV'
]
loader_webserver = WebBaseLoader (
    web_path=urls
)
loader_pdf = PyPDFLoader(
    file_path='C:/Users/dat.vuphat/Downloads/mobicloud.pdf'
)
data_csv = loader_csv.load()
data_web = loader_webserver.load()
data_pdf = loader_pdf.load()
data = data_csv + data_pdf + data_web
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

vectorstore.dump('vectorstore_data.pkl')
loaded_vectorstore = InMemoryVectorStore.load('vectorstore_data.pkl', embedding=embeddings)

# Kết nối PostgreSQL
conn = psycopg2.connect(
    dbname="ai_llm", user="postgres", password="Chelsea@19102002", host="localhost", port="5432"
)
cursor = conn.cursor()

# Đảm bảo bảng đã được tạo trước đó, hoặc tạo bảng nếu cần
cursor.execute("""
CREATE TABLE IF NOT EXISTS vector_data (
    document_id TEXT PRIMARY KEY,
    vector FLOAT8[]
)
""")
# Duyệt qua tất cả các vectors và chèn vào PostgreSQL (Sử dụng UPSERT)
for idx, doc in enumerate(all_splits):
    vector = embeddings.embed_documents([doc.page_content])[0]  # Lấy vector từ nội dung tài liệu
    document_id = doc.metadata.get('document_id', f'doc_{idx}')  # Sử dụng ID từ metadata hoặc tạo ID dự phòng
    
    # Chèn vector vào bảng (Sử dụng UPSERT)
    cursor.execute(
        """
        INSERT INTO vector_data (document_id, vector)
        VALUES (%s, %s)
        ON CONFLICT (document_id) DO UPDATE
        SET vector = EXCLUDED.vector;
        """,
        (document_id, np.array(vector).tolist())  # Chuyển đổi numpy array thành list
    )
# Lưu thay đổi và đóng kết nối
conn.commit()
cursor.close()
conn.close()
retriever = vectorstore.as_retriever(
    search_type='similarity', 
    search_kwargs={"k": 10000000000000}
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
    question_answer_chain)

chat_history = []

question = "How many gói cước?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)

second_question = "gói cước nào rẻ nhất?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})


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


if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = f"session_{np.random.randint(10000)}"

# Tạo giao diện chatbot
st.title('Chatbot')

# Hiển thị lịch sử trò chuyện
for message in st.session_state['chat_history']:
    if message['role'] == 'human':
        st.chat_message("user").write(message['content'])
    else:
        st.chat_message("assistant").write(message['content'])

# Nhận câu hỏi từ người dùng
user_input = st.chat_input("Bạn muốn hỏi gì?")

if user_input:
    # Lưu tin nhắn người dùng
    st.session_state['chat_history'].append({"role": "human", "content": user_input})
    st.chat_message("user").write(user_input)

    # Gọi hàm trả lời từ mô hình với session_id
    ai_response = conversational_rag_chain.invoke(
        {"input": user_input, "chat_history": st.session_state['chat_history']},
        {"configurable": {"session_id": st.session_state['session_id']}}  
    )
    chatbot_response = ai_response["answer"]  # Lấy câu trả lời từ mô hình

    # Lưu phản hồi của chatbot
    st.session_state['chat_history'].append({"role": "assistant", "content": chatbot_response})
    st.chat_message("assistant").write(chatbot_response)
