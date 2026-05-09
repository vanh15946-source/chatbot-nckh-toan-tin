import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ==========================================
# 1. TẢI TÀI NGUYÊN
# ==========================================
@st.cache_resource
def load_system():
    # Kiểm tra key tồn tại trước khi khởi động
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("❌ Không tìm thấy GROQ_API_KEY trong file .env!")
        st.stop()

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,  # ✅ Đọc từ biến, không hardcode
        temperature=0.3
    )

    prompt_template = """
    Bạn là một trợ lý AI học tập xuất sắc của sinh viên. Hãy trả lời câu hỏi DỰA TRÊN các đoạn tài liệu được cung cấp dưới đây.

    YÊU CẦU QUAN TRỌNG:
    - CHỈ trả lời bằng Tiếng Việt.
    - NẾU thông tin không có trong tài liệu, hãy nói rõ: "Tài liệu môn học hiện tại không có thông tin về vấn đề này".
    - Bỏ qua các lỗi chính tả trong tài liệu nếu bạn có thể đoán được nghĩa.

    TÀI LIỆU MÔN HỌC:
    {context}

    CÂU HỎI CỦA SINH VIÊN: {question}

    CÂU TRẢ LỜI CỦA BẠN:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return db, llm, prompt


db, llm, prompt = load_system()

# ==========================================
# 2. GIAO DIỆN STREAMLIT (giữ nguyên)
# ==========================================
st.set_page_config(page_title="Chatbot Học Tập AI", page_icon="🤖")
st.title("🤖 Chatbot Hỏi Đáp Tài Liệu Môn Học")
st.markdown("Hệ thống hỗ trợ giải đáp thắc mắc dựa trên tài liệu giáo trình.")

st.sidebar.header("⚙️ Cài đặt hệ thống")
mode = st.sidebar.radio(
    "Chọn chế độ hoạt động:",
    ("Sử dụng RAG (Tìm trong tài liệu)", "Không dùng RAG (Hỏi LLM trực tiếp)")
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Nhập câu hỏi của bạn về môn học..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Đang suy nghĩ..."):
            try:
                if mode == "Sử dụng RAG (Tìm trong tài liệu)":
                    results = db.similarity_search(user_input, k=3)
                    context_text = "\n\n".join([doc.page_content for doc in results])
                    final_prompt = prompt.format(context=context_text, question=user_input)
                    with st.expander("📄 Xem các đoạn tài liệu tìm được"):
                        st.text(context_text)
                    full_response = llm.invoke(final_prompt).content
                else:
                    full_response = llm.invoke(user_input).content

            except Exception as e:
                full_response = f"⚠️ Đã xảy ra lỗi: {str(e)}"

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
