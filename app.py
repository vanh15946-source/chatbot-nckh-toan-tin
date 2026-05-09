import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

# ==========================================
# 1. TẢI TÀI NGUYÊN
# ==========================================
@st.cache_resource
def load_system():
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
        api_key=groq_api_key,
        temperature=0.3
    )
    return db, llm


db, llm = load_system()


# ==========================================
# 2. HÀM XÂY DỰNG MESSAGES CÓ LỊCH SỬ
# ==========================================
def build_messages(chat_history: list, user_input: str, context: str = None):
    """
    Gộp system prompt + lịch sử hội thoại + câu hỏi mới thành 1 danh sách messages.
    """
    if context:
        system_content = f"""Bạn là một trợ lý AI học tập xuất sắc của sinh viên.
Hãy trả lời câu hỏi DỰA TRÊN tài liệu môn học bên dưới VÀ lịch sử hội thoại trước đó.

YÊU CẦU:
- CHỈ trả lời bằng Tiếng Việt.
- Nhớ các thông tin đã trao đổi trong cuộc hội thoại để trả lời liền mạch.
- NẾU thông tin không có trong tài liệu, hãy nói: "Tài liệu môn học hiện tại không có thông tin về vấn đề này".

TÀI LIỆU MÔN HỌC:
{context}"""
    else:
        system_content = """Bạn là một trợ lý AI học tập xuất sắc của sinh viên.
Hãy trả lời câu hỏi dựa trên kiến thức của bạn VÀ lịch sử hội thoại trước đó.

YÊU CẦU:
- CHỈ trả lời bằng Tiếng Việt.
- Nhớ các thông tin đã trao đổi để trả lời liền mạch, tự nhiên."""

    messages = [SystemMessage(content=system_content)]

    # Thêm lịch sử hội thoại (giới hạn 10 lượt gần nhất để tránh vượt context window)
    recent_history = chat_history[-10:]
    for msg in recent_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # Thêm câu hỏi hiện tại
    messages.append(HumanMessage(content=user_input))
    return messages


# ==========================================
# 3. GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Chatbot Học Tập AI", page_icon="🤖")
st.title("🤖 Chatbot Hỏi Đáp Tài Liệu Môn Học")
st.markdown("Hệ thống hỗ trợ giải đáp thắc mắc dựa trên tài liệu giáo trình.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Cài đặt hệ thống")
mode = st.sidebar.radio(
    "Chọn chế độ hoạt động:",
    ("Sử dụng RAG (Tìm trong tài liệu)", "Không dùng RAG (Hỏi LLM trực tiếp)")
)

# Nút xóa lịch sử chat
if st.sidebar.button("🗑️ Xóa lịch sử hội thoại"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.caption(f"💬 Số lượt hội thoại: {len(st.session_state.get('messages', [])) // 2}")

# --- QUẢN LÝ LỊCH SỬ CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- XỬ LÝ KHUNG CHAT ---
if user_input := st.chat_input("Nhập câu hỏi của bạn về môn học..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Đang suy nghĩ..."):
            try:
                # Lấy lịch sử TRƯỚC câu hỏi hiện tại (không gồm câu vừa append)
                history_so_far = st.session_state.messages[:-1]

                if mode == "Sử dụng RAG (Tìm trong tài liệu)":
                    results = db.similarity_search(user_input, k=3)
                    context_text = "\n\n".join([doc.page_content for doc in results])

                    with st.expander("📄 Xem các đoạn tài liệu tìm được"):
                        st.text(context_text)

                    # ✅ Truyền cả lịch sử + context tài liệu
                    messages = build_messages(history_so_far, user_input, context=context_text)
                else:
                    # ✅ Truyền lịch sử, không có context tài liệu
                    messages = build_messages(history_so_far, user_input, context=None)

                full_response = llm.invoke(messages).content

            except Exception as e:
                full_response = f"⚠️ Đã xảy ra lỗi: {str(e)}"

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
