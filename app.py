from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate


# ==========================================
# 1. TẢI TÀI NGUYÊN (CACHE ĐỂ KHÔNG BỊ LOAD LẠI NHIỀU LẦN)
# ==========================================
@st.cache_resource
def load_system():
    # Tải mô hình Embedding và Database
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Trở lại bản xịn ban đầu
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0.3
)

    # Tạo Prompt
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
# 2. XÂY DỰNG GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Chatbot Học Tập AI", page_icon="🤖")

st.title("🤖 Chatbot Hỏi Đáp Tài Liệu Môn Học")
st.markdown("Hệ thống hỗ trợ giải đáp thắc mắc dựa trên tài liệu giáo trình.")

# --- SIDEBAR: Lựa chọn chế độ ---
st.sidebar.header("⚙️ Cài đặt hệ thống")
mode = st.sidebar.radio(
    "Chọn chế độ hoạt động:",
    ("Sử dụng RAG (Tìm trong tài liệu)", "Không dùng RAG (Hỏi LLM trực tiếp)")
)

# --- QUẢN LÝ LỊCH SỬ CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- XỬ LÝ KHUNG CHAT ---
if user_input := st.chat_input("Nhập câu hỏi của bạn về môn học..."):
    # 1. Hiển thị câu hỏi của người dùng
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Xử lý câu trả lời của Bot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Đang suy nghĩ..."):
            if mode == "Sử dụng RAG (Tìm trong tài liệu)":
                # Kịch bản 1: Có RAG
                results = db.similarity_search(user_input, k=3)
                context_text = "\n\n".join([doc.page_content for doc in results])
                final_prompt = prompt.format(context=context_text, question=user_input)

                # Bật expander để show tài liệu tìm được (điểm cộng lớn khi báo cáo NCKH)
                with st.expander("📄 Xem các đoạn tài liệu tìm được"):
                    st.text(context_text)

                full_response = llm.invoke(final_prompt)
            else:
                # Kịch bản 2: Không RAG (Hỏi thẳng Qwen)
                full_response = llm.invoke(user_input)

        message_placeholder.markdown(full_response)

    # Lưu câu trả lời vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": full_response})
