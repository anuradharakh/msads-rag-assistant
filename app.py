import streamlit as st

from src.generation.rag_chain import RAGChain


st.set_page_config(page_title="MSADS RAG Assistant", layout="wide")

st.title("MSADS RAG Assistant")
st.caption("Ask questions about the UChicago MS in Applied Data Science program")

# Initialize RAG
@st.cache_resource
def get_rag():
    return RAGChain()

rag = get_rag()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.answer(user_input)

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

             # Copy option
            #with st.expander("Copy"):
            #    st.code(answer, language="markdown")

            # Sources
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(
                            f"- **{src['section_title']}** ({src['content_type']})  \n"
                            f"{src['url']}"
                        )

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )