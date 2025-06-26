from dataclasses import dataclass, field
import datetime
from typing import List, Optional
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from bson import ObjectId
import html
from agent import Agent
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from vectorstores import ElasticsearchStore
from langchain_elasticsearch import DenseVectorStrategy
import os
from streamlit_cookies_manager import EncryptedCookieManager

# Load environment variables from .env file
dotenv.load_dotenv("./.env")

cookies = EncryptedCookieManager(
    password=os.environ.get("COOKIE_PASSWORD"),
    prefix="bookchat",  # e.g. cookie will be named "myapp-user_id"
)

# Streamlit requires you to stop() until cookies are loaded
if not cookies.ready():
    st.stop()

if "user_id" not in cookies:
    # generate a new random UUID4
    new_id = str(ObjectId())
    cookies["user_id"] = new_id
    # write it to the browser
    cookies.save()

st.session_state["user_id"] = cookies["user_id"]


@dataclass
class DocumentChunk:
    _id: str
    text: str
    embedding: list[float]
    contributor: str
    date: str


@dataclass
class ChatMessage:
    type: str
    content: str
    timestamp: datetime.datetime
    citations: Optional[List[str]] = field(default_factory=list)


def time_now():
    return datetime.datetime.now(datetime.timezone.utc)


def get_document_chunk_by_ids(ids: List[str]) -> List[DocumentChunk]:
    # cursor = document_chunk_collection.find({"_id": {"$in": ids}})
    # return [DocumentChunk(**chunk) for chunk in cursor]
    # PLACEHOLDER

    return [DocumentChunk(_id=_id, text="", embedding=[], contributor="", date="")
            for _id in ids]  # Placeholder for actual implementation


def get_chat_messages(chat_id) -> List[ChatMessage]:
    # chat = chat_collection.find_one({"_id": ObjectId(chat_id)})
    # if chat and "messages" in chat:
    #     messages = [ChatMessage(**msg) for msg in chat["messages"]]
    #     return sorted(messages, key=lambda m: m.timestamp)
    # PLACEHOLDER
    return []


def create_chat(user_id):
    # now = time_now()
    # now_iso = now.isoformat()
    # title = f"Chat-{now_iso}"
    # new_chat = Chat(
    #     userId=ObjectId(user_id),
    #     createdAt=now,
    #     updatedAt=now,
    #     title=title,
    #     messages=[]
    # )
    # result = chat_collection.insert_one(asdict(new_chat))
    # return str(result.inserted_id)
    # PLACEHOLDER
    return "placeholder_chat_id"  # Placeholder for actual implementation


def get_user_chats(user_id):
    # chats = chat_collection.find(
    #     {"userId": ObjectId(user_id)}).sort(
    #     "updatedAt", -1)
    # return [Chat(**chat) for chat in chats]
    # PLACEHOLDER
    return []  # Placeholder for actual implementation


def set_chat_title(chat_id, title):
    # PLACEHOLDER
    pass
    # chat_collection.update_one(
    #     {"_id": ObjectId(chat_id)},
    #     {"$set": {"title": title}}
    # )


def delete_chat(chat_id):
    # PLACEHOLDER
    pass
    # chat_collection.delete_one({"_id": ObjectId(chat_id)})


def add_chat_message(chat_id, type, content, citations: List[str] = []):
    now = time_now()
    # PLACEHOLDER
    new_message = ChatMessage(
        type=type,
        content=content,
        timestamp=now,
        citations=citations if citations else []
    )
    # chat_collection.update_one(
    #     {"_id": ObjectId(chat_id)},
    #     {
    #         "$push": {"messages": asdict(new_message)},
    #         "$set": {"updatedAt": now}
    #     }
    # )


@st.cache_resource
def create_embedder():
    hf = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
    )
    return hf


@st.cache_resource
def create_vectorstore():
    ES_CLOUD_ID = os.environ.get("ES_CLOUD_ID")
    ES_USERNAME = os.environ.get("ES_USERNAME")
    ES_PASSWORD = os.environ.get("ES_PASSWORD")

    # Initialize ElasticsearchStore using the pre-loaded environment variables
    db = ElasticsearchStore(
        es_cloud_id=ES_CLOUD_ID,
        index_name="vectorsearch_sharded",
        embedding=create_embedder(),
        es_user=ES_USERNAME,
        es_password=ES_PASSWORD,
        strategy=DenseVectorStrategy(),  # strategy for dense vector search
    )
    return db


@st.dialog("Prompt Suggestions")
def prompt_help_dialog():
    # Suggested prompts
    suggested_prompts = [
        "Are there medical conditions caused by Maple Syrup?",
    ]

    st.write("Here are some suggested prompts to get you started:")

    for prompt in suggested_prompts:
        st.write(f"- {prompt}")


def chat_component():
    # ---------------
    # MAIN CONTENT
    # ---------------

    messages_container = st.container()

    @st.dialog("Citations")
    def citations_dialog(item):
        citations: List[DocumentChunk] = get_document_chunk_by_ids(
            item.citations)
        if len(citations) > 0:
            for chunk in citations:
                with st.expander(str(chunk._id)):
                    st.markdown(f"{chunk.text}")
                    st.markdown(
                        f"---\n**Contributor**: {chunk.contributor} | **Date**: {chunk.date}")
        else:
            st.markdown("No citations found.")

    if st.session_state.get("messages") is None:
        st.session_state["messages"] = []

    has_chat_messages = False
    if "selected_chat_id" in st.session_state:
        chat_id = st.session_state["selected_chat_id"]

        if "reload_chat" in st.session_state:
            st.session_state["messages"] = get_chat_messages(chat_id)
            st.session_state["reload_chat"] = False

        if len(st.session_state["messages"]) > 0:
            has_chat_messages = True
            for message in st.session_state["messages"]:
                messages_container.chat_message(
                    message.type).write(
                    message.content)
                if message.type == "ai":
                    with messages_container:
                        if st.button(
                                label="Citations",
                                key=f"citations_{message.timestamp}",
                                help="Show citations",
                                type="secondary"):
                            citations_dialog(message)

    # selected_projects = st.session_state["selected_projects"]
    # if len(selected_projects) == 0:
    #     filter = None
    # else:
    #     filter_list = [
    #         p.project for p in selected_projects]
    #     filter = {
    #         "project_name": {
    #             "$in": filter_list}}

    def do_response(human_prompt):
        rerun = False
        messages: List[BaseMessage] = []
        if not has_chat_messages:
            new_chat_id = create_chat(st.session_state["user_id"])
            st.session_state["selected_chat_id"] = new_chat_id
            rerun = True
        else:
            for message in st.session_state["messages"]:
                if message.type == "human":
                    messages.append(HumanMessage(message.content))
                elif message.type == "ai":
                    messages.append(AIMessage(message.content))

        # Add current user message to the chat history
        st.session_state["messages"].append(
            ChatMessage(
                type="human",
                content=human_prompt,
                timestamp=time_now()))
        messages = messages + [HumanMessage(human_prompt)]
        messages_container.chat_message("human").write(human_prompt)
        # Add user message to history
        add_chat_message(
            st.session_state["selected_chat_id"],
            "human", human_prompt)

        agent = Agent(
            vectorstore=create_vectorstore(),
            embedder=create_embedder()
        )

        response = agent.invoke(
            messages,
            metadata_filter=filter)

        # Create a container for streaming response
        response_container = messages_container.chat_message("ai")
        # Use a placeholder for streaming text
        stream_placeholder = response_container.empty()
        # Accumulate streamed text
        full_response = ""
        # Stream the response
        for chunk in response:
            full_response += chunk
            full_response = html.unescape(full_response).replace("\\n", "\n")
            stream_placeholder.markdown(full_response + "▌")

        stream_placeholder.markdown(full_response)

        obj_id_citations = [str(x) for x in agent.last_message_citations]

        # Add AI message to history
        new_ai_chat_message = ChatMessage(
            type="ai",
            content=full_response,
            citations=obj_id_citations,
            timestamp=time_now())
        st.session_state["messages"].append(new_ai_chat_message)

        with messages_container:
            if st.button(
                    label="Citations",
                    key=f"citations_{new_ai_chat_message.timestamp}",
                    help="Show citations",
                    type="secondary"):
                citations_dialog(new_ai_chat_message)

        add_chat_message(
            st.session_state["selected_chat_id"],
            "ai", full_response, obj_id_citations)
        if rerun:
            st.rerun()

    if user_query := st.chat_input(
            key="user_query", placeholder="Type your prompt here..."):
        do_response(user_query)

    if st.button('', icon=":material/help:",
                 type="tertiary", help="Prompt suggestions"):
        prompt_help_dialog()

    # if st.button('', icon=":material/refresh:",
    #              type="tertiary", help="Refresh chat"):
    #     st.rerun()


def main():
    with st.sidebar:
        # Create two columns: one for heading, one for logout button
        col_heading, col_logout = st.columns(
            [4, 1], vertical_alignment="bottom")  # Adjust ratio as desired

        # Left column: App title
        with col_heading:
            st.title("Chat with Books")
            st.subheader("Wellcome Collection")

        # Right column: Logout button
        with col_logout:
            if st.button('', icon=":material/logout:",
                         type="tertiary", help="Logout"):
                # components.restrict_access.logout()
                pass

        if st.session_state.get("role") == "admin":
            st.page_link("pages/admin.py", label="Admin Interface")

        st.markdown("---")

        chat_col = st.columns([4, 1])
        chat_col[0].markdown("## Chats")
        if chat_col[1].button("", icon=":material/add:",
                              help="Start new chat"):
            try:
                del st.session_state["selected_chat_id"]
            except KeyError:
                pass
            st.rerun()

        chats = get_user_chats(st.session_state["user_id"])

        for chat in chats:
            session_chat_id = st.session_state.get("selected_chat_id")
            is_selected = str(chat._id) == session_chat_id
            chat_cols = st.columns([4, 1])
            if chat_cols[0].button(
                    chat.title,
                    key=str(chat._id), type="primary"
                    if is_selected else
                    "tertiary"):
                st.session_state["selected_chat_id"] = str(chat._id)
                st.session_state["reload_chat"] = True
                st.rerun()
            with chat_cols[1].popover(label="", icon=":material/more_vert:"):
                new_title = st.text_input("Rename Chat", value=chat.title)
                if new_title != chat.title:
                    if st.button("Save", icon=":material/save:",
                                 key=f"save_{chat._id}"):
                        set_chat_title(chat._id, new_title)
                        st.rerun()
                if st.button("Delete", icon=":material/delete:",
                             key=f"del_{chat._id}"):
                    delete_chat(chat._id)
                    st.session_state["selected_chat_id"] = None
                    st.session_state["reload_chat"] = False
                    st.rerun()

    with st.container(height=0):
        chat_component()


if __name__ == "__main__":
    main()
