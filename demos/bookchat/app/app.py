from dataclasses import dataclass, field
import datetime
from typing import List, Optional
from urllib import response
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
    work_id: str
    chunk_index: int
    url: str
    text: str
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
    docs = []

    if len(ids) > 0:
        es = get_vectorstore()._store.client

        response = es.mget(
            index="vectorsearch_sharded",
            body={
                "ids": ids
            }
        )

        for doc in response["docs"]:
            if doc.get("found", False):
                work_id, chunk_index = doc["_id"].split("_")
                url = f"https://wellcomecollection.org/works/{work_id}"
                text = doc["_source"]["text"]
                metadata = doc["_source"]["metadata"]
                docs.append(DocumentChunk(
                    _id=doc["_id"],
                    work_id=work_id,
                    chunk_index=int(chunk_index),
                    url=url,
                    text=text,
                    contributor=metadata.get("contributor", "Unknown"),
                    date=metadata.get("date", "Unknown")
                ))
    return docs


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
def get_embedder():
    hf = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
    )
    return hf


@st.cache_resource
def get_vectorstore():
    ES_CLOUD_ID = os.environ.get("ES_CLOUD_ID")
    ES_USERNAME = os.environ.get("ES_USERNAME")
    ES_PASSWORD = os.environ.get("ES_PASSWORD")

    # Initialize ElasticsearchStore using the pre-loaded environment variables
    db = ElasticsearchStore(
        es_cloud_id=ES_CLOUD_ID,
        index_name="vectorsearch_sharded",
        embedding=get_embedder(),
        es_user=ES_USERNAME,
        es_password=ES_PASSWORD,
        strategy=DenseVectorStrategy(),
        es_params={"verify_certs": False}  # strategy for dense vector search
    )
    return db


@st.dialog("Prompt Suggestions")
def prompt_help_dialog():
    # Suggested prompts
    suggested_prompts = [
        "Are there medical conditions caused by Maple Syrup?",
        "Famous doctors from the 19th century.",
        "Creatively explore the interrelationships between science and art.",
        "How was red wine used in medicine pre-1900s?",
        "How did amputation methods in the 1800s and 1900s differ?",
        "What are the most common medical conditions in the 1800s?",
        "How did people treat infections before antibiotics?",
        "What were the most common surgical procedures in the 19th century?",
        "How did the understanding of anatomy change in the 1800s?",
        "What was the role of women in medicine during the 19th century?",
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
                        f"---\n**Contributor**: {chunk.contributor} | **Date**: {chunk.date} | [Link]({chunk.url})")
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

    # Since project filtering is commented out, set filter to None
    filter = None

    if st.session_state.get("user_query_state") is None:
        st.session_state["user_query_state"] = ""

    def do_response(human_prompt):
        # --- Throttle: max 5 requests per hour per user ---
        now = time_now()
        if "request_timestamps" not in st.session_state:
            st.session_state["request_timestamps"] = []

        # Remove timestamps older than 1 hour
        one_hour_ago = now - datetime.timedelta(hours=1)
        st.session_state["request_timestamps"] = [ts
                                                  for ts in st.session_state["request_timestamps"]
                                                  if ts > one_hour_ago]
        if len(st.session_state["request_timestamps"]) >= 5:
            st.warning(
                "You have reached the maximum of 5 requests per hour. Please wait before making more requests.")
            return
        st.session_state["request_timestamps"].append(now)

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
                timestamp=now))
        messages = messages + [HumanMessage(human_prompt)]
        messages_container.chat_message("human").write(human_prompt)
        # Add user message to history
        add_chat_message(
            st.session_state["selected_chat_id"],
            "human", human_prompt)

        agent = Agent(
            vectorstore=get_vectorstore(),
            embedder=get_embedder()
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

        # Show spinner while streaming
        with st.spinner("Generating response..."):
            # Stream the response
            for chunk in response:
                full_response += chunk
                full_response = html.unescape(
                    full_response).replace("\\n", "\n")
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

        # Clear the user query state after all processing is complete
        st.session_state["user_query_state"] = ""

        if rerun:
            st.rerun()

    if len(st.session_state["user_query_state"]) == 0:
        # --- Show request count for throttle ---
        now = time_now()
        if "request_timestamps" not in st.session_state:
            st.session_state["request_timestamps"] = []
        one_hour_ago = now - datetime.timedelta(hours=1)
        recent_requests = [
            ts for ts in st.session_state["request_timestamps"]
            if ts > one_hour_ago]
        request_count = len(recent_requests)
        st.info(f"Requests this hour: {request_count}/5", icon="⏳")
        if user_query := st.chat_input(
                key="user_query", placeholder="Type your prompt here..."):
            st.session_state["user_query_state"] = user_query
            st.rerun()
    else:
        do_response(st.session_state["user_query_state"])

    if st.button('', icon=":material/help:",
                 type="tertiary", help="Prompt suggestions"):
        prompt_help_dialog()

    # if st.button('', icon=":material/refresh:",
    #              type="tertiary", help="Refresh chat"):
    #     st.rerun()


def main():
    # --- EXPERIMENTAL DISCLAIMER ---
    st.markdown(
        """
        <div style="background-color:#fff3cd; color:#856404; border:2px solid #ffeeba; padding: 1.5em; border-radius: 8px; margin-bottom: 2em; font-size: 1.2em;">
        <strong>Disclaimer:</strong> <br>
        <ul>
        <li>This app is <b>experimental</b> and <b>short-lived</b>.</li>
        <li>It is only available to users on the <b>Wellcome Collection network</b>.</li>
        <li>The results are <b>not representative</b> of the organisation's views and may be inaccurate or misleading.</li>
        <li>Do not rely on the outputs for any official or public communication.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        # Create two columns: one for heading, one for logout button
        col_heading, col_logout = st.columns(
            [4, 1], vertical_alignment="bottom")  # Adjust ratio as desired

        # Left column: App title
        with col_heading:
            st.title("Book Chat")
            st.subheader(
                "Semantically search the Wellcome Collection OCR text corpus")
            st.markdown("""
            This app allows you to interact with the Wellcome Collection's text corpus using natural language queries.
            You can ask questions, explore topics, and get answers based on the collection's content.
            """)

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
