from typing import List
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import tools_condition
from langgraph.graph.message import add_messages
from langchain.tools.retriever import create_retriever_tool
from langchain_aws import ChatBedrock
from pydantic import BaseModel, Field
from langchain_elasticsearch import ElasticsearchStore
from retrievers import ElasticsearchRetriever


class ResponseFormat(BaseModel):
    """
    RESPONSE FORMAT:
    - In JSON with `answer` and `citations` keys
    - The `answer` key should contain the answer to the user question
        - Begin with a brief summary
        - Provide detailed explanation
        - Include relevant insights from the retrieved documents
    - The `citations` key should contain a list of citation ids from the retrieved documents
    """
    answer: str = Field(
        ..., description=(
            "A string containing the answer to the user question."
            " It should begin with a brief summary, followed by a detailed explanation,"
            " and include relevant insights from the retrieved documents."), example=(
            "Summary: LangChain can enforce structured output using Pydantic.\n"
            "Detailed explanation: Using Pydantic models and the StructuredOutputParser in LangChain,"
            " you can define a JSON schema for LLM responses.\n"
            "Insights:\n"
            "- From doc1: Fields can be validated at runtime.\n"
            "- From doc2: Parsers generate helpful format instructions."))
    citations: List[str] = Field(
        ...,
        description=(
            "A list of citation IDs corresponding to the retrieved documents"
        ),
        example=["doc1", "doc2", "doc3"]
    )


class QueryRewriteFormat(BaseModel):
    """
    RESPONSE FORMAT:
    - In JSON with an `answer` key
    - The `answer` key should contain the rewritten query for vector search
    """
    answer: str = Field(
        ...,
        description=(
            "A string containing the rewritten query suitable for vector search."
        ),
        example="""
        Find the best Italian restaurants in London open now
        """
    )


def get_llm(
        model_id: str = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
        temperature: float = 0.2) -> ChatBedrock:
    """

    Returns a ChatBedrock LLM instance with the specified model ID and parameters.

    Returns:
        ChatBedrock: An instance of the ChatBedrock LLM with the specified model ID and parameters.
    """
    # Set the model ID and parameters for the LLM

    # Claude is a good alternative to GPT-4o:
    # https://blog.promptlayer.com/big-differences-claude-3-5-vs-gpt-4o/

    llm = ChatBedrock(
        model_id=model_id,
        model_kwargs=dict(temperature=temperature),
    )

    return llm


##############################################################################
#  Incremental Answer Parser
##############################################################################


class IncrementalAnswerParser:
    def __init__(self, key: str):
        self.state = "search"  # possible states: 'search', 'stream', 'finished'
        self.buffer = ""
        self.escape = False
        self.key = f'"{key}"'
        self.finished = False

    def process_chunk(self, chunk):
        """Process an incoming chunk and return any complete answer characters."""
        self.buffer += str(chunk)
        output = []

        while self.buffer:
            if self.state == "search":
                # Look for the "answer" key.
                key_index = self.buffer.find(self.key)
                if key_index == -1:
                    # Not found yet—wait for more data.
                    break
                # Find the colon after the key.
                colon_index = self.buffer.find(":", key_index + len(self.key))
                if colon_index == -1:
                    break  # incomplete; wait for more data.
                # Skip whitespace to find the opening quote.
                i = colon_index + 1
                while i < len(self.buffer) and self.buffer[i] in " \t\n\r":
                    i += 1
                if i >= len(self.buffer):
                    break  # wait for more data
                if self.buffer[i] != '"':
                    raise ValueError("Expected opening quote for answer value")
                # Found the opening quote; move into streaming state.
                self.state = "stream"
                # Remove everything up to (and including) the opening quote.
                self.buffer = self.buffer[i + 1:]

            elif self.state == "stream":
                if not self.buffer:
                    break  # Wait for more data.
                char = self.buffer[0]
                self.buffer = self.buffer[1:]
                if self.escape:
                    # Previous character was a backslash: yield this character.
                    output.append(char)
                    self.escape = False
                elif char == '\\':
                    # Escape character encountered.
                    self.escape = True
                    # Optionally, you could handle escape decoding here.
                    output.append(char)
                elif char == '"':
                    # Unescaped closing quote: answer finished.
                    self.state = "finished"
                    self.finished = True
                    return "".join(output)
                else:
                    output.append(char)

            elif self.state == "finished":
                break

        return "".join(output)


##############################################################################
# Agentic Chat Stream with Citations
##############################################################################


# Define the prompt template for agentic chat with citations

GENERATE_PROMPT_TEMPLATE = """You are an expert AI assistant working for the Wellcome Collection with following guidelines.

CORE INSTRUCTIONS:
- You are doing this as part of a chat with books application
- The RETRIEVED DOCUMENTS are text from books and articles that have been OCRed by the Wellcome Collection. They are retrieved based on the user query via vectorsearch.
- Provide clear, concise, and accurate answers
- Base your response strictly on the provided RETRIEVED DOCUMENTS and messages
- Ignore the RETRIEVED DOCUMENTS if they are not relevant to the question
- If information is insufficient, clearly state limitations

RETRIEVED DOCUMENTS:
{context}

PROMPT: {user_prompt}
"""

# Define the rewrite prompt template

REWRITE_PROMPT_TEMPLATE = \
    """You are an expert AI assistant working for the Wellcome Collection. Rewrite the user prompt based on the context of the messages for a more relevant vector search query.

PROMPT: {user_prompt}
"""


# Define a structured model for citations


class Citation(BaseModel):
    source_id: str = Field(
        description="The _id of a SPECIFIC source which justifies the answer.")
    # quote: str = Field(
    # description="The VERBATIM quote from the specified source that
    # justifies the answer.")


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""
    answer: str = Field(
        description="The answer to the user question, which is based only on the given sources.")
    citations: List[Citation] = Field(
        description="Citations (_id) from the given sources that justify the answer.")


class RewriteQueryAnswer(BaseModel):
    answer: str = Field(
        description="The rewrite query for vector search based on the messages.")


class State(MessagesState):
    rewriten_query: str
    context: List[Citation]


def build_agent_app(vectorstore: ElasticsearchStore,
                    metadata_filter=None, k=50):

    search_kwargs = {"k": k}

    if metadata_filter:
        search_kwargs["pre_filter"] = metadata_filter

    # Set up the retriever and LLM
    retriever = vectorstore.as_retriever(
        search_kwargs=search_kwargs
    )

    llm = get_llm()

    # Define application steps

    def rewrite(state: State):
        """Rewrite the user question based on the context"""
        # Create a prompt template that instructs the model to rewrite the
        # question
        prompt = ChatPromptTemplate.from_messages(
            state["messages"][: -1] +
            [("human", REWRITE_PROMPT_TEMPLATE)])
        invoked_prompt = prompt.invoke(
            {"user_prompt": state["messages"][-1].content})
        response: QueryRewriteFormat = llm.with_structured_output(
            schema=QueryRewriteFormat).invoke(invoked_prompt.messages)
        return {"rewriten_query": response.answer}

    def retrieve(state: State):
        query = state["rewriten_query"]
        # retrieved_docs = []
        retrieved_docs = retriever.invoke(query)
        # Create docs context string with metadata
        docs_context = []
        for doc in retrieved_docs:
            # source_id = str(doc.metadata["_id"])
            # text = doc.page_content
            # docs_context.append(
            #     f"{source_id}: {text}")
            docs_context.append(
                str(doc))
        return {"context": docs_context}

    def generate(state: State):
        # Create a prompt template that instructs the model to cite sources
        user_prompt = state["messages"][-1]
        prompt = ChatPromptTemplate.from_messages(
            state["messages"][: -1] +
            [("human", GENERATE_PROMPT_TEMPLATE)])
        invoked_prompt = prompt.invoke(
            {
                "context": "\n\n".join(state["context"]),
                "user_prompt": user_prompt.content
            })
        response: ResponseFormat = llm.with_structured_output(
            # QuotedAnswer,
            schema=ResponseFormat).invoke(
            invoked_prompt.messages)
        return {"messages": [AIMessage(response.answer, response_metadata={
            "citations": response.citations})]}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence(
        [rewrite, retrieve, generate])
    graph_builder.add_edge(START, "rewrite")

    graph = graph_builder.compile()

    return graph


class Agent:
    def __init__(self, vectorstore, embedder):
        self.ai_messages: List[BaseMessage] = []
        self.last_message_citations: List[str] = []
        self.vectorstore = vectorstore
        self.embedder = embedder

    def invoke(self, messages, metadata_filter=None):
        graph = build_agent_app(
            vectorstore=self.vectorstore,
            metadata_filter=metadata_filter)
        parser = IncrementalAnswerParser(key="answer")
        self.ai_messages = []
        for stream_mode, chunk in graph.stream(
            {
                "messages": messages
            },
            stream_mode=["messages", "updates"],
        ):
            if stream_mode == "messages":
                print(f"stream_mode chunk: {chunk}")
                message, metadata = chunk
                if metadata["langgraph_node"] == "generate":
                    message_content: list[dict] = message.content

                    for element in message_content:
                        if isinstance(element, dict) and element["type"] == "tool_use" and (
                                "partial_json" in element):
                            answer_fragment = parser.process_chunk(
                                element["partial_json"])
                            # Immediately output the fragment
                            yield answer_fragment
                            if parser.finished:
                                continue
                if parser.finished:
                    continue
            elif stream_mode == "updates":
                print(f"updates chunk: {chunk}")
                if "generate" in chunk:
                    generate_chunk = chunk["generate"]
                    self.ai_messages = generate_chunk["messages"]
                    self.last_message_citations = self.ai_messages[-1].response_metadata["citations"]
