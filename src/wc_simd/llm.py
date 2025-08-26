from langchain_aws import ChatBedrock
import dotenv

from typing import Annotated
import re
from typing_extensions import TypedDict

import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage


def get_llm(
        model_id: str = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
        temperature: float = 0.2) -> ChatBedrock:
    """

    Returns a ChatBedrock LLM instance with the specified model ID and parameters.

    Returns:
        ChatBedrock: An instance of the ChatBedrock LLM with the specified model ID and parameters.
    """

    dotenv.load_dotenv()

    # Set the model ID and parameters for the LLM

    # Claude is a good alternative to GPT-4o:
    # https://blog.promptlayer.com/big-differences-claude-3-5-vs-gpt-4o/

    llm = ChatBedrock(
        model_id=model_id,
        # model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        # model_id="us.meta.llama3-3-70b-instruct-v1:0",
        model_kwargs=dict(temperature=temperature),
    )

    return llm


class ChatSessionState(TypedDict):
    messages: Annotated[list, add_messages]


class ChatSession:
    def get_config(self):
        return {"configurable": {"thread_id": self.session_id}}

    def __init__(self, session_id: str = None, system_instruction: str = None):

        self.system_instruction = system_instruction

        self.llm = ChatBedrock(
            model_id="openai.gpt-oss-120b-1:0",
            model_kwargs=dict(temperature=0.2),
            region="us-west-2"
        )

        if session_id is None:
            session_id = str(uuid.uuid4())

        self.session_id = session_id

        def chatbot(state: ChatSessionState):
            return {"messages": [self.llm.invoke(state["messages"])]}

        graph_builder = StateGraph(ChatSessionState)

        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

        # Create an in-memory checkpointer
        self.memory = InMemorySaver()

        self.graph = graph_builder.compile(checkpointer=self.memory)

    def response_generator(self, user_input: str, stdout: bool = False):

        messages = [HumanMessage(content=user_input)]

        if self.system_instruction is not None and len(
                self.get_messages()) == 0:
            messages.append(SystemMessage(content=self.system_instruction))

        for event in self.graph.stream(
            {"messages": messages},
                config=self.get_config()):
            for value in event.values():
                content = value["messages"][-1].content
                if stdout:
                    print("Assistant:", content)
                yield content

    def send(self, user_input: str, stdout: bool = False):
        def parse_response_event(event):
            # Remove everything between <reasoning> and </reasoning>
            parsed_event = re.sub(
                r"<reasoning>.*?</reasoning>",
                "",
                event,
                flags=re.DOTALL)
            return parsed_event

        return [parse_response_event(x) for x in self.response_generator(
            user_input, stdout=stdout)]

    def get_messages(self):
        state_snapshot = self.graph.get_state(self.get_config())
        messages = state_snapshot.values.get("chatbot", {}).get("messages", [])
        return messages

    def clear_history(self):
        self.memory.delete_thread(self.session_id)
