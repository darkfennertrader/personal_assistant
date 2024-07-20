# %%
import operator
from typing import Annotated, TypedDict, List, Literal, Dict, Any, Optional, Union
from pathlib import Path
import uuid
import requests
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from IPython.display import Image, display
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END


VISION_MODEL = "dall-e-3"

openai_client = OpenAI()
st_client = TavilySearchResults(max_results=3)
memory = SqliteSaver.from_conn_string(":memory:")

IMAGE_DIRECTORY = Path(__file__).parent.parent / "images"


class ImageOutput(BaseModel):
    """Final response to the user with after using the 'generate_image' tool"""

    response: Optional[str] = Field(
        description="this field is only made of text. It must exclude every filepath that represent an image."
    )
    image_paths: List[str] = Field(
        description="This is a list of all images found in the ToolMessage(content='...')"
    )


class WebSearchOutput(BaseModel):
    """Final response to the user with after using the 'tavily_search_results_json' tool"""

    response: str = Field(
        description="this filed is only made of text. It must exclude: hyperlinks that start with 'http://... or 'https://...'"
    )
    hyperlinks: List[str] = Field(
        description="this is a list of all hyperlinks that start with 'http://... or 'https://...'"
    )

    # @validator("hyperlinks")
    # def check_hyperlinks(cls, v):
    #     if not v:
    #         raise ValueError("field not present")
    #     print("*" * 100)
    #     print(v)
    #     return v


class Response(BaseModel):
    """Final format response to the user. Only one output format must be chosen (not both): ImageOutput or WebSearchOutput."""

    output: Union[ImageOutput, WebSearchOutput]


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


def image_downloader(image_url: str | None) -> str:
    if image_url is None:
        return "No image URL returned from the API."
    response = requests.get(image_url, timeout=30)
    if response.status_code != 200:
        return "Could not download image from URL."
    unique_id = uuid.uuid4()
    image_path = IMAGE_DIRECTORY / f"{unique_id}.png"
    # print(image_path)
    with open(image_path, "wb") as file:
        file.write(response.content)
    return str(image_path)


def generate_tools(
    size: Literal["1024x1024", "1792x1024", "1024x1792"],
    quality: Literal["standard", "hd"],
    style: Literal["vivid", "natural"],
):

    @tool
    def generate_image(text: str) -> str | None:
        """This tool generates an image based on the user prompt"""
        # print(size)
        # print(quality)
        # print(style)
        response = openai_client.images.generate(
            model=VISION_MODEL,
            prompt=text,
            size=size,
            quality=quality,
            style=style,
            response_format="url",
        )
        return response.data[0].url
        # return image_downloader(response.data[0].url)

    return generate_image


class Agent:

    def __init__(self, _model, _checkpointer, _system="") -> None:
        self.model = _model
        self.system = _system

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)  # type: ignore
        graph.add_node("action", self.take_action)  # type: ignore
        graph.add_conditional_edges(
            "llm", self.exist_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=_checkpointer)
        # self.graph = graph.compile()

    def exist_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0  # type: ignore

    def call_openai(self, state: AgentState, config: Dict[str, Any]):
        print("\ncalling LLM:")
        # print(state)
        # print(config)
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        tools = [
            st_client,
            generate_tools(config["size"], config["quality"], config["style"]),
        ]
        # print(config["output"])
        # model_with_tools = self.model.bind_tools(
        #     tools + [config["output"]], tool_choice="any"
        # )
        model_with_tools = self.model.bind_tools(tools)

        response = model_with_tools.invoke(messages)
        print(response)
        print("-" * 80)
        return {"messages": [response]}

    def take_action(self, state: AgentState, config: Dict[str, Any]):
        print("\nCALLING TOOL:")
        tool_calls = state["messages"][-1].tool_calls  # type: ignore
        tool_invocations = []
        # A ToolInvocation is any class with `tool` and `tool_input` attribute.
        for tool_call in tool_calls:
            action = ToolInvocation(
                tool=tool_call["name"],
                tool_input=tool_call["args"],
            )
            tool_invocations.append(action)

        agent_tools = [
            st_client,
            generate_tools(config["size"], config["quality"], config["style"]),
        ]
        # We can now wrap these tools in a simple ToolExecutor.
        tool_executor = ToolExecutor(agent_tools)
        responses = tool_executor.batch(tool_invocations, return_exceptions=True)
        print("*" * 30)
        print("TOOL CALLS:")
        print(tool_calls)
        print("RESPONSES:")
        print(responses)
        print("*" * 30)
        # print(tool_calls)
        tool_messages = [
            ToolMessage(
                content=str(response),
                name=tc["name"],
                tool_call_id=tc["id"],  # type: ignore
            )
            for tc, response in zip(tool_calls, responses)
        ]
        print(tool_messages)
        print("-" * 80)
        return {"messages": tool_messages}


if __name__ == "__main__":

    system_prompt = """
    You are a smart research assistant. Use the search engine to look up information. \
    You are also capable of generating images based on the user prompt. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    size = "1024x1024"
    quality = "standard"
    style = "vivid"

    model = ChatOpenAI(model="gpt-4o", streaming=True)
    chatbot = Agent(_model=model, _checkpointer=memory, _system=system_prompt)

    # display(Image(chatbot.graph.get_graph().draw_png()))  # type: ignore

    # %%

    # different threads inside the checkpointer for multiple conversation
    params = {
        "configurable": {"thread_id": "1"},  # for persistence
        "size": size,
        "quality": quality,
        "style": style,
        "output": Response,
    }
    query = "What is the weather in Milan and Rome?"
    query = "a mocking image of the Italian football squad elimination from Switzerland at Euro 2024"
    # query = "Who won the SuperBowl in 2024? What is the GDP of that state?"
    messages = [HumanMessage(content=query)]

    result = chatbot.graph.invoke({"messages": messages}, config=params)  # type: ignore

    print()
    print("\nFINAL OUTPUT")
    print(result["messages"][-1].content)
    #############################################################################
    # for output in chatbot.graph.stream(
    #     {"messages": messages}, config=params, stream_mode="values"
    # ):
    #     # last_msg = output["messages"][-1]
    #     print(output)
