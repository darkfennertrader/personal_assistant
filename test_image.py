from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage


client = OpenAI()

llm = ChatOpenAI(model="gpt-4o")

SYSTEM_MESSAGE = """
            You are a smart research assistant. Use the search engine to look up information. \
            You are allowed to make multiple calls (either together or in sequence). \
            Only look up information when you are sure of what you want. \
            If you need to look up some information before asking a follow up question, you are allowed to do that!
            You must adhere to this personality: \
                your name is IBO
                you are 9 years old
                you preferred colour is light green
                your best friend is aa girl whose name is Chiara
                you love telling stories to Chiara
                you love eating pizza
"""

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", SYSTEM_MESSAGE),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# chain = prompt | llm

# message = HumanMessage(
#     content=[
#         {
#             "type": "image_url",
#             "image_url": {
#                 "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#             },
#         },
#         {"type": "text", "text": "What’s in this image?"},
#     ]
# )

# response = chain.invoke(input={"messages": [message]})

# print(response)prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", SYSTEM_MESSAGE),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# chain = prompt | llm

# message = HumanMessage(
#     content=[
#         {
#             "type": "image_url",
#             "image_url": {
#                 "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#             },
#         },
#         {"type": "text", "text": "What’s in this image?"},
#     ]
# )

# response = chain.invoke(input={"messages": [message]})

#######################################################################################

# message = messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#                 },
#             },
#             {"type": "text", "text": "What’s in this image?"},
#         ],
#     }
# ]

# messages = [SystemMessage(content=SYSTEM_MESSAGE)] + message

# response = llm.invoke(messages)
# print(response)
########################################################################

openai_client = OpenAI()
llm = ChatOpenAI(model="gpt-4o")

SYSTEM_MESSAGE = """
You are a helpful assistant capable of generating stunning \
images bases on the user_prompt. \
\n\n user_prompt: {message}
"""


@tool()
def generate_image(text: str):
    """This tool generates an image based on the user prompt"""
    # print(size)
    # print(quality)
    # print(style)
    response = openai_client.images.generate(
        model="dall-e-3",
        size="1792x1024",
        quality="hd",
        prompt=text,
        response_format="url",
    )
    return response.data[0].url


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("human", "{message}"),
    ]
)


model_with_tool = llm.bind_tools([generate_image])
image_chain = prompt | model_with_tool


user_input = "generate two images of luxurious modern villa by the ocean"
resp = image_chain.invoke({"message": user_input})


print(resp)
