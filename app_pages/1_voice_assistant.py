import os
import uuid
from typing import List, Dict, Any
from time import sleep
import base64
import random
from io import BytesIO
from PIL import Image
import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from elevenlabs import play, stream, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs
from graphs.general_assistant import Agent, memory
from graphs.prompt import GENERAL_ASSISTANT

el_client = ElevenLabs()
client = OpenAI()
AUDIO_FILE = "./audio/audio.wav"
S2T_MODEL = "whisper-1"

voice_mapping = {
    "Chiara": {
        "voice_id": "EZsPq4XWFIqBFfdtMVEP",
        "stability": 0.8,
        "similarity_boost": 1,
        "style": 0.2,
    },
    "Raimondo": {
        "voice_id": "OdcqDnRQfQeN0ci8y03P",
        "stability": 0.7,
        "similarity_boost": 1,
        "style": 0,
    },
    "Cristina": {
        "voice_id": "ydvxIrLRYz62AceM21Uy",
        "stability": 0.8,
        "similarity_boost": 1,
        "style": 0.1,
    },
}

model = ChatOpenAI(model="gpt-4o", streaming=True)
chatbot = Agent(_model=model, _checkpointer=memory, _system=GENERAL_ASSISTANT)


def streaming_mode(content, no_chunks=1):
    chunks_to_yield = []
    for word in content.split(" "):
        # print(word, end="", flush=True)
        chunks_to_yield.append(word + " ")

        # Check if we have accumulated no_chunks
        if len(chunks_to_yield) == no_chunks:
            yield "".join(chunks_to_yield)
            chunks_to_yield = []  # Reset the list after yielding
            sleep(0.03)

    # If there are remaining chunks that haven't been yielded because they didn't make up a full group of no_chunks
    if chunks_to_yield:
        yield "".join(chunks_to_yield)
        sleep(0.03)


def text_to_speech(text, voice_id, stability, similarity_boost, style):
    """
    Use this tools to transform text to speech
    """
    # text = state["message"][-1].content
    audio_stream = el_client.generate(
        text=text,
        model="eleven_multilingual_v2",
        stream=True,
        optimize_streaming_latency=0,
        voice=Voice(
            voice_id=voice_id,
            settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                use_speaker_boost=True,
                style=style,
            ),
        ),
    )
    return stream(audio_stream)


###############################################################################


def chatbot_anwser(user_messages: List[Dict[str, Any]], config: Dict[str, Any]):
    # print(config)
    response = chatbot.graph.invoke(
        {"messages": [HumanMessage(content=user_messages)]}, config=config  # type: ignore
    )
    print("\nFINAL RESPONSE")
    print(response)
    return response["messages"][-1].content


# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()

    return base64.b64encode(img_byte).decode("utf-8")


def reset_conversation():
    # st.write("before deleting session.state")
    # st.write(st.session_state)
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        st.session_state.pop("messages", None)
    if "no_msgs" in st.session_state:
        st.session_state["no_msgs"] = 0
    if "old_msgs" in st.session_state:
        st.session_state["old_msgs"] = 0

    st.session_state["thread"] = {"thread_id": str(uuid.uuid4())}


def add_rag_to_chatbot():
    pass


def add_image_to_messages():
    if st.session_state.uploaded_img or (
        "camera_img" in st.session_state and st.session_state.camera_img
    ):
        img_type = (
            st.session_state.uploaded_img.type
            if st.session_state.uploaded_img
            else "image/jpeg"
        )
        if img_type == "video/mp4":
            # save the video file
            video_id = random.randint(100000, 999999)
            with open(f"video_{video_id}.mp4", "wb") as f:
                f.write(st.session_state.uploaded_img.read())
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_file",
                            "video_file": f"video_{video_id}.mp4",
                        }
                    ],
                }
            )
            st.session_state["no_msgs"] += 1
        else:
            raw_img = Image.open(
                st.session_state.uploaded_img or st.session_state.camera_img
            )
            img = get_image_base64(raw_img)
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img}"},
                        }
                    ],
                }
            )

        st.session_state["no_msgs"] += 1


def transcribe_audio(client, audio_path):
    """Transcribe audio to text."""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=S2T_MODEL, file=audio_file
        )
        return transcript.text


def speech_to_text(speech_input):
    with open(AUDIO_FILE, "wb") as file:
        file.write(speech_input["bytes"])
        response = transcribe_audio(client, AUDIO_FILE)
        # st.write(response)
    return response


###########################################################################


def show():
    if "thread" not in st.session_state:
        st.session_state["thread"] = {"thread_id": str(uuid.uuid4())}
    if "no_msgs" not in st.session_state:
        st.session_state["no_msgs"] = 0
    if "old_msgs" not in st.session_state:
        st.session_state["old_msgs"] = 0

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("IBO Assistant 🤖")

    # for message in st.session_state.messages:
    #     st.write(message)

    # Displaying the previous messages if there are any
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            for content in msg["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":
                    st.image(content["image_url"]["url"])
                elif content["type"] == "video_file":
                    st.video(content["video_file"])
                elif content["type"] == "audio_file":
                    st.audio(content["audio_file"])

    with st.sidebar:
        st.divider()
        st.button(
            "🗑️ Reset conversation",
            on_click=reset_conversation,
        )
        cols_audio = st.columns(2)
        with cols_audio[0]:
            st.write("### **🎤 Speak:**")
            audio_prompt = None
            if "prev_speech" not in st.session_state:
                st.session_state.prev_speech = None

            speech_input = mic_recorder(
                start_prompt="🎤",
                stop_prompt="🛑",
                just_once=True,
                use_container_width=True,
                format="wav",
                callback=None,
                args=(),
                kwargs={},
            )
            # st.write(speech_input)
            if speech_input and st.session_state.prev_speech != speech_input:
                st.session_state.prev_speech = speech_input
                # st.write(speech_input)
                audio_prompt = speech_to_text(speech_input)
                # st.write("TRUE")
                # st.write(audio_prompt)

        with cols_audio[1]:
            audio_response = st.toggle("Audio response", value=False)
            if audio_response:
                tts_voice = st.selectbox(
                    "Select a voice:",
                    ["Raimondo", "Chiara", "Cristina"],
                )

        st.divider()
        st.write("### **🏞️ Image Generation:**")
        col1, col2 = st.columns(2)
        with col1:
            size = st.radio(
                "Size",
                options=[
                    "1024x1024",
                    "1792x1024",
                    "1024x1792",
                ],
                index=0,
            )
        with col2:
            quality = st.radio(
                "Quality",
                options=[
                    "standard",
                    "hd",
                ],
                index=0,
            )
            style = st.radio(
                "Style",
                options=[
                    "vivid",
                    "natural",
                ],
                index=0,
            )

        st.divider()
        st.write("### **📝 Add a RAG to the chatbot:**")
        with st.popover("📁 Upload"):
            st.file_uploader(
                "Upload a document:",
                type=["doc", "docx", "pdf", "txt", "csv"],
                accept_multiple_files=False,
                key="uploaded_doc",
                on_change=add_rag_to_chatbot,
            )

        st.divider()
        st.write("### **🖼️ Add an image or a video:**")
        cols_img = st.columns(2)
        with cols_img[0]:
            with st.popover("📁 Upload"):
                st.file_uploader(
                    "Upload an image or a video:",
                    type=["png", "jpg", "jpeg", "webp"] + (["mp4"]),
                    accept_multiple_files=False,
                    key="uploaded_img",
                    on_change=add_image_to_messages,
                )

        with cols_img[1]:
            with st.popover("📸 Camera"):
                activate_camera = st.checkbox("Activate camera")
                if activate_camera:
                    st.camera_input(
                        "Take a picture",
                        key="camera_img",
                        on_change=add_image_to_messages,
                    )

    # Chat input
    if prompt := st.chat_input("Hi! Ask me anything...") or audio_prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt or audio_prompt,
                    }
                ],
            }
        )
        st.session_state["no_msgs"] += 1

        # Display the new messages
        with st.chat_message("user"):
            # st.write("BEFORE SENDING TO LLM:")
            # st.write(st.session_state.messages)
            # st.write("no_msgs: ", st.session_state.no_msgs)
            # st.write("old_msgs: ", st.session_state.old_msgs)

            # add all the messages that the user has sent to openai
            user_messages = []
            for idx in range(st.session_state.old_msgs, st.session_state.no_msgs):
                user_msg = st.session_state.messages[idx]["content"][0]
                user_messages.append(user_msg)
                # st.write(user_msg)

            user_message = st.session_state.messages[-1]["content"][0]["text"]
            # message(user_message, is_user=True)
            st.markdown(user_message)
            config = {
                "configurable": st.session_state["thread"],  # for persistence
                "size": size,
                "quality": quality,
                "style": style,
            }

        with st.chat_message("assistant"):

            with st.spinner():
                ai_response = chatbot_anwser(user_messages, config)

            st.write_stream(streaming_mode(ai_response))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ai_response,
                        }
                    ],
                }
            )

            st.session_state["no_msgs"] += 1
            st.session_state["old_msgs"] = st.session_state["no_msgs"]

        if audio_response:
            text = st.session_state.messages[-1]["content"][0]["text"]
            if isinstance(tts_voice, str):
                with st.spinner("generating audio..."):
                    response = text_to_speech(text, **voice_mapping[tts_voice])
            else:
                raise ValueError("variable 'tts_voice' is not a string")
            audio_base64 = base64.b64encode(response).decode("utf-8")
            audio_html = f"""
            <audio controls>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """

            st.html(audio_html)
