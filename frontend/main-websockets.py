import datetime
import queue
import threading
import time

import av
import numpy as np
import requests
import streamlit as st
import webrtcvad
from scipy.signal import resample
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from websockets.sync.client import connect


class SharedConfig:
    def __init__(self):
        self._lock = threading.Lock()
        self._target_length_ms = 500

    def set_target_length(self, value):
        with self._lock:
            self._target_length_ms = value

    def get_target_length(self):
        with self._lock:
            return self._target_length_ms


config = SharedConfig()


####################################
# Variables
####################################
if "ctr" not in st.session_state:
    st.session_state["ctr"] = 0
if "is_transcripted" not in st.session_state:
    st.session_state["is_transcripted"] = False
if "lang" not in st.session_state:
    st.session_state["lang"] = "it"
if "is_transcripted" not in st.session_state:
    st.session_state["is_transcripted"] = False
if "transcripted_text" not in st.session_state:
    st.session_state["transcripted_text"] = ""
if "exit" not in st.session_state:
    st.session_state["exit"] = "All"
if "realtime_content" not in st.session_state:
    st.session_state["realtime_content"] = [""] * 6
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = None
if "audio_started" not in st.session_state:
    st.session_state["audio_started"] = False
if "done" not in st.session_state:
    st.session_state["done"] = False
if "rt_exit" not in st.session_state:
    st.session_state.rt_exit = "All"

AUDIO_DIR = "user_files/"
file_to_transcript = ""


SENTINEL = "STOP"


@st.cache_resource
def get_audio_queue():
    return queue.Queue()


@st.cache_resource
def get_update_queue():
    return queue.Queue()


@st.cache_resource
def get_finish_queue():
    return queue.Queue()


update_queue = get_update_queue()
audio_queue = get_audio_queue()
finish_queue = get_finish_queue()

if "finished" not in st.session_state:
    st.session_state["finished"] = None

vad = webrtcvad.Vad(0)


def audio_frame_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray()

    if frame.layout.name == "stereo":
        audio = audio.reshape(-1, 2)
        audio = audio.mean(axis=1)

    num_samples = audio.shape[0]
    target_num_samples = int(num_samples * 16000 / frame.sample_rate)

    audio = resample(audio, target_num_samples)
    audio = audio.astype(np.int16)

    if vad.is_speech(audio.tobytes(), 16000):
        audio_queue.put(audio)

    return frame


def on_audio_ended():
    get_audio_queue().put(SENTINEL)
    get_finish_queue().put(True)


try:
    print(f'{st.session_state.rt_exit}')
    chosen_rt_exit = int(st.session_state.rt_exit)
except ValueError:
    chosen_rt_exit = 99


print(f'{chosen_rt_exit=}')

def sender_worker(audio_queue):
    buf = bytearray()

    # f = open("out.raw", 'wb')
    with connect("ws://127.0.0.1:8000/ws") as websocket:
        while True:
            try:
                data = audio_queue.get()

                target_ms = config.get_target_length()
                target_len = int((target_ms * 640) / 20)
                if isinstance(data, str) and data == SENTINEL:
                    if len(buf) > 0:
                        websocket.send(buf)
                        update_queue.put_nowait(websocket.recv())
                        buf = bytearray()
                    websocket.send(data)
                    msg = websocket.recv()
                    update_queue.put_nowait(msg)
                    print("sentinel recv")
                    print("stopped sending")
                    continue
                else:
                    audio_16 = data

                    print(audio_16)

                    raw_data = audio_16.tobytes()

                    buf.extend(raw_data)

                    if len(buf) >= 8000:
                        print('sending')
                        websocket.send(buf)
                        message = websocket.recv()
                        print(f'{message=}')
                        update_queue.put_nowait(message)
                        buf = bytearray()
                    continue
            except queue.Empty:
                continue


if "worker_thread" not in st.session_state:
    print("starting new thread")
    t = threading.Thread(target=sender_worker, args=(audio_queue,), daemon=True)
    t.start()
    st.session_state.worker_thread = t

####################################
# CSS Styling
####################################

st.markdown(
    """
<style>
    h1 {
        font-size: 24px;
        text-align: center;
        text-transform: uppercase;
    }
</style>
""",
    unsafe_allow_html=True,
)

####################################
# Functions
####################################

# UI
st.title("Transcriber")

# Sidebar
if not st.session_state["model_loaded"]:
    with st.sidebar:
        # Se il server dovesse essere partito ma i modelli non sono caricati
        resp = requests.get("http://127.0.0.1:8000/model_info")
        st.session_state.model_loaded = resp.json()["state"]

        if st.session_state.model_loaded:
            st.success("OK")
        else:
            st.error("Error")

# TABS
mic_file_tab, file_tab, mic_rt_tab = st.tabs(
    ["Record & Transcribe", "Upload & Transcribe", "Real-Time transcription"]
)


with mic_rt_tab:
    with st.container(
        border=True, height="stretch", width="stretch", horizontal_alignment="center"
    ):
        if st.session_state["model_loaded"]:
            ctx = webrtc_streamer(
                key="audio",
                audio_frame_callback=audio_frame_callback,
                on_audio_ended=on_audio_ended,
                mode=WebRtcMode.SENDONLY,
                media_stream_constraints={"video": False, "audio": True},
            )

            st.divider()
            with st.container(horizontal=True, horizontal_alignment="distribute"):
                with st.popover("Settings"):
                    new_length = st.number_input(
                        "Target buffer length in ms",
                        min_value=200,
                        max_value=1000,
                        value=config.get_target_length(),
                        step=40,
                    )

                    config.set_target_length(new_length)

                    st.session_state.lang = st.selectbox(
                        "Seleziona lingua",
                        key="mic_rt_chosen_lang",
                        options=["it", "en"],
                    )

                    # Scelta dell'uscita
                    st.session_state.rt_exit = st.selectbox(
                        "Scegli l'uscita", key="mic_rt_chosen_exit", options=["All", "1", "2", "3", "4", "5", "6"]
                    )

            if ctx.state.playing and not st.session_state["audio_started"]:
                st.session_state["audio_started"] = True
                resp = requests.post(
                    "http://127.0.0.1:8000/model_specs/",
                    data={"lang": st.session_state.lang},
                )
                requests.post('http://127.0.0.1:8000/set_exit/', data={'new_exit': chosen_rt_exit})
                

            st.divider()
            
            history_box1 = st.empty()
            history_box2 = st.empty()
            history_box3 = st.empty()
            history_box4 = st.empty()
            history_box5 = st.empty()
            history_box6 = st.empty()

            history_boxes = [
                history_box1,
                history_box2,
                history_box3,
                history_box4,
                history_box5,
                history_box6,
            ]

            transcript = ""

            poll_interval = 0.2
            while ctx.state.playing:
                try:
                    while True:
                        resp_json = update_queue.get_nowait()
                        text = resp_json.get("result", "")
                        if text:
                            for i in range(len(history_boxes)):
                                transcript += text[i]["text"] + " "
                                history_boxes[i].write(f"Exit {text[i]['exit']}: {transcript}")
                                st.session_state["realtime_content"][i] += transcript + " "
                except queue.Empty:
                    # Fallback, to fix
                    history_box1.write(transcript)

                time.sleep(poll_interval)

            st.divider()

            try:
                st.session_state["finished"] = finish_queue.get_nowait()
            except queue.Empty:
                st.session_state["finished"] = None

            # After stopping
            if st.session_state["finished"]:
                try:
                    st.session_state.done = st.session_state["finished"]
                    resp_json = update_queue.get_nowait()
                    text = resp_json.get("result", "")
                    if st.session_state.rt_exit == "All":
                        for i in range(len(st.session_state.realtime_content)):
                            st.session_state.realtime_content[i] += text[i]['text'] + " "
                            st.write(
                                f"Exit {i + 1}: {st.session_state.realtime_content[i]}"
                            )
                    else:
                        st.session_state.realtime_content[st.session_state.rt_exit-1] += text[st.session_state.rt_exit-1]['text'] + " "
                        st.write(
                            f"Exit {st.session_state.rt_exit}: {st.session_state.realtime_content[i]}"
                        )
                    st.session_state["audio_started"] = False
                except queue.Empty:
                    if st.session_state.rt_exit == "All":
                        for i in range(len(st.session_state.realtime_content)):
                            st.write(
                                f"Exit {i + 1}: {st.session_state.realtime_content[i]}"
                            )
                    else:
                        st.write(
                            f"Exit {st.session_state.rt_exit}: {st.session_state.realtime_content[st.session_state.rt_exit-1]}"
                        )
                        # To prevent overwriting the transcription because of streamlit's reloading of the script
            else:
                if st.session_state.rt_exit == "All":
                    for i in range(len(st.session_state.realtime_content)):
                        st.write(
                            f"Exit {i+ 1}: {st.session_state.realtime_content[i]}"
                        )
                else:
                    st.write(f"Exit {st.session_state.rt_exit}: {st.session_state.realtime_content[int(st.session_state.rt_exit)-1]}")
                    

        else:
            st.warning("Carica il modello dalla sidebar")

def file_tab_fn(mic_mode=False, key=""):

    with st.container(
        border=True, height="stretch", width="stretch", horizontal_alignment="center"
    ):
        if True:  # st.session_state["model_loaded"]:
            if mic_mode:
                file = st.audio_input(
                    "Audio", label_visibility="collapsed", sample_rate=16000
                )
            else:
                file = st.file_uploader(
                    "Upload file",
                    type=[".wav", ".ogg", ".flac"],
                    label_visibility="collapsed",
                )

            st.session_state.is_transcripted = False

            st.divider()

            with st.container(horizontal=True, horizontal_alignment="distribute"):
                with st.popover("Settings"):
                    # Scelta del linguaggio
                    st.session_state.lang = st.selectbox(
                        "Select language",
                        key=f"{key}_file_chosen_lang",
                        options=["it", "en"],
                    )

                    # Scelta della task
                    task_scelta = st.radio(
                        "Select Task", key=f"{key}_lang_chosen_task", options=["ASR"]
                    )

                    # Scelta dell'uscita
                    exit = st.selectbox(
                        "Select exit",
                        key=f"{key}_file_chosen_exit",
                        options=["All", "1", "2", "3", "4", "5", "6"],
                    )
                    st.session_state.exit = int(exit) - 1 if exit != "All" else 99

                # Si procede a trascrivere
                with st.container(horizontal_alignment="center", width="content"):
                    if st.button(
                        "Transcribe", key=f"{key}_file_transcribe_btn", type="tertiary"
                    ):
                        file_to_transcript = file
                        if file_to_transcript is not None:
                            file_to_transcript.name = "user_file.wav"
                            files = {
                                "file": (
                                    file_to_transcript.name,
                                    file_to_transcript,
                                    "audio/wav",
                                )
                            }
                            params = {
                                "lang": st.session_state.lang,
                                "exit": st.session_state.exit,
                            }
                            resp = requests.post(
                                "http://127.0.0.1:8000/uploads/",
                                files=files,
                                data=params,
                            )
                            st.session_state["transcripted_text"] = resp.json()[
                                "result"
                            ]
                            st.session_state.is_transcripted = True
                        else:
                            st.error("Nulla da trascrivere")

            st.divider()

            transc = st.session_state["transcripted_text"]
            for t in transc:
                st.write(f"Exit {t['exit'] + 1}: {t['text']}")
        else:
            st.warning("Load model from sidebar")


with file_tab:
    file_tab_fn(key="file")

with mic_file_tab:
    file_tab_fn(mic_mode=True, key="mic")


# Credits
# with st.container(horizontal_alignment="center"):
#     st.html("<p>Made by Giovanni Confente Broll Avila</p>")
