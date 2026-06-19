import json
import queue
import threading
import time

import av
import numpy as np
import requests
import sounddevice as sd
import streamlit as st
import webrtcvad
from scipy.signal import resample
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from websockets.sync.client import ClientConnection, connect


# Settings modified by UI and used by sender thread
class SharedConfig:
    def __init__(self):
        self._lock = threading.Lock()
        self._target_length_ms = 500
        self._chosen_exit = 99
        self._chosen_lang = "it"
        self._is_http = True

    def set_target_length(self, value):
        with self._lock:
            self._target_length_ms = value

    def set_chosen_lang(self, value):
        with self._lock:
            self._chosen_lang = value

    def set_target_exit(self, value: str):
        with self._lock:
            self._chosen_exit = int(value) - 1 if value != "All" else 99

    def set_target_comm(self, value: str):
        with self._lock:
            self._is_http = True if value == "HTTP" else False

    def get_target_length(self):
        with self._lock:
            return self._target_length_ms

    def get_target_exit(self):
        with self._lock:
            return self._chosen_exit

    def get_chosen_lang(self):
        with self._lock:
            return self._chosen_lang

    def get_target_comm(self):
        with self._lock:
            return self._is_http


####################################
# Variables
####################################
if "config" not in st.session_state:
    st.session_state["config"] = SharedConfig()
if "ctr" not in st.session_state:
    st.session_state["ctr"] = 0
if "is_transcripted" not in st.session_state:
    st.session_state["is_transcripted"] = False
if "lang" not in st.session_state:
    st.session_state["lang"] = "it"
if "transcripted_text" not in st.session_state:
    st.session_state["transcripted_text"] = ""
if "exit" not in st.session_state:
    st.session_state["exit"] = "All"
if "rt_exit" not in st.session_state:
    st.session_state["rt_exit"] = (
        int(st.session_state.exit) - 1 if st.session_state.exit != "All" else 99
    )
if "realtime_content" not in st.session_state:
    st.session_state["realtime_content"] = [""] * 6
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = None
if "audio_started" not in st.session_state:
    st.session_state["audio_started"] = False
if "done" not in st.session_state:
    st.session_state["done"] = False
if "finished" not in st.session_state:
    st.session_state["finished"] = True
if "file_ctr" not in st.session_state:
    st.session_state.file_ctr = 0
if "gain" not in st.session_state:
    st.session_state.gain = 5
if "queues" not in st.session_state:
    st.session_state["queues"] = queue.Queue(), queue.Queue()

audio_queue, update_queue = st.session_state.queues
vad = webrtcvad.Vad(2)
config = st.session_state.config
file_to_transcript = ""
SENTINEL = "STOP"
GAIN = st.session_state.gain


####################################
# Functions
####################################


def get_default_input_device():
    idx = sd.default.device[0]  # input device index
    info = sd.query_devices(idx)
    return idx, info


def classify_device(info):
    name = info["name"].lower()
    channels = info["max_input_channels"]

    if "array" in name or channels > 1:
        return "microphone_array"
    else:
        return "headset"


def get_gain_for_device(device_type):
    if device_type == "microphone_array":
        return 5
    if device_type == "headset":
        return 1
    return 1.5  # fallback


def get_dynamic_gain():
    idx, info = get_default_input_device()
    device_type = classify_device(info)
    gain = get_gain_for_device(device_type)

    return gain


def audio_frame_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray()

    # Convert to mono
    if frame.layout.name == "stereo":
        audio = audio.reshape(-1, 2)
        audio = audio.mean(axis=1)

    num_samples = audio.shape[0]
    target_num_samples = int(num_samples * 16000 / frame.sample_rate)

    # Resample audio to 16 kHz
    audio = resample(audio, target_num_samples)
    # Convert to float
    audio = audio.astype(np.float32) / 32767.0
    # Apply gain
    audio *= GAIN
    # Clip
    audio = np.clip(audio, -1.0, 1.0)
    # Reconvert to int
    audio_16 = (audio * 32767.0).astype(np.int16)

    # VAD control
    if vad.is_speech(audio_16.tobytes(), 16000):
        audio_queue.put(audio_16)

    return frame


# Send sentinel
def on_audio_ended():
    audio_queue.put(SENTINEL)


# WebSocket receiver
def receiver_worker(update_queue, ws):
    try:
        while True:
            msg = ws.recv()
            update_queue.put(json.loads(msg))
    except Exception as e:
        print("Receiver stopped:", e)


def sender_worker(audio_queue):
    buf = bytearray()
    URL = "http://127.0.0.1:8000/chunks/"
    session_id = None
    while True:
        is_http = config.get_target_comm()
        if not is_http:
            receiver_thread: threading.Thread | None = None
            ws: ClientConnection | None = None

            while True:
                try:
                    data = audio_queue.get()
                except queue.Empty:
                    continue

                if ws is None:
                    chosen_exit = config.get_target_exit()
                    chosen_lang = config.get_chosen_lang()
                    ws = connect(
                        f"ws://127.0.0.1:8000/ws?exit={chosen_exit}&lang={chosen_lang}"
                    )
                    receiver_thread = threading.Thread(
                        target=receiver_worker, args=(update_queue, ws)
                    )
                    receiver_thread.start()

                if isinstance(data, str) and data == SENTINEL:
                    ws.send(b"")
                    ws = None
                else:
                    audio_16 = data
                    ws.send(audio_16.tobytes())
        else:
            data = audio_queue.get()

            target_ms = config.get_target_length()
            target_len = int(target_ms * 640 / 20)
            chosen_exit = config.get_target_exit()
            chosen_lang = config.get_chosen_lang()
            if isinstance(data, str) and data == SENTINEL:
                files = {
                    "file": (
                        "microfono.lastpart",
                        bytes(),
                        "application/octet-stream",
                    )
                }

                params = {}

                if session_id:
                    params["session_id"] = session_id
                    params["final"] = True
                    params["lang"] = chosen_lang
                    params["exit"] = chosen_exit

                response = requests.post(
                    URL,
                    files=files,
                    data=params,
                )

                resp_json = response.json()
                session_id = resp_json["session_id"]
                if any(resp_json["result"]):
                    update_queue.put(resp_json)

                session_id = None
                continue

            audio_16 = data

            raw_data = audio_16.tobytes()

            buf.extend(raw_data)

            # If length of buffer greater than target len chosen by user, send
            if len(buf) >= target_len:
                files = {
                    "file": (
                        "microfono.part",
                        buf,
                        "application/octet-stream",
                    )
                }

                params = {}

                if session_id:
                    params["session_id"] = session_id

                params["lang"] = chosen_lang
                params["final"] = False
                params["exit"] = chosen_exit

                response = requests.post(
                    URL,
                    files=files,
                    data=params,
                )

                resp_json = response.json()
                session_id = resp_json["session_id"]
                if any(resp_json["result"]):
                    update_queue.put(resp_json)

                # Clear the buffer
                buf = bytearray()


# Activate thread
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

# UI
st.title("Transcriber")

# TABS
mic_file_tab, file_tab, mic_rt_tab = st.tabs(
    ["Record & Transcribe", "Upload & Transcribe", "Real-Time transcription"]
)

with mic_rt_tab:
    with st.container(
        border=True, height="stretch", width="stretch", horizontal_alignment="center"
    ):
        ctx = webrtc_streamer(
            key="audio",
            audio_frame_callback=audio_frame_callback,
            on_audio_ended=on_audio_ended,
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={"video": False, "audio": True},
        )

        st.divider()
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            new_length = st.number_input(
                "Buffer length in ms",
                min_value=200,
                max_value=1000,
                value=config.get_target_length(),
                step=20,
            )

            config.set_target_length(new_length)

            st.session_state.lang = st.selectbox(
                "Language",
                key="mic_rt_chosen_lang",
                options=["it", "en"],
            )

            rt_exit = st.selectbox(
                "Exit",
                key="mic_rt_chosen_exit",
                options=["All", "1", "2", "3", "4", "5", "6"],
            )

            target_comm = st.selectbox(
                "Communication Protocol",
                key="mic_rt_chosen_comm",
                options=["HTTP", "WebSockets"],
            )

        # Set config parameters before sending
        if ctx.state.playing and not st.session_state["audio_started"]:
            GAIN = get_dynamic_gain()
            config.set_target_comm(target_comm)
            config.set_target_exit(rt_exit)
            config.set_chosen_lang(st.session_state.lang)
            st.session_state.realtime_content = [""] * 6
            st.session_state["audio_started"] = True

        st.divider()

        history_boxes = [st.empty() for _ in range(6)]

        transcript = ""

        chosen_exit = config.get_target_exit()
        chosen_lang = config.get_chosen_lang()

        poll_interval = 0.2
        while ctx.state.playing:
            # Live transcription
            try:
                resp_json = update_queue.get_nowait()
                result = resp_json.get("result", [])
                for r in result:
                    text = r["text"]
                    exit = r["exit"]
                    st.session_state["realtime_content"][exit] += text + " "
            except queue.Empty:
                if chosen_exit == 99:
                    for i, hb in enumerate(history_boxes):
                        hb.write(
                            f"Exit {i + 1}: {st.session_state['realtime_content'][i]}"
                        )
                else:
                    history_boxes[0].write(
                        f"Exit {chosen_exit + 1}: {st.session_state['realtime_content'][chosen_exit]}"
                    )

            time.sleep(poll_interval)

        st.divider()

        final_boxes = [st.empty() for _ in range(6)]

        # Transcription after stopping
        if st.session_state["finished"]:
            try:
                st.session_state["audio_started"] = False
                resp_json = update_queue.get_nowait()
                result = resp_json.get("result", [])
                if chosen_exit == 99:
                    for r in result:
                        text = r["text"]
                        exit = r["exit"]
                        st.session_state["realtime_content"][exit] += text

                        for i in range(len(st.session_state.realtime_content)):
                            st.session_state.realtime_content[i] += text + " "
                            final_boxes[i].write(
                                f"Exit {i + 1}: {st.session_state.realtime_content[i]}"
                            )
                else:
                    chosen_exit = result[0]["exit"]
                    st.session_state.realtime_content[chosen_exit] += (
                        result[0]["text"] + " "
                    )
                    final_boxes[0].write(
                        f"Exit {chosen_exit + 1}: {st.session_state.realtime_content[chosen_exit]}"
                    )
            except queue.Empty:
                if chosen_exit == 99:
                    for i in range(len(st.session_state.realtime_content)):
                        final_boxes[i].write(
                            f"Exit {i + 1}: {st.session_state.realtime_content[i]}"
                        )
                else:
                    final_boxes[0].write(
                        f"Exit {chosen_exit + 1}: {st.session_state.realtime_content[chosen_exit]}"
                    )


def file_tab_fn(mic_mode=False, key=""):
    with st.container(
        border=True, height="stretch", width="stretch", horizontal_alignment="center"
    ):
        if mic_mode:
            file = st.audio_input(
                "Audio", label_visibility="collapsed", sample_rate=16000
            )
        else:
            file = st.file_uploader(
                "Upload file",
                type="audio",
                label_visibility="collapsed",
            )

        st.session_state.is_transcripted = False

        st.divider()

        with st.container(horizontal=True, horizontal_alignment="distribute"):
            with st.popover("Settings", type="primary"):
                st.session_state.lang = st.selectbox(
                    "Choose Model",
                    key=f"{key}_file_chosen_lang",
                    options=["EE-it", "EE-en", "whisper"],
                )

                exit = st.selectbox(
                    "Exit",
                    key=f"{key}_file_chosen_exit",
                    options=["All", "1", "2", "3", "4", "5", "6"],
                )
                st.session_state.exit = int(exit) - 1 if exit != "All" else 99

            with st.container(horizontal_alignment="center", width="content"):
                if st.button(
                    "Transcribe", key=f"{key}_file_transcribe_btn", type="primary"
                ):
                    file_to_transcript = file
                    if file_to_transcript is not None:
                        config.set_chosen_lang(st.session_state.lang)
                        # Get file extension
                        ext = file_to_transcript.name.split(".")[-1]
                        file_to_transcript.name = (
                            f"user_file_{st.session_state.file_ctr}." + ext
                        )
                        st.session_state.file_ctr += 1
                        files = {
                            "file": (
                                file_to_transcript.name,
                                file_to_transcript,
                                "audio",
                            )
                        }
                        params = {
                            "model_type": st.session_state.lang,
                            "exit": st.session_state.exit,
                        }
                        # send file and parameters
                        resp = requests.post(
                            "http://127.0.0.1:8000/uploads/",
                            files=files,
                            data=params,
                        )
                        st.session_state["transcripted_text"] = resp.json()["result"]
                        st.session_state.is_transcripted = True
                    else:
                        st.error("Nothing to transcribe")

        st.divider()

        transc = st.session_state["transcripted_text"]
        if config.get_chosen_lang() == "whisper":
            st.write(st.session_state["transcripted_text"])
        else:
            for t in transc:
                st.write(f"Exit {t['exit'] + 1}: {t['text']}")


with file_tab:
    file_tab_fn(key="file")

with mic_file_tab:
    file_tab_fn(mic_mode=True, key="mic")


# Credits
# with st.container(horizontal_alignment="center"):
#     st.html("<p>Made by Giovanni Confente Broll Avila</p>")
