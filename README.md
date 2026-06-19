# ASR_DemoStreamlit

This repository contains a full web application used to transcribe user audio.
There are 3 main features:
- Batch recording and transcription
- File upload and transcription
- Real-time transcription

Currently the only supported models are these: https://huggingface.co/SpeechTek/Italian-EE-conformer https://huggingface.co/SpeechTek/English-EE-conformer and Whisper, with [faster-whisper](https://pypi.org/project/faster-whisper/) (WIP). The latter one only for batch transcription.

## How to run
**N.B.** To run this application install and use Python 3.11, to develop this I personally used Python 3.11.14.
- Clone the repository
- Inside of /backend, put both the directories of the EE-Conformer models. Whisper is automatically downloaded when installing faster-whisper as a requirement
- Create a virtual environment with ```python3.11 -m venv .venv``` inside of the backend directory
- Install the requirements with ```pip install -r requirements.txt``` in the backend directory
- Open a second terminal, in the frontend directory
- Create another virtual environment and install the frontend's requirements
- In the first terminal, get inside of the backend directory and launch the backend server with ```python backend.py```
- After both models are loaded and the server is ready.
- Launch the UI from the frontend directory with ```streamlit run main.py```

## N.B.
I have an error with streamlit-webrtc, specifically on_audio_ended parameter of webrtc_streamer. Before running, Modify the shutdown.py file in ```.venv/lib/python3.11/site-packages/streamlit_webrtc/``` line 126.
Change it from ```if self._polling_thread.is_alive():``` to ```if self._polling_thread is not None and self._polling_thread.is_alive():```.
