# ASR_DemoStreamlit

This repository contains a full web application used to transcribe user audio.
There are 3 main features:
- Batch recording and transcription
- File upload and transcription
- Real-time transcription

Currently the only supported models are these: https://huggingface.co/SpeechTek/Italian-EE-conformer https://huggingface.co/SpeechTek/English-EE-conformer.

## How to run
**N.B.** To run this application install and use Python 3.11, to develop this I personally used Python 3.11.14.
- Clone the repository
- Create a virtual environment with ```python3.11 -m venv .venv```
- Inside of /backend, put both models' directories
- Install the requirements with ```pip install -r requirements.txt``` in the root directory
- Open a second terminal, with the same virtual environment
- In the first terminal, get inside of the backend directory and launch the backend server with ```python backend.py```
- After both models are loaded and the server is ready. Launch the UI from the root directory with ```streamlit run ./frontend/main.py```

## N.B.
I have an error with streamlit-webrtc, specifically on_audio_ended parameter of webrtc_streamer. Before running, Modify the shutdown.py file in ```.venv/lib/python3.11/site-packages/streamlit_webrtc/``` line 126.
Change it from ```if self._polling_thread.is_alive():``` to ```if self._polling_thread is not None and self._polling_thread.is_alive():```.
