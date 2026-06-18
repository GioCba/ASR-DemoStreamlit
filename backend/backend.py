import json
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from data import get_infer_data_loader
from fastapi import FastAPI, File, Form, UploadFile, WebSocket
from inference_online import handler_batch, handler_chunks
from models.model.early_exit import Early_conformer
from torch import nn, optim
from torchaudio.models.decoder import ctc_decoder
from util.beam_infer import BeamInference
from util.conf import get_args
from util.data_loader import text_transform
from util.epoch_timer import epoch_time
from util.model_utils import *
from util.tokenizer import *


def load_audio(audio_file: str, target_sr=16000):

    waveform, sr = torchaudio.load(audio_file)

    # stereo -> mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    return waveform, sr


UPLOAD_DIR = None


class Model:
    def __init__(self, args, model, inf, valid_len, dev):
        self.args = args
        self.model = model
        self.inf = inf
        self.valid_len = valid_len
        self.dev = dev


langs = {
    "it": "Italian",
    "en": "English",
}

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded
    global UPLOAD_DIR
    print("App is starting...")
    # Create directory for uploaded files
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    for lang in ["it", "en"]:
        if lang == "it":
            args = get_args([], "Italian")
        else:
            args = get_args([], "English")
        model, inf, valid_len, dev = await load_model(args)
        print(f"{lang}-model loaded")
        models[lang] = Model(args, model, inf, valid_len, dev)

    models["rt"] = models["it"]

    model_loaded = True

    yield
    print("App is shutting down...")
    # Keep files stored after each shutdown of the server
    # if UPLOAD_DIR.exists():
    #     for item in UPLOAD_DIR.iterdir():
    #         try:
    #             if item.is_dir():
    #                 shutil.rmtree(item)
    #             else:
    #                 item.unlink()
    #         except Exception:
    #             pass


app = FastAPI(lifespan=lifespan)

ALL_EXITS = 99


@app.post("/uploads/")
async def upload(
    file: UploadFile = File(...), lang: str = Form("it"), exit: int = Form(5)
):
    # Save file in uploads directory
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as out_file:
        while content := await file.read(1024 * 1024):
            out_file.write(content)
    await file.close()
    m = models[lang]

    # Get waveform and sample rate
    audio = load_audio(f"uploads/{file.filename}")

    transc = handler_batch(
        m.args,
        m.model,
        m.valid_len,
        m.inf,
        m.dev,
        audio,
        exit=exit,
    )

    return {"result": transc}


class Session:
    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)


sessions = {}
session_cnt = 0


@app.post("/chunks/")
async def handle_chunk(
    file: Annotated[bytes, File()],
    session_id: str | None = Form(None),
    final: bool | None = Form(None),
    lang: str = Form("it"),
    exit: int = Form(5),
):
    global session_cnt
    m = models[lang]

    s = sessions.get(session_id)
    if s is None:
        s = Session()
        session_cnt += 1
        session_id = str(session_cnt)
        sessions[session_id] = s

    transc, s.buffer = handler_chunks(
        m.args,
        m.model,
        m.valid_len,
        m.inf,
        m.dev,
        data=file,
        buffer=s.buffer,
        final=final,
        exit=exit,
    )

    return {"result": transc, "session_id": session_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, exit: int = 5, lang: str = "it"):
    global session_cnt

    m = models[lang]

    s = Session()

    await websocket.accept()
    while True:
        message = await websocket.receive()

        if "bytes" in message and message["bytes"] is not None:
            audio = message["bytes"]

            final = len(audio) == 0

            transc, s.buffer = handler_chunks(
                m.args,
                m.model,
                m.valid_len,
                m.inf,
                m.dev,
                data=audio,
                buffer=s.buffer,
                final=final,
                exit=exit,
            )

            if transc:
                await websocket.send_text(
                    json.dumps({"result": transc, "final": final}).encode("utf-8")
                )
            else:
                pass

            if final:
                return


async def load_model(args):
    # If model checkpoint path is provided, load it.
    # (Overrides conf parameters)

    """
    if lang == "eng":
        args.load_model_dir = os.getcwd() + '/English-EE-conformer'
        args.load_model_path = args.load_model_dir + "/english-EE-conformer"

    if lang == "it":
        args.load_model_dir = os.getcwd() + '/Italian-EE-conformer'
        args.load_model_path = args.load_model_dir + "/italian-EE-conformer" """

    args.batch_size = 1
    args.device = "cpu"

    # Parse config from command line arguments

    # Define model
    print(args)

    model = Early_conformer(
        src_pad_idx=args.src_pad_idx,
        n_enc_exits=args.n_enc_exits,
        d_model=args.d_model,
        enc_voc_size=args.enc_voc_size,
        dec_voc_size=args.dec_voc_size,
        max_len=args.max_len,
        d_feed_forward=args.d_feed_forward,
        n_head=args.n_heads,
        n_enc_layers=args.n_enc_layers_per_exit,
        features_length=args.n_mels,
        drop_prob=args.drop_prob,
        depthwise_kernel_size=args.depthwise_kernel_size,
        device=args.device,
    ).to(args.device)

    model_path = args.load_model_dir + "/model"
    model.load_state_dict(
        torch.load(model_path, map_location=args.device, weights_only=True)
    )
    print(f"The model has {count_parameters(model):,} trainable parameters")

    inf = BeamInference(args=args)
    valid_len = 0
    dev = args.device

    return model, inf, valid_len, dev


# @app.post("/files/")
# async def handle_file(file: Annotated[bytes, File()]):
#     data = load_audio(file)
#     transc = handler(args, model, valid_len, inf, dev, data=data)

#     return {"text": transc}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
