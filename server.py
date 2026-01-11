import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
    Query,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add index-tts to sys.path
ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "index-tts"))

try:
    from indextts.infer_v2 import IndexTTS2
except ImportError as e:
    print(f"Error importing IndexTTS2: {e}")
    # We might be in a state where we can't run, but let's try to continue
    # so we can at least show the error when running
    IndexTTS2 = None

app = FastAPI()

# Configuration
STORAGE_DIR = os.path.join(ROOT_DIR, "storage")
SPEAKERS_DIR = os.path.join(STORAGE_DIR, "speakers")
OUTPUTS_DIR = os.path.join(STORAGE_DIR, "outputs")
SPEAKERS_FILE = os.path.join(STORAGE_DIR, "speakers.json")

os.makedirs(SPEAKERS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Global TTS model
tts = None


def init_model():
    global tts
    if tts is None and IndexTTS2 is not None:
        try:
            model_dir = os.path.join(ROOT_DIR, "index-tts", "checkpoints")
            cfg_path = os.path.join(model_dir, "config.yaml")
            print(f"Initializing TTS model from {model_dir}...")
            # Using basic settings suitable for server (assuming CUDA if available)
            # Adjust use_fp16/cuda_kernel based on system capabilities if needed
            import torch

            use_fp16 = torch.cuda.is_available()

            tts = IndexTTS2(
                model_dir=model_dir,
                cfg_path=cfg_path,
                use_fp16=use_fp16,
                use_cuda_kernel=False,
            )
            print("TTS model initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize TTS model: {e}")


@app.on_event("startup")
async def startup_event():
    init_model()


# Data Models
class StatusEnum(str, Enum):
    InProgress = "InProgress"
    Success = "Success"
    Failed = "Failed"


class SpeakerListResponse(BaseModel):
    status: StatusEnum
    message: str
    data: Dict[str, List[List[str]]]  # {speakers: [[name, uuid], ...]}


class BaseResponse(BaseModel):
    status: StatusEnum
    message: str
    data: Optional[Dict[str, Any]] = None


class GenerateRequest(BaseModel):
    speaker_uuid: str
    text: str
    emo_audio_prompt: Optional[str] = None
    emo_alpha: float = 1.0
    use_emo_text: bool = False
    emo_text: Optional[str] = None
    use_random: bool = False
    max_text_tokens_per_segment: int = 120


# In-memory task store
# Format: {task_id: {"status": StatusEnum, "output_path": str, "message": str, "created_at": float}}
tasks: Dict[str, Dict[str, Any]] = {}


def load_speakers() -> Dict[str, Dict[str, str]]:
    if os.path.exists(SPEAKERS_FILE):
        try:
            with open(SPEAKERS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_speakers(speakers: Dict[str, Dict[str, str]]):
    with open(SPEAKERS_FILE, "w") as f:
        json.dump(speakers, f, indent=2)


def generate_audio_task(
    task_id: str, speaker_uuid: str, text: str, params: dict
):
    print(f"Starting task {task_id} for speaker {speaker_uuid}")
    try:
        if tts is None:
            raise Exception("TTS model not initialized")

        speakers = load_speakers()
        if speaker_uuid not in speakers:
            raise Exception("Speaker not found")

        spk_audio_path = os.path.join(SPEAKERS_DIR, f"{speaker_uuid}.wav")
        if not os.path.exists(spk_audio_path):
            raise Exception("Speaker audio file missing")

        output_filename = f"{task_id}.wav"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)

        # Determine emo_audio_prompt path if provided
        # If it's a relative path, assume it's relative to root or handle it.
        # For now, we only support speaker audio or maybe uploaded emo audio?
        # The API doesn't specify how to upload emo audio separately.
        # If emo_audio_prompt is passed, it might be a path on server?
        # Assuming for now user passes None or we use speaker audio as default if mode requires it.

        # Call infer
        tts.infer(
            spk_audio_prompt=spk_audio_path,
            text=text,
            output_path=output_path,
            emo_audio_prompt=params.get("emo_audio_prompt"),
            emo_alpha=params.get("emo_alpha", 1.0),
            use_emo_text=params.get("use_emo_text", False),
            emo_text=params.get("emo_text"),
            use_random=params.get("use_random", False),
            max_text_tokens_per_segment=params.get(
                "max_text_tokens_per_segment", 120
            ),
            verbose=True,
        )

        tasks[task_id]["status"] = StatusEnum.Success
        tasks[task_id]["output_path"] = output_path
        print(f"Task {task_id} completed successfully")

    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id]["status"] = StatusEnum.Failed
        tasks[task_id]["message"] = str(e)


@app.post("/api/v1/register_speaker_audio", response_model=BaseResponse)
async def register_speaker(
    speaker_name: str = Form(...), file: UploadFile = File(...)
):
    try:
        speaker_uuid = str(uuid.uuid4())
        file_path = os.path.join(SPEAKERS_DIR, f"{speaker_uuid}.wav")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        speakers = load_speakers()
        speakers[speaker_uuid] = {"name": speaker_name, "uuid": speaker_uuid}
        save_speakers(speakers)

        return BaseResponse(
            status=StatusEnum.Success,
            message="Speaker registered successfully",
            data=None,
        )
    except Exception as e:
        return BaseResponse(status=StatusEnum.Failed, message=str(e), data=None)


@app.get("/api/v1/get_speakers", response_model=SpeakerListResponse)
async def get_speakers():
    try:
        speakers_dict = load_speakers()
        # Format: list[(speaker name, uuid)]
        speaker_list = [[s["name"], s["uuid"]] for s in speakers_dict.values()]
        return SpeakerListResponse(
            status=StatusEnum.Success,
            message="Speakers retrieved",
            data={"speakers": speaker_list},
        )
    except Exception as e:
        return SpeakerListResponse(
            status=StatusEnum.Failed, message=str(e), data={"speakers": []}
        )


@app.post("/api/v1/generate", response_model=BaseResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    try:
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": StatusEnum.InProgress,
            "created_at": time.time(),
            "message": "Task started",
        }

        params = request.dict(exclude={"speaker_uuid", "text"})
        background_tasks.add_task(
            generate_audio_task,
            task_id,
            request.speaker_uuid,
            request.text,
            params,
        )

        return BaseResponse(
            status=StatusEnum.InProgress,
            message="Task submitted",
            data={"task_id": task_id},
        )
    except Exception as e:
        return BaseResponse(status=StatusEnum.Failed, message=str(e), data=None)


@app.get("/api/v1/generate")
async def check_status(task_id: str = Query(...)):
    if task_id not in tasks:
        # Return JSON error
        return JSONResponse(
            content={
                "status": "Failed",
                "message": "Task not found",
                "data": None,
            },
            status_code=404,
        )

    task = tasks[task_id]
    if task["status"] == StatusEnum.Success:
        # Return the file
        return FileResponse(
            task["output_path"],
            media_type="audio/wav",
            filename="generated.wav",
        )
    elif task["status"] == StatusEnum.Failed:
        return JSONResponse(
            content={
                "status": "Failed",
                "message": task.get("message", "Unknown error"),
                "data": None,
            }
        )
    else:
        return JSONResponse(
            content={
                "status": "InProgress",
                "message": "Task is running",
                "data": None,
            }
        )


@app.delete("/api/v1/delete_speakers", response_model=BaseResponse)
async def delete_speaker(
    speaker_uuid: str,
):  # Usually DELETE uses query param or path param, but spec says input: speaker uuid. Can be body or query. I'll use query for simplicity.
    try:
        speakers = load_speakers()
        if speaker_uuid in speakers:
            del speakers[speaker_uuid]
            save_speakers(speakers)

            # Delete file
            path = os.path.join(SPEAKERS_DIR, f"{speaker_uuid}.wav")
            if os.path.exists(path):
                os.remove(path)

            return BaseResponse(
                status=StatusEnum.Success, message="Speaker deleted", data=None
            )
        else:
            return BaseResponse(
                status=StatusEnum.Failed, message="Speaker not found", data=None
            )

            # Delete file
            path = os.path.join(SPEAKERS_DIR, f"{speaker_uuid}.wav")
            if os.path.exists(path):
                os.remove(path)

            return BaseResponse(status=StatusEnum.Success, message="Speaker deleted", data=None)
        else:
            return BaseResponse(status=StatusEnum.Failed, message="Speaker not found", data=None)
    except Exception as e:
        return BaseResponse(status=StatusEnum.Failed, message=str(e), data=None)
