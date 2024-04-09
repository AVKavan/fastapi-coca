from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import shutil
import tempfile
import cv2
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import VideoFileSink

app = FastAPI()

# Initialize the inference pipeline
model_id = "cocoa-fxvcr/3"
output_file_name = "outputfile.mp4"

@app.post("/process_video/")
async def process_video(video_file: UploadFile = File(...)) -> FileResponse:
    # Save the uploaded video file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        shutil.copyfileobj(video_file.file, temp_video)

    # Initialize the video sink
    video_sink = VideoFileSink.init(video_file_name=output_file_name)

    # Initialize the inference pipeline
    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=temp_video.name,
        on_prediction=video_sink.on_prediction,
    )

    # Start the inference pipeline
    pipeline.start()
    pipeline.join()
    video_sink.release()

    # Return the processed video file
    return FileResponse(output_file_name)



