"""Audio Recording Socket.IO Example

Implements server-side audio recording.
"""
import uuid
import wave
from flask import Blueprint, current_app, session, url_for, render_template
from flask_socketio import emit


audio = Blueprint(
    "audio", __name__, static_folder="static", template_folder="templates"
)
from app import socketio


@audio.route("/")
def index():
    """Return the client application."""
    print("hello!")
    return render_template("audio/main.html")


@socketio.on("start-recording", namespace="/audio")
def start_recording(options):
    print("Hello WORLD!")

    """Start recording audio from the client."""
    id = uuid.uuid4().hex  # server-side filename
    session["wavename"] = id + ".wav"
    print(current_app.config["FILEDIR"] + session["wavename"])
    wf = wave.open(current_app.config["FILEDIR"] + session["wavename"], "wb")
    wf.setnchannels(options.get("numChannels", 1))
    wf.setsampwidth(options.get("bps", 16) // 8)
    wf.setframerate(options.get("fps", 44100))
    session["wavefile"] = wf


@socketio.on("write-audio", namespace="/audio")
def write_audio(data):
    """Write a chunk of audio from the client."""
    session["wavefile"].writeframes(data)


@socketio.on("end-recording", namespace="/audio")
def end_recording():
    """Stop recording audio from the client."""
    emit("add-wavefile", url_for("static", filename="_files/" + session["wavename"]))
    print(session["wavename"])
    session["wavefile"].close()
    del session["wavefile"]
    del session["wavename"]
