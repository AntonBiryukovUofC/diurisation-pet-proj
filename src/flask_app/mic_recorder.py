"""Audio Recording Socket.IO Example

Implements server-side audio recording.
"""
import sys
import os
import uuid
import wave
from flask import Blueprint, current_app, session, url_for, render_template,Flask
from flask_socketio import emit,SocketIO

app = Flask(__name__)
app.config['FILEDIR'] = 'static/_files/'
socketio = SocketIO(app)

from audio import bp as audio_bp
app.register_blueprint(audio_bp, url_prefix='/audio')
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app)