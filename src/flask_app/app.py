import atexit
import logging
import os
import subprocess
import uuid
import wave
from pathlib import Path

import numpy
from bokeh.embed import server_document
from flask import Flask, flash
from flask import current_app, session, url_for, render_template, redirect
from flask import request, jsonify
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_socketio import SocketIO
from flask_socketio import emit

project_dir = Path(__file__).resolve().parents[2]
print(project_dir)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
nav = Nav()

app = Flask(__name__)
app.config["FILEDIR"] = f"{project_dir}/src/flask_app/static/_files/"
app.config["SECRET_KEY"] = "hello_diurisation"
app.config["DEBUG_TB_INTERCEPT_REDIRECTS"] = False
nav.init_app(app)
bootstrap = Bootstrap(app)
socketio = SocketIO(app)
#
path_to_bokeh_py = f"{project_dir}/src/bokeh-visual.py"
if os.getenv('bokeh_runs','no') == 'no':
    bokeh_process = subprocess.Popen(
        [
            "python",
            "-m",
            "bokeh",
            "serve",
            "--allow-websocket-origin=localhost:5000",
            path_to_bokeh_py,
        ],
        stdout=subprocess.PIPE,
    )
    os.environ['bokeh_runs'] = 'yes'


@atexit.register
def kill_server():
    bokeh_process.kill()
    os.environ['bokeh_runs'] = 'no'


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/all-links")
def all_links():
    links = []
    print(app.url_map)
    for rule in app.url_map.iter_rules():
        if len(rule.defaults) >= len(rule.arguments):
            url = url_for(rule.endpoint, **(rule.defaults or {}))
            links.append((url, rule.endpoint))
    return render_template("all_links.html", links=links)


@app.route("/bokeh-precalc/<string:wavfile>")
def bokeh_precalc(wavfile):
    script = server_document(
        url="http://localhost:5006/bokeh-visual", arguments={"wavfile": wavfile}
    )
    return render_template("bokeh_template.html", bokeh_script=script)


@app.route("/bokeh-user")
def bokeh_user():

    sname = session.get("wavename", "empty")
    print(sname)
    pth_str = f'{project_dir}/src/flask_app/' + url_for("static", filename="_files/" + sname)[1:]
    pth = Path(pth_str)
    print(type(pth_str))
    print(os.path.isfile(pth_str))
    if pth.is_file():
        print(f"All good, file exists {pth_str}")
        # Move file to data/raw:
        os.rename(pth_str, f"{project_dir}/data/raw/{sname}")
        # Pass the name without extension
        script = server_document(
            url="http://localhost:5006/bokeh-visual",
            arguments={"wavfile": sname.split(".")[0]},
        )
        return render_template("bokeh_template.html", bokeh_script=script)

    else:
        print(f"The path does not exist {pth_str}")
        flash(
            f"No user file exists {pth_str}..Did you forget to press the record button (mic) first ?",
            "danger",
        )
        #return redirect(url_for("index_audio"))


@app.route("/get_my_ip", methods=["GET"])
def get_my_ip():
    return jsonify({"ip": request.remote_addr}), 200


# Audio section
@app.route("/audio")
def index_audio():
    """Return the client application."""
    id_audio = uuid.uuid4().hex
    session["wavename"] = id_audio + ".wav"
    return render_template("audio/main.html")


@socketio.on("start-recording", namespace="/audio")
def start_recording(options):
    """Start recording audio from the client."""
    session["wavefile"] = None

    #session["wavename"] = str(numpy.random.randint(0,10,1)[0])+session["wavename"]

    log.warning('Started recording')
    flash("Started recording...", "primary")
    print('Started recording!!!!')
    wf = wave.open(current_app.config["FILEDIR"] + session["wavename"], "wb")
    wf.setnchannels(options.get("numChannels", 1))
    wf.setsampwidth(options.get("bps", 16) // 8)
    wf.setframerate(options.get("fps", 44100))
    session["wavefile"] = wf


@socketio.on("write-audio", namespace="/audio")
def write_audio(data):
    """Write a chunk of audio from the client."""
    if 'wavefile' in session.keys():
        session["wavefile"].writeframes(data)
    else:
        print('No wavefile in session!')



@socketio.on("end-recording", namespace="/audio")
def end_recording():
    """Stop recording audio from the client."""
    emit("add-wavefile", url_for("static", filename="_files/" + session["wavename"]))
    print(session["wavename"])
    log.warning('Ended recording!')
    session["wavefile"].close()
    flash("Finished recording...", "success")
    #del session["wavefile"]


if __name__ == "__main__":
    #app.run(debug=True)
    socketio.run(app,host='0.0.0.0', debug=True)

# docker run -it -p 5000:5000 -p 5006:5006 --device /dev/snd \
# 	-e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
# 	-v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
# 	--group-add $(getent group audio | cut -d: -f3) --device /dev/snd celeb-recog:latest