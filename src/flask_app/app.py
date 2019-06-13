import uuid
import wave
import os
import bokeh
from bokeh.embed import server_document
from flask import Flask,flash
from flask import request,jsonify
from flask import current_app, session, url_for, render_template,redirect
from flask_bootstrap import Bootstrap
from flask_debugtoolbar import DebugToolbarExtension
from flask_socketio import SocketIO
from flask_socketio import emit
import subprocess
import atexit
from flask_nav import Nav,register_renderer
from pathlib import Path
from flask_nav.elements import Navbar, View,Separator,Subgroup


project_dir = Path(__file__).resolve().parents[2]
print(project_dir)

nav = Nav()

app = Flask(__name__)
app.config['FILEDIR'] = 'static/_files/'
app.config['SECRET_KEY'] = 'hello'
app.config['DEBUG_TB_INTERCEPT_REDIRECTS'] = False
app.debug = True
#app.register_blueprint(audio,url_prefix='/audio')


toolbar = DebugToolbarExtension(app)
bootstrap = Bootstrap(app)
socketio = SocketIO(app)

path_to_bokeh_py = f'{project_dir}/src/bokeh-visual.py'

bokeh_process = subprocess.Popen(
    ['python', '-m', 'bokeh', 'serve', '--allow-websocket-origin=localhost:5000','--allow-websocket-origin=127.0.0.1:5000', path_to_bokeh_py], stdout=subprocess.PIPE)

@atexit.register
def kill_server():
    bokeh_process.kill()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


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
    print('hello from bokeh!')
    script = server_document(url="http://localhost:5006/bokeh-visual",arguments={'wavfile':wavfile})
    return render_template('bokeh_template.html',bokeh_script = script)

@app.route("/bokeh-user")
def bokeh_user():
    from pathlib import Path
    import os
    print('hello from bokeh-user!')
    sname = session.get('wavename','empty')
    pth_str = url_for('static', filename='_files/' + sname)[1:]
    pth = Path(pth_str)
    print(type(pth_str))
    print(os.path.isfile(pth_str))
    if pth.is_file():
        print(f'All good, file exists {pth_str}')
        # Move file to data/raw:
        os.rename(pth_str,f'{project_dir}/data/raw/{sname}')
        # Pass the name without extension
        script = server_document(url="http://localhost:5006/bokeh-visual", arguments={'wavfile': sname.split('.')[0]})
        return render_template('bokeh_template.html', bokeh_script=script)

    else:
        print(f'The path does not exist {pth_str}')
        flash('No user file exists..Did you forget to press the record button (mic) first ?','danger')
        return redirect(url_for('index_audio'))


@app.route("/get_my_ip", methods=["GET"])
def get_my_ip():
    #print(type(request.remote_addr))
    return jsonify({'ip': request.remote_addr}), 200

# Audio section
@app.route('/audio')
def index_audio():
    """Return the client application."""
    id = uuid.uuid4().hex
    session['wavename'] = id + '.wav'
    return render_template('audio/main.html')

@socketio.on('start-recording', namespace='/audio')
def start_recording(options):
    print('Hello WORLD!')
    flash('Started recording...', 'primary')
    """Start recording audio from the client."""
    #id = uuid.uuid4().hex  # server-side filename
    # Generate ID from IP:
    #id = request.remote_addr
    print(current_app.config['FILEDIR'] + session['wavename'])
    wf = wave.open(current_app.config['FILEDIR'] + session['wavename'], 'wb')
    wf.setnchannels(options.get('numChannels', 1))
    wf.setsampwidth(options.get('bps', 16) // 8)
    wf.setframerate(options.get('fps', 44100))
    session['wavefile'] = wf


@socketio.on('write-audio', namespace='/audio')
def write_audio(data):
    """Write a chunk of audio from the client."""
    session['wavefile'].writeframes(data)


@socketio.on('end-recording', namespace='/audio')
def end_recording():
    """Stop recording audio from the client."""
    emit('add-wavefile', url_for('static',
                                 filename='_files/' + session['wavename']))
    print(session['wavename'])
    session['wavefile'].close()
    flash('Finished recording...', 'success')

    del session['wavefile']





if __name__ == '__main__':
    #print(os.path.isfile('static/_files/tmp.wav'))
    nav.init_app(app)
    socketio.run(app,debug=True)
    app.run(debug=True)


