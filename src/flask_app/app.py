import uuid
import wave

from bokeh.embed import server_document
from flask import Flask
from flask import request,jsonify
from flask import current_app, session, url_for, render_template
from flask_bootstrap import Bootstrap
from flask_debugtoolbar import DebugToolbarExtension
from flask_socketio import SocketIO
from flask_socketio import emit
import subprocess
import atexit
from flask_nav import Nav,register_renderer
from flask_nav.elements import Navbar, View,Separator,Subgroup
from flask_bootstrap.nav import BootstrapRenderer

class DarkNavBar(BootstrapRenderer):
    def visit_Navbar(self, node):
        nav_tag = super(DarkNavBar, self).visit_Navbar(node)
        nav_tag['class'] = "navbar navbar-inverse"
        return nav_tag


nav = Nav()


app = Flask(__name__)
app.config['FILEDIR'] = 'static/_files/'
app.config['SECRET_KEY'] = 'hello'
app.debug = True
#app.register_blueprint(audio,url_prefix='/audio')


toolbar = DebugToolbarExtension(app)
bootstrap = Bootstrap(app)
socketio = SocketIO(app)

bokeh_process = subprocess.Popen(
    ['python', '-m', 'bokeh', 'serve', '--allow-websocket-origin=localhost:5000', r'D:\Repos\diurisation-pet-proj\src\bokeh-visual.py'], stdout=subprocess.PIPE)

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

@app.route("/get_my_ip", methods=["GET"])
def get_my_ip():
    #print(type(request.remote_addr))
    return jsonify({'ip': request.remote_addr}), 200

# Audio section
@app.route('/audio')
def index_audio():
    """Return the client application."""
    print('hello!')
    return render_template('audio/main.html')

@socketio.on('start-recording', namespace='/audio')
def start_recording(options):
    print('Hello WORLD!')

    """Start recording audio from the client."""
    #id = uuid.uuid4().hex  # server-side filename
    # Generate ID from IP:
    id = request.remote_addr
    session['wavename'] = id + '.wav'
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
    del session['wavefile']
    del session['wavename']

@nav.navigation()
def mynavbar():
    return Navbar(
        'Celebrity Voice Recognition Demo',
        View('Explore examples', 'index'),
        View('Test with your own voice','index_audio'),
        View('About', 'about')
    )




if __name__ == '__main__':

    nav.init_app(app)
    register_renderer(app,'darknav',DarkNavBar)
    socketio.run(app,debug=True)
    #app.run(debug=True)


