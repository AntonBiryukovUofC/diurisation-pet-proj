from flask import render_template,Flask,url_for
from flask_socketio import SocketIO
from audio import audio
from flask_debugtoolbar import DebugToolbarExtension
from flask_bootstrap import Bootstrap
from bokeh.embed import server_document
app = Flask(__name__)
app.config['FILEDIR'] = 'static/_files/'
app.config['SECRET_KEY'] = 'hello'
app.debug = True
app.register_blueprint(audio,url_prefix='/audio')

socketio = SocketIO(app)
toolbar = DebugToolbarExtension(app)
bootstrap = Bootstrap(app)
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/all-links")
def all_links():
    links = []
    print(app.url_map)
    for rule in app.url_map.iter_rules():
        if len(rule.defaults) >= len(rule.arguments):
            url = url_for(rule.endpoint, **(rule.defaults or {}))
            links.append((url, rule.endpoint))
    return render_template("all_links.html", links=links)

@app.route("/bokeh-precalc")
def bokeh_precalc():
    script = server_document(url="http://localhost:5006/bokeh-visual")
    return render_template('bokeh_template.html',bokeh_script = script)

if __name__ == '__main__':
    socketio.run(app,debug=True)
    #app.run(debug=True)


