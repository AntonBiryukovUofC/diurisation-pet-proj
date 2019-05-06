from flask import render_template,Flask,url_for
from flask_socketio import SocketIO
from audio import audio
from flask_debugtoolbar import DebugToolbarExtension


app = Flask(__name__)
app.config['FILEDIR'] = 'static/_files/'
app.register_blueprint(audio)
socketio = SocketIO(app)
toolbar = DebugToolbarExtension(app)
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

if __name__ == '__main__':
#    toolbar = DebugToolbarExtension(app)
#    toolbar.init_app(app)
    #socketio.run(app,debug=True)
    app.run(debug=True)


