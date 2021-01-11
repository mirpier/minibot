import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from werkzeug.middleware.proxy_fix import ProxyFix
import add_min_chatboot_v2 as minbot
#import keras.backend.tensorflow_backend as tb 
#from flask_cors import CORS, cross_origin

#inizializzaziione applicazione
app = Flask(__name__)
app.config[ 'SECRET_KEY' ] = ''        
socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins="*")

@app.route( '/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        return render_template('./minibootTpl.html')
    else:
        return render_template('./minibootTpl.html')
        

def messageRecived():
  print( 'Messaggio ricevuto!!!' )


#gestione degli eventi
@socketio.on( 'my eventes')
def handle_my_custom_event1( json1 ):
      message = json1['message']
      #tb._SYMBOLIC_SCOPE.value = True #Correzione baco keras
      answer=minbot.dialoga(message)
      json1['answer'] = answer
      json1['bot']='Mininterno AI'
      print( 'recived my event: ' + str(json1 ))
      socketio.emit( 'my response', json1, callback=messageRecived )        
    
if __name__ == '__main__':
    app.config['PREFERRED_URL_SCHEME'] = 'https'   
    app.wsgi_app = ProxyFix(app.wsgi_app)   
    try:
        socketio.run( app, port=8089, host='0.0.0.0', debug=True, keyfile='key.pem', certfile='cert.pem' )
    except:
        pass #ignora l'errore
 