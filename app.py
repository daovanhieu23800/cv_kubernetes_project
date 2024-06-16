from flask import Flask
import socket
host_name = socket.gethostname()
app = Flask(__name__)

@app.route('/')
def hello():
    return f'Hello, World from {host_name}!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
