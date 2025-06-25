from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>✅ Flask is working correctly!</h1><p>If you see this message, your setup is 100% correct.</p>"

if __name__ == "__main__":
    app.run(debug=True, port=5050)
