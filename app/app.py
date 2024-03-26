from flask import Flask, render_template, request
from chat_bot.predict import make_prediction

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        message = request.form.get("message")
        response = make_prediction(str(message))
        return render_template("home.html", response=response)
    
    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)

