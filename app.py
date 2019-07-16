from flask import Flask
import json

app = Flask(__name__)

@app.route("/getScore")
def parseRequest():
  return getScore()

def getScore():
  return json.dumps({
    "score": 90
  })

if __name__ == '__main__':
  app.run(debug=True)