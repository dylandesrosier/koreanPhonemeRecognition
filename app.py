from flask import Flask
import json
import os

app = Flask(__name__)

@app.route("/getScore")
def parseRequest():
  return getScore()

def getScore():
  return json.dumps({
    "score": 90
  })

if __name__ == '__main__':
  port = int(os.environ.get("PORT", 17995))
  app.run(host='0.0.0.0', port=port)