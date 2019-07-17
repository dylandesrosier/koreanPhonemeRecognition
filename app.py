from flask import Flask, request
import json
import os
import scoring

app = Flask(__name__)

@app.route("/getScore", methods=["POST"])
def parseRequest():

  return scoring.getScore(request)

@app.route("/*")
def catchAll():
  return "nope"

if __name__ == '__main__':
  port = int(os.environ.get("PORT", 17995))
  app.run(host='0.0.0.0', port=port)

