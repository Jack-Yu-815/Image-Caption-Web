from flask import Flask, request, render_template
from eval import infer_from_file
from pathlib import Path

application = Flask(__name__)


@application.route('/caption', methods=["POST"])
def get_caption():  # put application's code here
    file = request.files['inputFile']
    file_path = "__cache__/" + file.filename

    file.save(file_path)
    caption = infer_from_file(file_path)
    print(caption)

    # delete the temp file
    p = Path(file_path)
    if p.exists() and p.is_file():
        p.unlink()

    return {
        "caption": caption
    }


@application.route("/")
def home_page():
    return render_template("index.html")


if __name__ == '__main__':
    application.run()
