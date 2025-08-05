from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# Load YOLO model
model = YOLO("best (1).pt")

@app.route("/", methods=["GET", "POST"])
def index():
    user_image = None
    uploaded_image = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_image = url_for('static', filename=f"uploads/{filename}")

            results = model.predict(
                source=filepath, save=True, project="static", name="results", exist_ok=True
            )

            pred_dir = results[0].save_dir
            pred_file = os.listdir(pred_dir)[0]
            user_image = url_for('static', filename=f"results/{pred_file}")

    return render_template("index.html", user_image=user_image, uploaded_image=uploaded_image)

if __name__ == "__main__":
    app.run(debug=True)
