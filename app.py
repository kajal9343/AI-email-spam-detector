from flask import Flask,render_template,request

import sqlite3
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
def create_db():

    conn = sqlite3.connect("spam.db")

    cur = conn.cursor()

    cur.execute("""

    CREATE TABLE IF NOT EXISTS history(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    email TEXT,

    result TEXT,

    confidence REAL

    )

    """)

    conn.commit()

    conn.close()

create_db()
def get_stats():

    conn = sqlite3.connect("spam.db")

    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM history")
    total = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM history WHERE result='spam'"
    )
    spam = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM history WHERE result='ham'"
    )
    ham = cur.fetchone()[0]

    conn.close()

    return total,spam,ham
def get_history():

    conn = sqlite3.connect("spam.db")

    cur = conn.cursor()

    cur.execute("""

    SELECT email,result,confidence

    FROM history

    ORDER BY id DESC

    LIMIT 10

    """)

    rows = cur.fetchall()

    conn.close()

    return rows
@app.route("/")
def home():

    history = get_history()

    total,spam,ham = get_stats()

    return render_template(
        "dashboard.html",
        history=history,
        total=total,
        spam=spam,
        ham=ham
    )
@app.route("/predict",methods=["POST"])
def predict():

    email = request.form["email"]

    vector = vectorizer.transform([email])

    result = model.predict(vector)[0]

    confidence = round(
        model.predict_proba(vector).max()*100,
        2
    )

    conn = sqlite3.connect("spam.db")

    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO history
        (email,result,confidence)
        VALUES(?,?,?)
        """,
        (email,result,confidence)
    )

    conn.commit()
    conn.close()

    history = get_history()

    total,spam,ham = get_stats()

    return render_template(
        "dashboard.html",
        prediction=result,
        confidence=confidence,
        history=history,
        total=total,
        spam=spam,
        ham=ham
    )
if __name__=="__main__":
    app.run(debug=True)
