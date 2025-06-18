# backend/app.py
from collections import defaultdict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from datetime import datetime
import sqlite3
import os


from mood_detect import detect_mood_once
import sys
print("✅ Backend is running with Python version:", sys.version)


app = Flask(__name__)
app.secret_key = 'your_secret_key'
CORS(app)

DB_PATH = 'database.db'

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS moods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                mood TEXT,
                timestamp TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT,
                mood TEXT,
                songs TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                artist TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_song_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                mood TEXT,
                song_name TEXT,
                artist TEXT,
                timestamp TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()

def cleanup_old_data():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            DELETE FROM user_song_activity
            WHERE timestamp < date('now', '-3 months')
        """)
        conn.commit()
        

@app.route('/')
def home():
    return jsonify({"message": "Backend is running"})

@app.route('/detect', methods=['GET'])
def detect():
    result = detect_mood_once()
    return jsonify(result)

@app.route('/track_song_play', methods=['POST'])
def track_song_play():
    data = request.get_json()
    user_id = data.get('user_id', 1)
    mood = data.get('mood')
    song_name = data.get('name')
    artist = data.get('artist')
    note = data.get('note', '')
    timestamp = datetime.now().isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO user_song_activity (user_id, mood, song_name, artist, timestamp,note)
            VALUES (?, ?, ?, ?, ?,?)
        ''', (user_id, mood, song_name, artist, timestamp,note))
        conn.commit()

    return jsonify({"status": "success", "message": "Song play recorded"})

@app.route('/save_mood_note', methods=['POST'])
def save_mood_note():
    data = request.get_json()
    user_id = data['user_id']
    mood = data['mood']
    note = data['note']

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO user_song_activity (user_id, mood, note, timestamp)
            VALUES (?, ?, ?, datetime('now'))
        ''', (user_id, mood, note))
        conn.commit()

    return jsonify({"success": True})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()

    if user:
        return jsonify({"success": True, "user_id": user[0]})
    else:
        return jsonify({"success": False, "message": "Invalid username or password"})

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required."})

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return jsonify({"success": True})
        except sqlite3.IntegrityError:
            return jsonify({"success": False, "message": "Username already exists."})

@app.route('/user_mood_summary/<int:user_id>/<string:month>')
def user_mood_summary(user_id, month):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            SELECT mood, COUNT(*) FROM user_song_activity
            WHERE user_id = ? AND strftime('%Y-%m', timestamp) = ?
            GROUP BY mood
        ''', (user_id, month))
        results = c.fetchall()
        mood_map = {}
        for mood, count in results:
            mood_cap = mood.capitalize()
            if mood_cap in mood_map:
                mood_map[mood_cap] += count
            else:
                mood_map[mood_cap] = count
        return jsonify(mood_map)


@app.route('/get_username/<int:user_id>')
def get_username(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        result = c.fetchone()
    if result:
        return jsonify({"username": result[0]})
    return jsonify({"username": None})

@app.route('/mood_trend/<int:user_id>/<string:month>')
def mood_trend(user_id, month):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()

        # Fetch current month moods
        c.execute('''
            SELECT mood, COUNT(*) FROM user_song_activity
            WHERE user_id = ? AND strftime('%Y-%m', timestamp) = ?
            GROUP BY mood
        ''', (user_id, month))
        current = dict(c.fetchall())

        # Get previous month
        d = datetime.strptime(month, "%Y-%m")
        prev_month = (d - relativedelta(months=1)).strftime("%Y-%m")

        # Fetch previous month moods
        c.execute('''
            SELECT mood, COUNT(*) FROM user_song_activity
            WHERE user_id = ? AND strftime('%Y-%m', timestamp) = ?
            GROUP BY mood
        ''', (user_id, prev_month))
        previous = dict(c.fetchall())

    if not current:
        return jsonify({})

    top_mood = max(current, key=current.get)
    current_count = current[top_mood]
    prev_count = previous.get(top_mood, 0)

    if prev_count == 0:
        change = 100
        direction = "up"
    else:
        change = abs(current_count - prev_count) * 100 // prev_count
        direction = "up" if current_count > prev_count else "down"

    return jsonify({
        "mood": top_mood.capitalize(),
        "change": change,
        "direction": direction
    })

@app.route("/update_username/<int:user_id>", methods=["POST"])
def update_username(user_id):
    data = request.get_json()
    new_username = data.get("username")
    if not new_username:
        return jsonify({"success": False, "error": "Missing username"})

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET username = ? WHERE id = ?", (new_username, user_id))
        conn.commit()

    return jsonify({"success": True})

if __name__ == '__main__':
    if not os.path.exists(DB_PATH):
        init_db()
    cleanup_old_data()
    app.run(debug=True)

