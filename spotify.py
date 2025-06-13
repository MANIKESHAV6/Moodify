import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, request
import threading

# Flask server for callback
app = Flask(__name__)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    if code:
        print("Authentication successful!")
        return "Close this window and check your app."
    return "Missing auth code", 400

def run_flask():
    app.run(port=8888)

# Start Flask server
threading.Thread(target=run_flask, daemon=True).start()

# Spotify Auth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="009e8f037340461890ccd016ec8150b7",
    client_secret="0b127ae3c80344e18d4bba6cb0d0236f",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="playlist-read-private"
))

# Test connection
try:
    print(sp.current_user()['display_name'])
except Exception as e:
    print(f"Error: {e}")