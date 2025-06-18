import cv2
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from deepface import DeepFace
import time
import random
import threading
from collections import Counter
import webbrowser
import urllib.parse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MoodMusicRecommendationSystem:
    def __init__(self):
        print("🎵 Mood-Based Music Recommendation System (Free Version)")
        print("=" * 60)
        
        # Setup Spotify for search only (no premium required)
        self.setup_spotify()
        
        # Mood to music genre mapping
        self.mood_to_genres = {
            'happy': ['pop', 'dance', 'funk', 'disco', 'electronic', 'upbeat'],
            'sad': ['blues', 'indie', 'alternative', 'folk', 'acoustic', 'melancholy'],
            'angry': ['rock', 'metal', 'punk', 'hard-rock', 'heavy-metal', 'aggressive'],
            'fear': ['ambient', 'classical', 'chill', 'new-age', 'meditation', 'calm'],
            'surprise': ['experimental', 'electronic', 'indie', 'alternative', 'unique'],
            'disgust': ['grunge', 'alternative', 'indie-rock', 'punk', 'raw'],
            'neutral': ['pop', 'indie', 'alternative', 'chill', 'acoustic', 'mainstream']
        }
        
        # Mood to search keywords
        self.mood_to_keywords = {
            'happy': ['happy', 'upbeat', 'cheerful', 'energetic', 'positive', 'joyful', 'celebration'],
            'sad': ['sad', 'melancholy', 'heartbreak', 'emotional', 'lonely', 'crying', 'depression'],
            'angry': ['angry', 'rage', 'aggressive', 'intense', 'furious', 'mad', 'powerful'],
            'fear': ['calm', 'peaceful', 'soothing', 'relaxing', 'gentle', 'comfort', 'anxiety relief'],
            'surprise': ['surprising', 'unexpected', 'unique', 'unusual', 'creative', 'innovative'],
            'disgust': ['authentic', 'real', 'raw', 'unfiltered', 'honest', 'genuine'],
            'neutral': ['popular', 'trending', 'mainstream', 'easy listening', 'casual', 'background']
        }
        
        # Current state
        self.current_mood = None
        self.last_recommendation = 0
        self.mood_history = []
        self.current_recommendations = []
        self.recommendation_window = None
        
        # Setup camera
        self.setup_camera()
        
    def setup_spotify(self):
        """Setup Spotify client for search only (no premium required)"""
        print("1. Setting up Spotify for search...")
        
        # Using Client Credentials flow (no user login required)
        SPOTIPY_CLIENT_ID = "009e8f037340461890ccd016ec8150b7"
        SPOTIPY_CLIENT_SECRET = "0b127ae3c80344e18d4bba6cb0d0236f"
        
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id="009e8f037340461890ccd016ec8150b7",
                client_secret="0b127ae3c80344e18d4bba6cb0d0236f"
            )
            
            self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            
            # Test connection with a simple search
            test_search = self.sp.search(q="happy", type='track', limit=1)
            if test_search:
                print("✅ Connected to Spotify API (Search Only)")
            else:
                print("❌ Failed to connect to Spotify")
                
        except Exception as e:
            print(f"❌ Spotify setup error: {e}")
            print("   Will use offline recommendations instead")
            self.sp = None
    
    def setup_camera(self):
        """Setup camera for face detection"""
        print("2. Setting up camera...")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Default camera not found! Trying other indices...")
            for i in range(5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"✅ Found camera at index {i}")
                    break
            else:
                print("❌ No camera found!")
                return False
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Setup face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print("✅ Camera setup complete!")
        return True
    
    def get_spotify_recommendations(self, mood, limit=10):
        """Get song recommendations from Spotify based on mood"""
        if not self.sp:
            return []
        
        try:
            recommendations = []
            keywords = self.mood_to_keywords.get(mood, ['music'])
            genres = self.mood_to_genres.get(mood, ['pop'])
            
            # Search using mood keywords
            for keyword in keywords[:3]:  # Use first 3 keywords
                try:
                    results = self.sp.search(q=keyword, type='track', limit=15)
                    
                    for track in results['tracks']['items']:
                        if track and track['artists']:
                            song_info = {
                                'name': track['name'],
                                'artist': track['artists'][0]['name'],
                                'album': track['album']['name'] if track['album'] else 'Unknown',
                                'spotify_url': track['external_urls']['spotify'] if 'external_urls' in track else '',
                                'preview_url': track['preview_url'],
                                'popularity': track['popularity'],
                                'mood': mood
                            }
                            recommendations.append(song_info)
                            
                except Exception as e:
                    continue
            
            # Search by genre
            for genre in genres[:2]:
                try:
                    results = self.sp.search(q=f'genre:"{genre}"', type='track', limit=10)
                    
                    for track in results['tracks']['items']:
                        if track and track['artists']:
                            song_info = {
                                'name': track['name'],
                                'artist': track['artists'][0]['name'],
                                'album': track['album']['name'] if track['album'] else 'Unknown',
                                'spotify_url': track['external_urls']['spotify'] if 'external_urls' in track else '',
                                'preview_url': track['preview_url'],
                                'popularity': track['popularity'],
                                'mood': mood
                            }
                            recommendations.append(song_info)
                            
                except Exception as e:
                    continue
            
            # Remove duplicates and sort by popularity
            unique_songs = {}
            for song in recommendations:
                key = f"{song['name']}_{song['artist']}"
                if key not in unique_songs:
                    unique_songs[key] = song
            
            # Sort by popularity and return top results
            sorted_songs = sorted(unique_songs.values(), key=lambda x: x['popularity'], reverse=True)
            return sorted_songs[:limit]
            
        except Exception as e:
            print(f"❌ Error getting Spotify recommendations: {e}")
            return []
    
    def get_offline_recommendations(self, mood):
        """Fallback recommendations when Spotify is not available"""
        offline_songs = {
            'happy': [
                {'name': 'Happy', 'artist': 'Pharrell Williams', 'mood': 'happy'},
                {'name': 'Good as Hell', 'artist': 'Lizzo', 'mood': 'happy'},
                {'name': 'Can\'t Stop the Feeling', 'artist': 'Justin Timberlake', 'mood': 'happy'},
                {'name': 'Walking on Sunshine', 'artist': 'Katrina and the Waves', 'mood': 'happy'},
                {'name': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars', 'mood': 'happy'}
            ],
            'sad': [
                {'name': 'Someone Like You', 'artist': 'Adele', 'mood': 'sad'},
                {'name': 'Mad World', 'artist': 'Gary Jules', 'mood': 'sad'},
                {'name': 'Hurt', 'artist': 'Johnny Cash', 'mood': 'sad'},
                {'name': 'Black', 'artist': 'Pearl Jam', 'mood': 'sad'},
                {'name': 'Tears in Heaven', 'artist': 'Eric Clapton', 'mood': 'sad'}
            ],
            'angry': [
                {'name': 'Break Stuff', 'artist': 'Limp Bizkit', 'mood': 'angry'},
                {'name': 'Bodies', 'artist': 'Drowning Pool', 'mood': 'angry'},
                {'name': 'Killing in the Name', 'artist': 'Rage Against the Machine', 'mood': 'angry'},
                {'name': 'B.Y.O.B.', 'artist': 'System of a Down', 'mood': 'angry'},
                {'name': 'Chop Suey!', 'artist': 'System of a Down', 'mood': 'angry'}
            ],
            'fear': [
                {'name': 'Weightless', 'artist': 'Marconi Union', 'mood': 'fear'},
                {'name': 'Clair de Lune', 'artist': 'Claude Debussy', 'mood': 'fear'},
                {'name': 'Gymnopédie No. 1', 'artist': 'Erik Satie', 'mood': 'fear'},
                {'name': 'River', 'artist': 'Joni Mitchell', 'mood': 'fear'},
                {'name': 'Mad About You', 'artist': 'Sting', 'mood': 'fear'}
            ],
            'neutral': [
                {'name': 'Blinding Lights', 'artist': 'The Weeknd', 'mood': 'neutral'},
                {'name': 'Levitating', 'artist': 'Dua Lipa', 'mood': 'neutral'},
                {'name': 'Good 4 U', 'artist': 'Olivia Rodrigo', 'mood': 'neutral'},
                {'name': 'Stay', 'artist': 'The Kid LAROI & Justin Bieber', 'mood': 'neutral'},
                {'name': 'Industry Baby', 'artist': 'Lil Nas X & Jack Harlow', 'mood': 'neutral'}
            ]
        }
        
        return offline_songs.get(mood, offline_songs['neutral'])
    
    def create_youtube_url(self, song_name, artist_name):
        """Create YouTube search URL for a song"""
        query = f"{song_name} {artist_name}"
        encoded_query = urllib.parse.quote(query)
        return f"https://www.youtube.com/results?search_query={encoded_query}"
    
    def show_recommendations(self, mood):
        """Display recommendations in terminal and prepare for viewing"""
        print(f"\n🎵 MOOD DETECTED: {mood.upper()}")
        print("=" * 50)
        
        # Get recommendations
        if self.sp:
            recommendations = self.get_spotify_recommendations(mood)
        else:
            recommendations = self.get_offline_recommendations(mood)
        
        if not recommendations:
            print("❌ No recommendations found")
            return
        
        self.current_recommendations = recommendations[:8]  # Top 8 songs
        
        print(f"🎶 Top songs for your {mood} mood:")
        print("-" * 50)
        
        for i, song in enumerate(self.current_recommendations, 1):
            print(f"{i}. {song['name']}")
            print(f"   👤 Artist: {song['artist']}")
            if 'album' in song:
                print(f"   💿 Album: {song['album']}")
            
            # Create YouTube URL
            youtube_url = self.create_youtube_url(song['name'], song['artist'])
            print(f"   🔗 YouTube: {youtube_url}")
            
            if 'spotify_url' in song and song['spotify_url']:
                print(f"   🎵 Spotify: {song['spotify_url']}")
            
            print()
        
        print("💡 Commands:")
        print("  Press 1-8 to open song on YouTube")
        print("  Press 'r' to refresh recommendations")
        print("  Press 'q' to quit")
        print("=" * 50)
    
    def handle_song_selection(self, key):
        """Handle user song selection"""
        if key >= ord('1') and key <= ord('8'):
            song_index = key - ord('1')
            if song_index < len(self.current_recommendations):
                song = self.current_recommendations[song_index]
                youtube_url = self.create_youtube_url(song['name'], song['artist'])
                print(f"🎵 Opening: {song['name']} by {song['artist']}")
                try:
                    webbrowser.open(youtube_url)
                except Exception as e:
                    print(f"❌ Error opening browser: {e}")
                    print(f"   Manual URL: {youtube_url}")
    
    def update_recommendations_for_mood(self, mood):
        """Update recommendations based on detected mood"""
        current_time = time.time()
        
        # Don't update too frequently
        if current_time - self.last_recommendation < 8:  # Wait 8 seconds
            return
        
        # Build mood history for stability
        self.mood_history.append(mood)
        if len(self.mood_history) > 5:
            self.mood_history.pop(0)
        
        # Use most common mood from recent history
        if len(self.mood_history) >= 3:
            stable_mood = Counter(self.mood_history).most_common(1)[0][0]
        else:
            stable_mood = mood
        
        # Only update if mood changed
        if stable_mood != self.current_mood:
            self.current_mood = stable_mood
            self.last_recommendation = current_time
            
            # Show recommendations in a separate thread
            threading.Thread(
                target=self.show_recommendations,
                args=(stable_mood,),
                daemon=True
            ).start()
    
    def run(self):
        """Main loop for mood detection and music recommendations"""
        if not self.cap:
            print("❌ Camera not available!")
            return
        
        print("\n3. Starting mood-based music recommendations...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'space' - Force mood detection")
        print("  '1-8' - Open recommended song on YouTube")
        print("  'r' - Refresh recommendations")
        print("=" * 60)
        
        frame_count = 0
        last_detection = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 1.3, 5, minSize=(100, 100)
                )
                
                # Process faces
                for (x, y, w, h) in faces:
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Emotion detection every 5 seconds or on spacebar
                    if (current_time - last_detection > 5.0) or (cv2.waitKey(1) & 0xFF == ord(' ')):
                        try:
                            face_roi = frame[y:y+h, x:x+w]
                            
                            # Analyze emotion
                            result = DeepFace.analyze(
                                face_roi,
                                actions=['emotion'],
                                enforce_detection=False,
                                silent=True
                            )
                            
                            if result:
                                emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
                                dominant_emotion = max(emotions, key=emotions.get)
                                confidence = emotions[dominant_emotion]
                                
                                # Display emotion on frame
                                emotion_text = f"{dominant_emotion}: {confidence:.1f}%"
                                color = (0, 255, 0) if confidence > 50 else (0, 255, 255)
                                
                                cv2.putText(frame, emotion_text, (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                
                                # Update recommendations if confidence is high
                                if confidence > 45:
                                    threading.Thread(
                                        target=self.update_recommendations_for_mood,
                                        args=(dominant_emotion,),
                                        daemon=True
                                    ).start()
                                
                                last_detection = current_time
                                
                        except Exception as e:
                            cv2.putText(frame, "Detection failed", (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Display system info
                info_y = 30
                cv2.putText(frame, f"Faces: {len(faces)}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if self.current_mood:
                    info_y += 30
                    cv2.putText(frame, f"Current Mood: {self.current_mood.title()}", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show recommendations count
                if self.current_recommendations:
                    info_y += 30
                    cv2.putText(frame, f"Recommendations: {len(self.current_recommendations)}", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Mood-Based Music Recommendations', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and self.current_mood:
                    # Refresh recommendations
                    threading.Thread(
                        target=self.show_recommendations,
                        args=(self.current_mood,),
                        daemon=True
                    ).start()
                else:
                    # Handle song selection
                    self.handle_song_selection(key)
                        
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete!")

def main():
    """Main function"""
    print("🎵 Welcome to Mood-Based Music Recommendations!")
    print("This system detects your mood and suggests music accordingly.")
    print("No Spotify Premium required - songs open on YouTube!")
    print()
    
    # Create and run the system
    system = MoodMusicRecommendationSystem()
    system.run()

if __name__ == "__main__":
    main()
    
from collections import Counter
import time

def detect_mood_once():
    """Detect mood from webcam over multiple frames and return most frequent result"""
    system = MoodMusicRecommendationSystem()

    detected_moods = []
    frame_count = 0
    max_frames = 10
    print("📸 Starting mood detection...")

    start_time = time.time()
    while frame_count < max_frames:
        ret, frame = system.cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = system.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

        if len(faces) == 0:
            continue

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                result = DeepFace.analyze(
                    face_roi,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
                dominant = max(emotions, key=emotions.get)
                confidence = emotions[dominant]

                if confidence >= 50:  # ✅ Only accept strong predictions
                    detected_moods.append(dominant)
                    frame_count += 1
                    print(f"[{frame_count}] {dominant} ({confidence:.1f}%)")

            except Exception as e:
                continue

        # Optional timeout: 10 seconds
        if time.time() - start_time > 10:
            break

    system.cap.release()

    if not detected_moods:
        return {"error": "No confident mood detected. Please try again."}

    # ✅ Get the most frequent mood from captured results
    final_mood = Counter(detected_moods).most_common(1)[0][0]

    # ✅ Fetch songs
    songs = system.get_spotify_recommendations(final_mood, limit=6) if system.sp else system.get_offline_recommendations(final_mood)

    return {
        "mood": final_mood,
        "songs": songs
    }
