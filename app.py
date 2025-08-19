#!/usr/bin/env python3
"""
🎧 Enhanced Music Recommender Web Application
============================================

Flask web interface for the complete music recommendation system.

Features:
- Interactive feedback collection with emojis and SOAP notes
- Real-time progress tracking
- Comprehensive visualizations
- API integrations (Spotify, OpenAI)
- Downloadable results and playlists

Usage:
    python app.py
    
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import json
import uuid
from datetime import datetime
import base64
import io
import threading
import time
import pandas as pd

# Import our complete recommender system
from complete_music_recommender import CompleteMusicRecommender, APIIntegrations, MOOD_EMOJIS

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global storage for sessions and progress
sessions_data = {}
progress_data = {}

# Initialize the recommender system
recommender = None
api_integrations = None

def init_system():
    """Initialize the recommendation system."""
    global recommender, api_integrations
    try:
        recommender = CompleteMusicRecommender()
        recommender.load_data()
        api_integrations = APIIntegrations()
        print("✅ System initialized successfully")
        return True
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

# Initialize on startup
system_ready = init_system()

@app.route('/')
def index():
    """Main landing page."""
    if not system_ready:
        return render_template('error.html', 
                             error="System not ready. Please check FMA dataset paths.")
    
    return render_template('index.html', 
                         total_tracks=len(recommender.valid_track_ids) if recommender else 0)

@app.route('/configure', methods=['GET', 'POST'])
def configure():
    """Configuration page for system settings."""
    if request.method == 'POST':
        # Store configuration in session
        session['config'] = {
            'seed_count': int(request.form.get('seed_count', 6)),
            'discovery_pool_size': int(request.form.get('discovery_pool_size', 100)),
            'top_recommendations': int(request.form.get('top_recommendations', 10)),
            'feedback_type': request.form.get('feedback_type', 'interactive'),
            'enable_spotify': 'enable_spotify' in request.form,
            'enable_gpt': 'enable_gpt' in request.form
        }
        
        # Create new session
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        sessions_data[session_id] = {
            'created_at': datetime.now().isoformat(),
            'config': session['config'],
            'status': 'configured'
        }
        
        return redirect(url_for('select_seeds'))
    
    return render_template('configure.html', 
                         api_status={
                             'spotify': bool(api_integrations and api_integrations.spotify_client_id),
                             'openai': bool(api_integrations and api_integrations.openai_api_key)
                         })

@app.route('/select_seeds')
def select_seeds():
    """Seed track selection page."""
    if 'session_id' not in session:
        return redirect(url_for('configure'))
    
    session_id = session['session_id']
    config = session.get('config', {})
    
    # Start seed selection in background
    def select_seeds_background():
        try:
            progress_data[session_id] = {'status': 'selecting_seeds', 'progress': 0}
            
            seed_features = recommender.select_diverse_seeds(config.get('seed_count', 6))
            
            # Enhance seed features with track metadata
            enhanced_seeds = []
            for _, track in seed_features.iterrows():
                track_dict = track.to_dict()
                track_id = track_dict['track_id']
                
                # Get artist and title from metadata if available
                if track_id in recommender.tracks_df.index:
                    try:
                        artist = recommender.tracks_df.loc[track_id, ('artist', 'name')]
                        title = recommender.tracks_df.loc[track_id, ('track', 'title')]
                        
                        if not (pd.isna(artist) or pd.isna(title)):
                            track_dict['artist'] = artist
                            track_dict['title'] = title
                        else:
                            track_dict['artist'] = 'Unknown Artist'
                            track_dict['title'] = f'Track {track_id}'
                    except:
                        track_dict['artist'] = 'Unknown Artist'
                        track_dict['title'] = f'Track {track_id}'
                else:
                    track_dict['artist'] = 'Unknown Artist'
                    track_dict['title'] = f'Track {track_id}'
                
                enhanced_seeds.append(track_dict)
            
            sessions_data[session_id]['seed_features'] = enhanced_seeds
            progress_data[session_id] = {'status': 'seeds_selected', 'progress': 100}
            
        except Exception as e:
            progress_data[session_id] = {'status': 'error', 'error': str(e)}
    
    thread = threading.Thread(target=select_seeds_background)
    thread.start()
    
    return render_template('select_seeds.html', session_id=session_id)

@app.route('/api/progress/<session_id>')
def get_progress(session_id):
    """API endpoint to get progress for a session."""
    return jsonify(progress_data.get(session_id, {'status': 'unknown'}))

@app.route('/api/seeds/<session_id>')
def get_seeds(session_id):
    """API endpoint to get selected seed tracks."""
    if session_id in sessions_data and 'seed_features' in sessions_data[session_id]:
        return jsonify(sessions_data[session_id]['seed_features'])
    return jsonify([])

@app.route('/feedback')
def feedback():
    """Feedback collection page."""
    if 'session_id' not in session:
        return redirect(url_for('configure'))
    
    session_id = session['session_id']
    seed_tracks = sessions_data.get(session_id, {}).get('seed_features', [])
    
    if not seed_tracks:
        flash('No seed tracks found. Please restart the process.', 'error')
        return redirect(url_for('configure'))
    
    return render_template('feedback.html', 
                         seed_tracks=seed_tracks,
                         mood_emojis=MOOD_EMOJIS,
                         session_id=session_id)

@app.route('/api/submit_feedback', methods=['POST'])
def submit_feedback():
    """API endpoint to submit user feedback."""
    if 'session_id' not in session:
        return jsonify({'error': 'No session found'}), 400
    
    session_id = session['session_id']
    feedback_data = request.json
    
    # Store feedback
    sessions_data[session_id]['feedback'] = feedback_data
    
    # Start processing in background
    def process_feedback_background():
        try:
            progress_data[session_id] = {'status': 'processing_feedback', 'progress': 20}
            
            # Apply feedback to recommender
            recommender.feedback_data = []
            for fb in feedback_data:
                fb['timestamp'] = datetime.now().isoformat()
                recommender.feedback_data.append(fb)
            
            progress_data[session_id] = {'status': 'building_profile', 'progress': 40}
            
            # Build user profile
            import pandas as pd
            seed_features_df = pd.DataFrame(sessions_data[session_id]['seed_features'])
            user_profile = recommender.build_user_profile(seed_features_df)
            sessions_data[session_id]['user_profile'] = user_profile
            
            progress_data[session_id] = {'status': 'generating_discovery', 'progress': 60}
            
            # Generate discovery tracks
            config = sessions_data[session_id]['config']
            discovery_tracks = recommender.simulate_discovery_tracks(
                config.get('discovery_pool_size', 100)
            )
            sessions_data[session_id]['discovery_tracks'] = discovery_tracks.to_dict('records')
            
            progress_data[session_id] = {'status': 'generating_recommendations', 'progress': 80}
            
            # Generate recommendations
            recommendations = recommender.generate_recommendations(
                seed_features_df, discovery_tracks, config.get('top_recommendations', 10)
            )
            sessions_data[session_id]['recommendations'] = recommendations.to_dict('records')
            
            progress_data[session_id] = {'status': 'complete', 'progress': 100}
            
        except Exception as e:
            progress_data[session_id] = {'status': 'error', 'error': str(e)}
    
    thread = threading.Thread(target=process_feedback_background)
    thread.start()
    
    return jsonify({'status': 'processing_started'})

@app.route('/processing')
def processing():
    """Processing progress page."""
    if 'session_id' not in session:
        return redirect(url_for('configure'))
    
    return render_template('processing.html', session_id=session['session_id'])

@app.route('/results')
def results():
    """Results display page."""
    if 'session_id' not in session:
        return redirect(url_for('configure'))
    
    session_id = session['session_id']
    session_data = sessions_data.get(session_id, {})
    
    if 'recommendations' not in session_data:
        flash('Results not ready yet. Please wait for processing to complete.', 'warning')
        return redirect(url_for('processing'))
    
    return render_template('results.html',
                         user_profile=session_data.get('user_profile', {}),
                         recommendations=session_data.get('recommendations', []),
                         seed_tracks=session_data.get('seed_features', []),
                         session_id=session_id,
                         mood_emojis=MOOD_EMOJIS)

@app.route('/api/generate_playlist', methods=['POST'])
def generate_playlist():
    """Generate AI playlist concept."""
    if 'session_id' not in session:
        return jsonify({'error': 'No session found'}), 400
    
    session_id = session['session_id']
    session_data = sessions_data.get(session_id, {})
    
    if not api_integrations:
        return jsonify({'error': 'API integrations not available'}), 500
    
    try:
        user_profile = session_data.get('user_profile', {})
        seed_tracks = session_data.get('seed_features', [])
        
        playlist_concept = api_integrations.generate_gpt_playlist_concept(
            user_profile, seed_tracks
        )
        
        sessions_data[session_id]['playlist_concept'] = playlist_concept
        
        return jsonify(playlist_concept)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search_spotify', methods=['POST'])
def search_spotify():
    """Search tracks on Spotify."""
    if 'session_id' not in session:
        return jsonify({'error': 'No session found'}), 400
    
    session_id = session['session_id']
    session_data = sessions_data.get(session_id, {})
    
    if not api_integrations:
        return jsonify({'error': 'API integrations not available'}), 500
    
    try:
        recommendations = session_data.get('recommendations', [])[:5]  # Top 5
        spotify_matches = []
        
        # Get Spotify token
        if api_integrations.get_spotify_token():
            for rec in recommendations:
                track_id = rec['track_id']
                
                # Get track metadata from recommender
                if track_id in recommender.tracks_df.index:
                    try:
                        artist = recommender.tracks_df.loc[track_id, ('artist', 'name')]
                        title = recommender.tracks_df.loc[track_id, ('track', 'title')]
                        
                        if not (pd.isna(artist) or pd.isna(title)):
                            spotify_track = api_integrations.search_spotify_track(artist, title)
                            
                            if spotify_track:
                                spotify_matches.append({
                                    'fma_id': track_id,
                                    'spotify_id': spotify_track['id'],
                                    'artist': spotify_track['artists'][0]['name'],
                                    'title': spotify_track['name'],
                                    'spotify_url': spotify_track['external_urls']['spotify'],
                                    'preview_url': spotify_track.get('preview_url')
                                })
                    except Exception as e:
                        print(f"Error processing track {track_id}: {e}")
        
        sessions_data[session_id]['spotify_matches'] = spotify_matches
        return jsonify(spotify_matches)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/create_visualization')
def create_visualization():
    """Generate visualization for the session."""
    if 'session_id' not in session:
        return jsonify({'error': 'No session found'}), 400
    
    session_id = session['session_id']
    session_data = sessions_data.get(session_id, {})
    
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        seed_features = pd.DataFrame(session_data.get('seed_features', []))
        recommendations = pd.DataFrame(session_data.get('recommendations', []))
        
        if seed_features.empty or recommendations.empty:
            return jsonify({'error': 'Insufficient data for visualization'}), 400
        
        # Create simple visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Seed tracks
        ax1.scatter(seed_features['tempo'], seed_features['valence_estimate'], 
                   c='red', s=100, alpha=0.7, label='Seed Tracks')
        ax1.set_xlabel('Tempo (BPM)')
        ax1.set_ylabel('Valence (Happiness)')
        ax1.set_title('🎵 Your Seed Tracks')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Recommendations
        scatter = ax2.scatter(recommendations['tempo'], recommendations['valence_estimate'],
                             c=recommendations['discovery_score'], 
                             cmap='plasma', s=100, alpha=0.8)
        ax2.set_xlabel('Tempo (BPM)')
        ax2.set_ylabel('Valence (Happiness)')
        ax2.set_title('🎯 Your Recommendations')
        plt.colorbar(scatter, ax=ax2, label='Discovery Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({'image': f'data:image/png;base64,{img_base64}'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_results')
def export_results():
    """Export complete results as JSON."""
    if 'session_id' not in session:
        return jsonify({'error': 'No session found'}), 400
    
    session_id = session['session_id']
    session_data = sessions_data.get(session_id, {})
    
    # Create export data
    export_data = {
        'session_id': session_id,
        'timestamp': datetime.now().isoformat(),
        'config': session_data.get('config', {}),
        'user_profile': session_data.get('user_profile', {}),
        'seed_tracks': session_data.get('seed_features', []),
        'recommendations': session_data.get('recommendations', []),
        'feedback': session_data.get('feedback', []),
        'playlist_concept': session_data.get('playlist_concept', {}),
        'spotify_matches': session_data.get('spotify_matches', [])
    }
    
    # Save to file
    filename = f'music_recommender_results_{session_id[:8]}.json'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files from the FMA dataset."""
    try:
        # Extract track ID from filename (e.g., "123456.mp3" -> 123456)
        track_id = filename.split('.')[0]
        
        # Construct path to the actual FMA audio file
        # FMA files are organized in subdirectories like: 000/000123.mp3
        track_id_padded = f"{int(track_id):06d}"
        subdir = track_id_padded[:3]
        audio_file_path = os.path.join('fma_small', subdir, f"{track_id_padded}.mp3")
        
        if os.path.exists(audio_file_path):
            return send_file(audio_file_path, mimetype='audio/mpeg')
        else:
            # If file doesn't exist, return a placeholder message
            return jsonify({'error': f'Audio file not found for track {track_id}'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analytics')
def analytics():
    """System analytics page."""
    stats = {
        'total_sessions': len(sessions_data),
        'active_sessions': len([s for s in sessions_data.values() if 'recommendations' in s]),
        'total_tracks': len(recommender.valid_track_ids) if recommender else 0,
        'api_status': {
            'spotify': bool(api_integrations and api_integrations.spotify_client_id),
            'openai': bool(api_integrations and api_integrations.openai_api_key)
        }
    }
    
    return render_template('analytics.html', stats=stats, sessions=sessions_data)

if __name__ == '__main__':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("🚀 Starting Enhanced Music Recommender Web Application")
    print("📱 Open your browser to: http://localhost:5000")
    print("🎵 Ready to create personalized music recommendations!")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
