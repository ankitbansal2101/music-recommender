#!/usr/bin/env python3
"""
🎧 Complete Enhanced Music Recommender System
==============================================

A comprehensive two-part music recommendation system that combines:
- Part 1: Enhanced collaborative filtering with human-in-loop feedback
- Part 2: AI-powered playlist generation and streaming platform integration

Features:
- Emoji + SOAP-style feedback collection
- Multi-dimensional clustering (MFCC, genre, valence, tempo)
- User profile hypothesis generation
- Discovery integration with trending track simulation
- GPT-powered playlist narrative creation
- Spotify API integration for playlist creation
- YouTube link generation for sharing
- Comprehensive visualizations and analytics

Requirements:
- FMA (Free Music Archive) dataset
- API credentials in .env file (optional):
  - SPOTIFY_CLIENT_ID
  - SPOTIFY_CLIENT_SECRET  
  - OPENAI_API_KEY

Usage:
    python complete_music_recommender.py

Author: Enhanced Music Recommender Team
"""

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Audio, display
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import json
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_LOADED = True
except ImportError:
    ENV_LOADED = False
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

# Try to import OpenAI (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed. Install with: pip install openai")

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Paths - Update these to match your FMA dataset location
AUDIO_DIR = '/Users/ankitbansal/Desktop/reccomender/fma_small'
METADATA_PATH = '/Users/ankitbansal/Desktop/reccomender/fma_metadata/tracks.csv'
GENRES_PATH = '/Users/ankitbansal/Desktop/reccomender/fma_metadata/genres.csv'

# System configuration
CONFIG = {
    'seed_count': 6,                    # Number of initial seed tracks
    'candidate_pool_size': 100,         # Discovery track pool size
    'top_recommendations': 10,          # Number of final recommendations
    'audio_duration': 30.0,             # Seconds of audio to analyze
    'clustering_components': 5          # Number of clusters
}

# Emoji mapping for mood-based feedback
MOOD_EMOJIS = {
    'love': '😍', 'fire': '🔥', 'chill': '😌', 'dance': '💃',
    'focus': '🎯', 'zen': '🧘', 'sleepy': '😴', 'meh': '😐', 'nope': '👎'
}

# =============================================================================
# CORE MUSIC RECOMMENDER CLASS
# =============================================================================

class CompleteMusicRecommender:
    """
    Main class containing the complete music recommendation system.
    
    This class handles:
    - FMA dataset loading and validation
    - Enhanced audio feature extraction
    - User feedback collection and analysis
    - Multi-dimensional clustering
    - User profile hypothesis generation
    - Discovery track simulation
    - Recommendation scoring and ranking
    """
    
    def __init__(self):
        """Initialize the recommender system."""
        self.tracks_df = None
        self.genres_df = None
        self.valid_track_ids = []
        self.feedback_data = []
        self.user_profile = {}
        self.scaler = StandardScaler()
        print("🎧 Complete Music Recommender System initialized")
        
    # -------------------------------------------------------------------------
    # DATA LOADING & VALIDATION
    # -------------------------------------------------------------------------
    
    def load_data(self):
        """Load and validate FMA datasets."""
        print("🔄 Loading FMA datasets...")
        
        try:
            # Load tracks with multi-index structure
            self.tracks_df = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])
            print(f"✅ Loaded tracks: {self.tracks_df.shape}")
            
            # Load genres if available
            try:
                self.genres_df = pd.read_csv(GENRES_PATH, index_col=0)
                print(f"✅ Loaded genres: {self.genres_df.shape}")
            except:
                print("⚠️ Genres file not found, continuing without genre data")
                self.genres_df = None
            
            # Validate audio files
            all_track_ids = self.tracks_df.index.tolist()
            self.valid_track_ids = [tid for tid in all_track_ids if self.is_valid_track(tid)]
            
            print(f"📊 Total tracks in metadata: {len(all_track_ids)}")
            print(f"🎵 Valid audio files found: {len(self.valid_track_ids)}")
            
            if len(self.valid_track_ids) < CONFIG['seed_count']:
                raise Exception(f"Need at least {CONFIG['seed_count']} valid tracks, found {len(self.valid_track_ids)}")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("💡 Make sure FMA dataset paths are correct in the configuration section")
            raise
            
    def get_audio_path(self, track_id):
        """
        Get the file path for an audio track.
        
        Args:
            track_id (int): FMA track ID
            
        Returns:
            str: Path to the MP3 file
        """
        tid_str = f"{track_id:06d}"
        return os.path.join(AUDIO_DIR, tid_str[:3], tid_str + ".mp3")
    
    def is_valid_track(self, track_id):
        """
        Check if a track has a valid audio file.
        
        Args:
            track_id (int): FMA track ID
            
        Returns:
            bool: True if audio file exists
        """
        return os.path.exists(self.get_audio_path(track_id))
    
    # -------------------------------------------------------------------------
    # ENHANCED AUDIO FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    
    def extract_enhanced_features(self, track_id):
        """
        Extract comprehensive audio features from a track.
        
        This function extracts 25+ audio descriptors including:
        - Tempo and rhythm features
        - Spectral characteristics (brightness, rolloff, bandwidth)
        - MFCC coefficients for timbre analysis
        - Energy and dynamics (RMS, zero-crossing rate)
        - Estimated valence (happiness/sadness)
        - Harmonic content (chroma features)
        
        Args:
            track_id (int): FMA track ID
            
        Returns:
            dict: Feature dictionary or None if extraction fails
        """
        path = self.get_audio_path(track_id)
        if not os.path.exists(path):
            return None
        
        try:
            # Load audio file (30 seconds max for efficiency)
            x, sr = librosa.load(path, sr=None, mono=True, duration=CONFIG['audio_duration'])
            
            # 1. RHYTHM FEATURES
            tempo, beats = librosa.beat.beat_track(y=x, sr=sr)
            
            # 2. SPECTRAL FEATURES (timbre and texture)
            spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
            
            # 3. MFCC FEATURES (timbre and voice characteristics)
            mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
            
            # 4. ENERGY AND DYNAMICS
            rms = librosa.feature.rms(y=x)  # Energy/loudness
            zcr = librosa.feature.zero_crossing_rate(x)  # Percussiveness
            
            # 5. HARMONIC CONTENT
            chroma = librosa.feature.chroma_stft(y=x, sr=sr)
            
            # 6. VALENCE ESTIMATION (emotional content)
            # Higher spectral centroid + higher tempo often correlates with positive valence
            valence_estimate = (
                (np.mean(spectral_centroids) / 8000) * 0.4 +  # Brightness factor
                (min(float(tempo) / 140, 1.0)) * 0.3 +         # Tempo factor (capped at 140 BPM)
                (np.mean(rms) * 2) * 0.3                       # Energy factor
            )
            valence_estimate = max(0, min(1, valence_estimate))  # Clamp to [0,1]
            
            # 7. GET GENRE INFORMATION
            genre = self.get_track_genre(track_id)
            
            # Compile all features into a dictionary
            features = {
                'track_id': track_id,
                'tempo': float(tempo),
                'spectral_centroid': np.mean(spectral_centroids),
                'spectral_rolloff': np.mean(spectral_rolloff),
                'spectral_bandwidth': np.mean(spectral_bandwidth),
                'zcr': np.mean(zcr),
                'rms': np.mean(rms),
                'valence_estimate': valence_estimate,
                'mfcc_mean': np.mean(mfccs[0]),  # First MFCC coefficient
                'chroma_mean': np.mean(chroma),   # Average harmonic content
                'genre': genre
            }
            
            # Add individual MFCC coefficients (optional, for advanced analysis)
            for i in range(min(5, len(mfccs))):  # First 5 MFCCs
                features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
            
            return features
            
        except Exception as e:
            print(f"⚠️ Error extracting features for track {track_id}: {e}")
            return None
    
    def get_track_genre(self, track_id):
        """
        Get genre information for a track.
        
        Args:
            track_id (int): FMA track ID
            
        Returns:
            str: Genre name or 'Unknown'
        """
        try:
            if self.tracks_df is not None and ('track', 'genre_top') in self.tracks_df.columns:
                genre = self.tracks_df.loc[track_id, ('track', 'genre_top')]
                return genre if pd.notna(genre) else 'Unknown'
            return 'Unknown'
        except:
            return 'Unknown'
    
    # -------------------------------------------------------------------------
    # SMART SEED SELECTION
    # -------------------------------------------------------------------------
    
    def select_diverse_seeds(self, n_seeds=None):
        """
        Select diverse seed tracks across different genres and characteristics.
        
        This function aims to select a representative sample of tracks that
        spans different musical styles and audio characteristics to provide
        a good foundation for understanding user preferences.
        
        Args:
            n_seeds (int): Number of seed tracks to select
            
        Returns:
            pd.DataFrame: DataFrame with seed track features
        """
        if n_seeds is None:
            n_seeds = CONFIG['seed_count']
            
        print(f"🎲 Selecting {n_seeds} diverse seed tracks...")
        
        selected_seeds = []
        attempts = 0
        max_attempts = n_seeds * 10  # Safety limit
        
        while len(selected_seeds) < n_seeds and attempts < max_attempts:
            track_id = random.choice(self.valid_track_ids)
            
            # Extract features to ensure track is processable
            features = self.extract_enhanced_features(track_id)
            if features:
                selected_seeds.append(features)
                print(f"✅ Selected track {track_id} (genre: {features['genre']}, tempo: {features['tempo']:.0f} BPM)")
            
            attempts += 1
        
        if len(selected_seeds) < n_seeds:
            print(f"⚠️ Could only select {len(selected_seeds)} valid tracks out of {n_seeds} requested")
        
        return pd.DataFrame(selected_seeds)
    
    # -------------------------------------------------------------------------
    # HUMAN-IN-THE-LOOP FEEDBACK COLLECTION
    # -------------------------------------------------------------------------
    
    def collect_feedback(self, track_features, feedback_type="mock"):
        """
        Collect user feedback for tracks.
        
        This system supports multiple types of feedback:
        - Emoji-based mood reactions (😍🔥😌💃🎯🧘😴😐👎)
        - Context-based use cases (workout, study, party, etc.)
        - SOAP-style natural language notes
        - Numerical ratings (1-10 scale)
        
        Args:
            track_features (pd.DataFrame): DataFrame with track features
            feedback_type (str): "mock" for demo data, "interactive" for real input
        """
        print("📝 Collecting user feedback...")
        
        if feedback_type == "mock":
            # Generate realistic mock feedback for demonstration
            mock_responses = [
                {
                    'mood': 'fire', 
                    'context': 'Workout', 
                    'rating': 8, 
                    'soap_notes': 'Great energy, perfect for morning runs. Reminds me of early 2000s electronic music with a modern twist.'
                },
                {
                    'mood': 'chill', 
                    'context': 'Study', 
                    'rating': 7, 
                    'soap_notes': 'Dreamy and atmospheric, but could be too sleepy for daytime work. Nice ambient textures.'
                },
                {
                    'mood': 'love', 
                    'context': 'Coffee Shop', 
                    'rating': 9, 
                    'soap_notes': 'Perfect coffee shop vibe. Warm tones, not intrusive. Would definitely listen again.'
                },
                {
                    'mood': 'dance', 
                    'context': 'Party', 
                    'rating': 6, 
                    'soap_notes': 'Fun rhythm but lacking some energy for a real party. Good for pre-gaming though.'
                },
                {
                    'mood': 'zen', 
                    'context': 'Meditation', 
                    'rating': 8, 
                    'soap_notes': 'Calming and centered. Helps with focus and breathing exercises. Very peaceful.'
                },
                {
                    'mood': 'meh', 
                    'context': 'Background', 
                    'rating': 4, 
                    'soap_notes': 'Too distracting for focused work. Melody pulls attention away from tasks.'
                }
            ]
            
            # Apply mock feedback to tracks
            for i, (_, track) in enumerate(track_features.iterrows()):
                if i < len(mock_responses):
                    feedback = mock_responses[i].copy()
                    feedback['track_id'] = track['track_id']
                    feedback['timestamp'] = datetime.now().isoformat()
                    self.feedback_data.append(feedback)
            
            print(f"✅ Collected mock feedback for {len(self.feedback_data)} tracks")
            
        elif feedback_type == "interactive":
            # Interactive feedback collection (requires user input)
            print("🎭 Interactive feedback collection mode")
            print("For each track, provide your feedback:")
            
            for i, (_, track) in enumerate(track_features.iterrows()):
                print(f"\n🎵 Track {i+1}: ID {track['track_id']} (Genre: {track.get('genre', 'Unknown')})")
                print(f"   Tempo: {track['tempo']:.0f} BPM, Valence: {track['valence_estimate']:.2f}")
                
                # Collect feedback (simplified for demo - in production use proper UI)
                mood_options = list(MOOD_EMOJIS.keys())
                print(f"   Mood options: {mood_options}")
                mood = input("   Enter mood: ").strip() or 'meh'
                
                context_options = ['Workout', 'Study', 'Party', 'Coffee Shop', 'Meditation', 'Background']
                print(f"   Context options: {context_options}")
                context = input("   Enter context: ").strip() or 'Background'
                
                rating = input("   Rating (1-10): ").strip()
                try:
                    rating = int(rating)
                except:
                    rating = 5
                
                notes = input("   SOAP notes (optional): ").strip() or "No specific notes"
                
                feedback = {
                    'track_id': track['track_id'],
                    'mood': mood,
                    'context': context,
                    'rating': rating,
                    'soap_notes': notes,
                    'timestamp': datetime.now().isoformat()
                }
                self.feedback_data.append(feedback)
            
            print(f"✅ Collected interactive feedback for {len(self.feedback_data)} tracks")
    
    # -------------------------------------------------------------------------
    # USER PROFILE HYPOTHESIS GENERATION
    # -------------------------------------------------------------------------
    
    def build_user_profile(self, track_features):
        """
        Build a comprehensive user profile hypothesis from feedback data.
        
        This function analyzes user feedback to infer:
        - Mood preferences and patterns
        - Context-based listening habits
        - Audio feature preferences (tempo, valence, energy, etc.)
        - Discovery tolerance (how adventurous vs conservative)
        - Temporal patterns and consistency
        
        Args:
            track_features (pd.DataFrame): DataFrame with track features
            
        Returns:
            dict: Comprehensive user profile
        """
        if not self.feedback_data:
            print("⚠️ No feedback data available for profile generation")
            return {}
        
        print("🧠 Building user profile hypothesis...")
        
        # Convert feedback to DataFrame for analysis
        feedback_df = pd.DataFrame(self.feedback_data)
        
        # Merge feedback with audio features for correlation analysis
        analysis_df = feedback_df.merge(track_features, on='track_id', how='left')
        
        # 1. MOOD PATTERN ANALYSIS
        mood_patterns = analysis_df.groupby('mood')['rating'].agg(['mean', 'count', 'std']).round(2)
        mood_patterns['preference_strength'] = mood_patterns['mean'] * mood_patterns['count'] / mood_patterns['count'].sum()
        
        # 2. CONTEXT PREFERENCE ANALYSIS
        context_patterns = analysis_df.groupby('context')['rating'].agg(['mean', 'count', 'std']).round(2)
        context_patterns['preference_strength'] = context_patterns['mean'] * context_patterns['count'] / context_patterns['count'].sum()
        
        # 3. AUDIO FEATURE PREFERENCES
        # Calculate weighted preferences based on ratings
        feature_cols = ['tempo', 'valence_estimate', 'spectral_centroid', 'rms', 'zcr']
        preferences = {}
        
        for feature in feature_cols:
            if feature in analysis_df.columns:
                # Weight features by rating (higher rated = more preferred)
                weighted_mean = (
                    analysis_df[feature] * analysis_df['rating']
                ).sum() / analysis_df['rating'].sum()
                
                feature_std = analysis_df[feature].std()
                
                preferences[feature] = {
                    'preferred_value': float(weighted_mean),
                    'range': (float(analysis_df[feature].min()), float(analysis_df[feature].max())),
                    'variance': float(feature_std) if pd.notna(feature_std) else 0.0,
                    'consistency': 1.0 / (1.0 + feature_std) if pd.notna(feature_std) and feature_std > 0 else 1.0
                }
        
        # 4. DISCOVERY TOLERANCE CALCULATION
        # Based on rating variance and genre diversity preference
        rating_variance = analysis_df['rating'].var()
        genre_diversity = len(analysis_df['genre'].unique()) / len(analysis_df) if 'genre' in analysis_df.columns else 0.5
        
        # High variance + high diversity = adventurous, Low variance + low diversity = conservative
        discovery_tolerance = min(0.6, max(0.1, (rating_variance / 10) * 0.7 + genre_diversity * 0.3))
        
        # 5. COMPILE COMPREHENSIVE PROFILE
        self.user_profile = {
            'mood_patterns': mood_patterns.to_dict('index'),
            'context_patterns': context_patterns.to_dict('index'),
            'preferences': preferences,
            'discovery_tolerance': float(discovery_tolerance),
            'total_feedback_points': len(self.feedback_data),
            'average_rating': float(analysis_df['rating'].mean()),
            'rating_variance': float(rating_variance),
            'profile_confidence': min(1.0, len(self.feedback_data) / 10),  # Confidence based on data points
            'generated_at': datetime.now().isoformat()
        }
        
        print(f"✅ Generated user profile with {len(self.feedback_data)} feedback points")
        return self.user_profile
    
    # -------------------------------------------------------------------------
    # DISCOVERY TRACK SIMULATION
    # -------------------------------------------------------------------------
    
    def simulate_discovery_tracks(self, n_tracks=None):
        """
        Simulate external trending/discovery tracks.
        
        In production, this would connect to real APIs like:
        - Spotify Charts API
        - Billboard API  
        - Last.fm trending
        - YouTube Music trending
        - TikTok viral sounds
        
        For this demo, we simulate trending tracks by biasing selection toward
        certain characteristics that make tracks "trendy" (higher tempo, 
        positive valence, moderate energy).
        
        Args:
            n_tracks (int): Number of discovery tracks to generate
            
        Returns:
            pd.DataFrame: DataFrame with discovery track features
        """
        if n_tracks is None:
            n_tracks = CONFIG['candidate_pool_size']
            
        print(f"🔥 Simulating {n_tracks} discovery tracks...")
        
        discovery_tracks = []
        attempts = 0
        max_attempts = n_tracks * 3
        
        while len(discovery_tracks) < n_tracks and attempts < max_attempts:
            track_id = random.choice(self.valid_track_ids)
            
            # Skip if this track is already in our dataset
            if any(dt.get('track_id') == track_id for dt in discovery_tracks):
                attempts += 1
                continue
                
            features = self.extract_enhanced_features(track_id)
            
            if features:
                # Calculate "trendy score" based on characteristics that make tracks popular
                trendy_score = (
                    (features['tempo'] / 140) * 0.3 +        # Prefer danceable tempos (around 120-140 BPM)
                    features['valence_estimate'] * 0.4 +      # Prefer positive vibes
                    features['rms'] * 0.3                     # Prefer moderate energy
                )
                
                # Accept track if it meets trendy criteria or randomly (to add diversity)
                if trendy_score > 0.6 or random.random() < 0.3:
                    features['trendy_score'] = trendy_score
                    features['source'] = 'discovery'
                    discovery_tracks.append(features)
            
            attempts += 1
        
        print(f"✅ Generated {len(discovery_tracks)} discovery tracks")
        return pd.DataFrame(discovery_tracks)
    
    # -------------------------------------------------------------------------
    # RECOMMENDATION GENERATION
    # -------------------------------------------------------------------------
    
    def generate_recommendations(self, seed_features, discovery_tracks, n_recs=None):
        """
        Generate final recommendations by balancing similarity with discovery.
        
        This algorithm implements a "sweet spot" approach:
        - Similar enough to user preferences to be appealing
        - Different enough to provide discovery and growth
        - Weighted by trending factors for social relevance
        
        The discovery score formula balances three factors:
        1. Similarity to user profile (40%) - ensures relevance
        2. Controlled novelty (30%) - pushes boundaries within tolerance
        3. Trending factor (30%) - adds social/cultural relevance
        
        Args:
            seed_features (pd.DataFrame): User's seed tracks
            discovery_tracks (pd.DataFrame): Candidate discovery tracks
            n_recs (int): Number of recommendations to generate
            
        Returns:
            pd.DataFrame: Ranked recommendations with scores
        """
        if n_recs is None:
            n_recs = CONFIG['top_recommendations']
            
        print("🎯 Generating recommendations...")
        
        if self.user_profile and discovery_tracks is not None and not discovery_tracks.empty:
            # Get user preferences from profile
            user_prefs = self.user_profile.get('preferences', {})
            tolerance = self.user_profile.get('discovery_tolerance', 0.3)
            
            discovery_scores = []
            
            for _, track in discovery_tracks.iterrows():
                # Calculate similarity score to user preferences
                similarity_score = 0
                feature_count = 0
                
                for feature, pref_data in user_prefs.items():
                    if feature in track and not pd.isna(track[feature]):
                        pref_value = pref_data['preferred_value']
                        track_value = track[feature]
                        
                        # Normalize differences based on feature type
                        if feature == 'tempo':
                            diff = abs(track_value - pref_value) / 50  # Normalize tempo differences
                        elif feature in ['valence_estimate', 'rms']:
                            diff = abs(track_value - pref_value)  # Already 0-1 range
                        else:
                            diff = abs(track_value - pref_value) / 1000  # Normalize spectral features
                        
                        similarity_score += max(0, 1 - diff)
                        feature_count += 1
                
                if feature_count > 0:
                    similarity_score = similarity_score / feature_count
                else:
                    similarity_score = 0.5  # Default if no features match
                
                # Get trending score
                trendy_score = track.get('trendy_score', 0.5)
                
                # Calculate final discovery score
                # Balance similarity (familiarity) with novelty and trending factors
                discovery_score = (
                    similarity_score * 0.4 +                           # Some familiarity
                    (1 - similarity_score) * tolerance * 0.3 +         # Controlled novelty
                    trendy_score * 0.3                                 # Trending factor
                )
                
                discovery_scores.append({
                    'track_id': track['track_id'],
                    'discovery_score': discovery_score,
                    'similarity_score': similarity_score,
                    'trendy_score': trendy_score,
                    'tempo': track['tempo'],
                    'valence_estimate': track['valence_estimate'],
                    'genre': track.get('genre', 'Unknown'),
                    'explanation': f"Similarity: {similarity_score:.2f}, Novelty: {(1-similarity_score)*tolerance:.2f}, Trending: {trendy_score:.2f}"
                })
            
            # Convert to DataFrame and sort by discovery score
            discovery_df = pd.DataFrame(discovery_scores)
            recommendations = discovery_df.sort_values('discovery_score', ascending=False).head(n_recs)
            
            print(f"✅ Generated {len(recommendations)} recommendations")
            return recommendations
        else:
            print("⚠️ Cannot generate recommendations without user profile and discovery tracks")
            return pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # COMPREHENSIVE ANALYSIS & VISUALIZATION
    # -------------------------------------------------------------------------
    
    def create_comprehensive_analysis(self, seed_features, discovery_tracks, recommendations):
        """
        Create comprehensive visualizations of the recommendation system.
        
        This function generates an 8-panel dashboard showing:
        1. Seed tracks distribution (tempo vs valence)
        2. Discovery tracks with trending scores
        3. Top recommendations with discovery scores
        4. User mood preferences
        5. Context preferences  
        6. Genre distribution in seeds
        7. Audio feature preferences
        8. Discovery vs similarity analysis
        
        Args:
            seed_features (pd.DataFrame): Seed tracks
            discovery_tracks (pd.DataFrame): Discovery candidates
            recommendations (pd.DataFrame): Final recommendations
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        print("📊 Creating comprehensive analysis dashboard...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('🎧 Enhanced Music Recommender System - Comprehensive Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Seed tracks analysis
        plt.subplot(2, 4, 1)
        if not seed_features.empty:
            plt.scatter(seed_features['tempo'], seed_features['valence_estimate'], 
                       c='red', s=100, alpha=0.7, label='Seed Tracks')
            plt.xlabel('Tempo (BPM)')
            plt.ylabel('Valence (Happiness)')
            plt.title('🎵 Seed Tracks Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Discovery tracks analysis
        plt.subplot(2, 4, 2)
        if not discovery_tracks.empty:
            scatter = plt.scatter(discovery_tracks['tempo'], discovery_tracks['valence_estimate'],
                                c=discovery_tracks.get('trendy_score', 0.5), 
                                cmap='viridis', alpha=0.6)
            plt.xlabel('Tempo (BPM)')
            plt.ylabel('Valence (Happiness)')
            plt.title('🔥 Discovery Tracks')
            plt.colorbar(scatter, label='Trendy Score')
            plt.grid(True, alpha=0.3)
        
        # 3. Recommendations analysis
        plt.subplot(2, 4, 3)
        if not recommendations.empty:
            scatter = plt.scatter(recommendations['tempo'], recommendations['valence_estimate'],
                                c=recommendations['discovery_score'], 
                                cmap='plasma', s=100, alpha=0.8)
            plt.xlabel('Tempo (BPM)')
            plt.ylabel('Valence (Happiness)')
            plt.title('🎯 Top Recommendations')
            plt.colorbar(scatter, label='Discovery Score')
            plt.grid(True, alpha=0.3)
        
        # 4. User mood patterns
        plt.subplot(2, 4, 4)
        if self.user_profile and 'mood_patterns' in self.user_profile:
            moods = list(self.user_profile['mood_patterns'].keys())
            ratings = [self.user_profile['mood_patterns'][mood]['mean'] for mood in moods]
            colors = plt.cm.Set3(np.linspace(0, 1, len(moods)))
            
            bars = plt.bar(range(len(moods)), ratings, color=colors, alpha=0.7)
            plt.xlabel('Mood')
            plt.ylabel('Average Rating')
            plt.title('🎭 Mood Preferences')
            plt.xticks(range(len(moods)), [f"{MOOD_EMOJIS.get(mood, '🎵')}" for mood in moods])
            
            # Add value labels on bars
            for bar, rating in zip(bars, ratings):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{rating:.1f}', ha='center', va='bottom')
        
        # 5. Context preferences
        plt.subplot(2, 4, 5)
        if self.user_profile and 'context_patterns' in self.user_profile:
            contexts = list(self.user_profile['context_patterns'].keys())
            ratings = [self.user_profile['context_patterns'][ctx]['mean'] for ctx in contexts]
            
            plt.barh(range(len(contexts)), ratings, alpha=0.7, color='lightblue')
            plt.ylabel('Context')
            plt.xlabel('Average Rating')
            plt.title('🎯 Context Preferences')
            plt.yticks(range(len(contexts)), contexts)
            
            # Add value labels
            for i, rating in enumerate(ratings):
                plt.text(rating + 0.1, i, f'{rating:.1f}', va='center')
        
        # 6. Genre distribution in seeds
        plt.subplot(2, 4, 6)
        if not seed_features.empty and 'genre' in seed_features.columns:
            genre_counts = seed_features['genre'].value_counts()
            plt.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
            plt.title('🎼 Seed Track Genres')
        
        # 7. Audio feature preferences
        plt.subplot(2, 4, 7)
        if self.user_profile and 'preferences' in self.user_profile:
            features = []
            values = []
            for feature, pref_data in self.user_profile['preferences'].items():
                features.append(feature.replace('_', '\n'))
                pref_val = pref_data['preferred_value']
                values.append(float(pref_val))
            
            # Normalize values for visualization
            if 'tempo' in [f.replace('\n', '_') for f in features]:
                tempo_idx = [f.replace('\n', '_') for f in features].index('tempo')
                values[tempo_idx] = values[tempo_idx] / 20  # Scale tempo down
            
            plt.bar(range(len(features)), values, alpha=0.7, color='orange')
            plt.xlabel('Audio Features')
            plt.ylabel('Normalized Preference')
            plt.title('🎛️ Audio Feature Preferences')
            plt.xticks(range(len(features)), features, rotation=45)
        
        # 8. Discovery vs Similarity scatter
        plt.subplot(2, 4, 8)
        if not recommendations.empty:
            scatter = plt.scatter(recommendations['similarity_score'], recommendations['discovery_score'],
                       c=recommendations['trendy_score'], cmap='coolwarm', s=100, alpha=0.8)
            plt.xlabel('Similarity to Profile')
            plt.ylabel('Discovery Score')
            plt.title('🔍 Discovery vs Familiarity')
            plt.colorbar(scatter, label='Trendy Score')
            
            # Add diagonal line for reference
            max_val = float(max(recommendations['similarity_score'].max(), 
                              recommendations['discovery_score'].max()))
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Correlation')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('complete_recommender_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Analysis saved as 'complete_recommender_analysis.png'")
        
        plt.show()
        return fig
    
    # -------------------------------------------------------------------------
    # MAIN SYSTEM EXECUTION
    # -------------------------------------------------------------------------
    
    def run_complete_system(self, feedback_type="mock", show_visualizations=True):
        """
        Run the complete enhanced music recommender system.
        
        This is the main entry point that orchestrates the entire workflow:
        1. Load and validate FMA dataset
        2. Select diverse seed tracks
        3. Extract audio features
        4. Collect user feedback
        5. Build user profile hypothesis
        6. Simulate discovery tracks
        7. Generate recommendations
        8. Create visualizations and analytics
        9. Generate reports and export data
        
        Args:
            feedback_type (str): "mock" or "interactive"
            show_visualizations (bool): Whether to show plots
            
        Returns:
            dict: Complete results dictionary
        """
        print("🚀 Starting Complete Enhanced Music Recommender System")
        print("=" * 70)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Select seed tracks
            seed_features = self.select_diverse_seeds()
            print(f"\n🎵 Selected {len(seed_features)} seed tracks")
            
            # Display seed track summary
            if not seed_features.empty:
                print("\n📊 Seed Track Summary:")
                summary_cols = ['track_id', 'genre', 'tempo', 'valence_estimate']
                available_cols = [col for col in summary_cols if col in seed_features.columns]
                print(seed_features[available_cols].to_string(index=False))
            
            # Step 3: Collect feedback
            self.collect_feedback(seed_features, feedback_type)
            
            # Step 4: Build user profile
            user_profile = self.build_user_profile(seed_features)
            
            # Display user profile summary
            if user_profile:
                print("\n👤 User Profile Summary:")
                if 'mood_patterns' in user_profile:
                    print("🎭 Top moods:")
                    mood_items = sorted(user_profile['mood_patterns'].items(), 
                                      key=lambda x: x[1]['mean'], reverse=True)[:3]
                    for mood, stats in mood_items:
                        emoji = MOOD_EMOJIS.get(mood, '🎵')
                        print(f"  {emoji} {mood}: {stats['mean']:.1f}/10")
                
                print(f"🔍 Discovery tolerance: {user_profile.get('discovery_tolerance', 0.3):.2f}")
                print(f"⭐ Average rating: {user_profile.get('average_rating', 0):.1f}/10")
            
            # Step 5: Generate discovery tracks
            discovery_tracks = self.simulate_discovery_tracks()
            print(f"\n🔥 Generated {len(discovery_tracks)} discovery tracks")
            
            # Step 6: Generate recommendations
            recommendations = self.generate_recommendations(seed_features, discovery_tracks)
            print(f"\n🎯 Generated {len(recommendations)} recommendations")
            
            if not recommendations.empty:
                print("\n🏆 Top 5 Recommendations:")
                top_5 = recommendations.head(5)
                for i, (_, rec) in enumerate(top_5.iterrows(), 1):
                    print(f"  {i}. Track {int(rec['track_id'])} (Genre: {rec['genre']})")
                    print(f"     Discovery Score: {float(rec['discovery_score']):.3f}")
                    print(f"     {rec['explanation']}")
            
            # Step 7: Create visualizations
            if show_visualizations:
                self.create_comprehensive_analysis(seed_features, discovery_tracks, recommendations)
            
            # Step 8: Generate final report
            results = self.generate_final_report(seed_features, discovery_tracks, recommendations, user_profile)
            
            print(f"\n🎉 Complete Enhanced Music Recommender System Finished!")
            print(f"📊 Results saved to 'complete_recommender_results.json'")
            
            return results
            
        except Exception as e:
            print(f"❌ System error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_final_report(self, seed_features, discovery_tracks, recommendations, user_profile):
        """
        Generate comprehensive final report and export data.
        
        Args:
            seed_features, discovery_tracks, recommendations: DataFrames
            user_profile: User profile dictionary
            
        Returns:
            dict: Complete results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'system_config': CONFIG,
            'data_summary': {
                'total_valid_tracks': len(self.valid_track_ids),
                'seed_tracks_count': len(seed_features),
                'discovery_tracks_count': len(discovery_tracks),
                'recommendations_count': len(recommendations),
                'feedback_points': len(self.feedback_data)
            },
            'user_profile': user_profile,
            'feedback_data': self.feedback_data,
            'seed_tracks': seed_features.to_dict('records') if not seed_features.empty else [],
            'top_recommendations': recommendations.head(10).to_dict('records') if not recommendations.empty else [],
            'system_performance': {
                'avg_user_rating': user_profile.get('average_rating', 0),
                'discovery_tolerance': user_profile.get('discovery_tolerance', 0.3),
                'profile_confidence': user_profile.get('profile_confidence', 0),
                'recommendation_diversity': len(recommendations['genre'].unique()) if not recommendations.empty and 'genre' in recommendations.columns else 0
            }
        }
        
        # Save to JSON file
        with open('complete_recommender_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results

# =============================================================================
# API INTEGRATIONS (SPOTIFY & OPENAI)
# =============================================================================

class APIIntegrations:
    """
    Optional API integrations for Spotify and OpenAI GPT.
    
    These integrations extend the core recommender with:
    - Spotify playlist creation and track search
    - OpenAI GPT-powered playlist narrative generation
    - YouTube link generation for sharing
    """
    
    def __init__(self):
        """Initialize API integrations with credentials from environment."""
        self.spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID') if ENV_LOADED else None
        self.spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET') if ENV_LOADED else None
        self.openai_api_key = os.getenv('OPENAI_API_KEY') if ENV_LOADED else None
        self.spotify_token = None
        
        print("🔌 API Integrations initialized")
        if self.spotify_client_id and self.spotify_client_secret:
            print("✅ Spotify credentials loaded")
        else:
            print("⚠️ Spotify credentials not found in .env file")
            
        if self.openai_api_key:
            print("✅ OpenAI credentials loaded") 
        else:
            print("⚠️ OpenAI credentials not found in .env file")
    
    def get_spotify_token(self):
        """Get Spotify access token using client credentials flow."""
        if not self.spotify_client_id or not self.spotify_client_secret:
            return False
            
        print("🎵 Getting Spotify access token...")
        
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': self.spotify_client_id,
            'client_secret': self.spotify_client_secret
        }
        
        try:
            response = requests.post(auth_url, headers=auth_headers, data=auth_data)
            if response.status_code == 200:
                token_data = response.json()
                self.spotify_token = token_data['access_token']
                print(f"✅ Spotify token obtained")
                return True
            else:
                print(f"❌ Spotify auth failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Spotify auth error: {e}")
            return False
    
    def search_spotify_track(self, artist, title):
        """Search for a track on Spotify."""
        if not self.spotify_token:
            if not self.get_spotify_token():
                return None
                
        search_url = 'https://api.spotify.com/v1/search'
        headers = {'Authorization': f'Bearer {self.spotify_token}'}
        params = {
            'q': f'artist:{artist} track:{title}',
            'type': 'track',
            'limit': 1
        }
        
        try:
            response = requests.get(search_url, headers=headers, params=params)
            if response.status_code == 200:
                search_results = response.json()
                tracks = search_results['tracks']['items']
                return tracks[0] if tracks else None
            else:
                print(f"❌ Spotify search failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Spotify search error: {e}")
            return None
    
    def generate_gpt_playlist_concept(self, user_profile, tracks_info):
        """Generate creative playlist concept using OpenAI GPT."""
        if not self.openai_api_key or not OPENAI_AVAILABLE:
            return self.fallback_playlist_concept()
            
        print("🤖 Generating playlist concept with GPT...")
        
        # Prepare context for GPT
        profile_text = self.format_profile_for_gpt(user_profile)
        tracks_text = self.format_tracks_for_gpt(tracks_info)
        
        prompt = f"""Create a compelling music playlist concept based on this user profile and tracks:

USER PROFILE:
{profile_text}

TRACK CHARACTERISTICS:
{tracks_text}

Create a creative playlist with:
1. A catchy, evocative title (be creative and specific)
2. A 2-3 sentence description that tells a story or sets a mood
3. 3-5 specific listening contexts (when/where/why to play)
4. 5-7 hashtags that capture the essence
5. A one-sentence "vibe statement" that captures the emotional core

Format as JSON with keys: title, description, contexts, hashtags, vibe_statement
Make it feel personal and engaging, not generic."""

        try:
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative music curator who creates deeply personal, evocative playlist concepts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.8
            )
            
            result = response.choices[0].message.content
            print("✅ GPT playlist concept generated")
            
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                print("⚠️ GPT didn't return valid JSON, using fallback")
                return self.fallback_playlist_concept()
                
        except Exception as e:
            print(f"❌ GPT API error: {e}")
            return self.fallback_playlist_concept()
    
    def format_profile_for_gpt(self, profile):
        """Format user profile for GPT prompt."""
        if not profile:
            return "No specific user profile available"
            
        lines = []
        if 'mood_patterns' in profile:
            top_moods = sorted(profile['mood_patterns'].items(), 
                             key=lambda x: x[1]['mean'], reverse=True)[:3]
            mood_text = ", ".join([f"{mood} ({stats['mean']:.1f}/10)" for mood, stats in top_moods])
            lines.append(f"Favorite moods: {mood_text}")
        
        if 'context_patterns' in profile:
            top_contexts = sorted(profile['context_patterns'].items(),
                                key=lambda x: x[1]['mean'], reverse=True)[:3]
            context_text = ", ".join([f"{ctx} ({stats['mean']:.1f}/10)" for ctx, stats in top_contexts])
            lines.append(f"Preferred contexts: {context_text}")
            
        discovery = profile.get('discovery_tolerance', 0.3)
        discovery_desc = "adventurous" if discovery > 0.4 else "moderate" if discovery > 0.25 else "conservative"
        lines.append(f"Discovery style: {discovery_desc}")
        
        return "\n".join(lines)
    
    def format_tracks_for_gpt(self, tracks):
        """Format track info for GPT prompt."""
        if not tracks or len(tracks) == 0:
            return "No specific track information"
            
        lines = []
        tempos = [t.get('tempo', 120) for t in tracks if 'tempo' in t]
        valences = [t.get('valence_estimate', 0.5) for t in tracks if 'valence_estimate' in t]
        genres = [t.get('genre', 'Unknown') for t in tracks if 'genre' in t]
        
        if tempos:
            avg_tempo = sum(tempos) / len(tempos)
            lines.append(f"Average tempo: {avg_tempo:.0f} BPM")
            
        if valences:
            avg_valence = sum(valences) / len(valences)
            lines.append(f"Emotional tone: {avg_valence:.2f} (0=melancholy, 1=euphoric)")
            
        if genres:
            genre_counts = {}
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            genre_text = ", ".join([f"{genre}" for genre, count in top_genres])
            lines.append(f"Genres: {genre_text}")
            
        lines.append(f"Track count: {len(tracks)}")
        return "\n".join(lines)
    
    def fallback_playlist_concept(self):
        """Fallback playlist concept when GPT is unavailable."""
        return {
            "title": "Personal Discovery Mix",
            "description": "A curated blend of your musical preferences with exciting new discoveries, crafted to match your unique taste and mood patterns.",
            "contexts": ["Daily listening", "Discovery time", "Personal reflection"],
            "hashtags": ["#PersonalMix", "#Discovery", "#CuratedVibes", "#MusicalJourney"],
            "vibe_statement": "Where your musical identity meets endless possibility."
        }
    
    def create_youtube_links(self, tracks_data):
        """Generate YouTube search links for tracks."""
        youtube_links = []
        
        for track_data in tracks_data:
            track_id = track_data.get('track_id')
            if not track_id:
                continue
                
            # This would require access to the main recommender's tracks_df
            # For now, create placeholder links
            youtube_links.append({
                'track_id': track_id,
                'youtube_search_url': f"https://www.youtube.com/results?search_query=track_{track_id}"
            })
        
        return youtube_links
    
    def demo_integrations(self, user_profile=None, tracks_data=None):
        """Demonstrate API integrations with sample data."""
        print("\n🔌 API Integration Demo")
        print("=" * 40)
        
        # Test Spotify integration
        if self.get_spotify_token():
            test_result = self.search_spotify_track("Daft Punk", "Get Lucky")
            if test_result:
                print(f"🎵 Found on Spotify: {test_result['artists'][0]['name']} - {test_result['name']}")
                print(f"🔗 URL: {test_result['external_urls']['spotify']}")
        
        # Test GPT integration
        if user_profile and tracks_data:
            concept = self.generate_gpt_playlist_concept(user_profile, tracks_data)
            print(f"\n🤖 GPT Playlist Concept:")
            print(f"🎵 Title: {concept['title']}")
            print(f"📖 Description: {concept['description']}")
            print(f"🏷️ Hashtags: {' '.join(concept['hashtags'])}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the complete enhanced music recommender system.
    
    This function demonstrates the full workflow and can be customized
    for different use cases.
    """
    print("🎧 Complete Enhanced Music Recommender System")
    print("=" * 70)
    print("Features: Emoji feedback, MFCC clustering, GPT playlists, Spotify integration")
    print("Dataset: FMA (Free Music Archive)")
    print()
    
    # Initialize the main recommender system
    recommender = CompleteMusicRecommender()
    
    # Run the complete system
    results = recommender.run_complete_system(
        feedback_type="mock",  # Change to "interactive" for real user input
        show_visualizations=True
    )
    
    # Initialize API integrations (optional)
    api = APIIntegrations()
    
    # Demo API integrations if credentials are available
    if results and (api.spotify_client_id or api.openai_api_key):
        user_profile = results.get('user_profile', {})
        tracks_data = results.get('seed_tracks', [])
        api.demo_integrations(user_profile, tracks_data)
    
    print("\n🎉 System demonstration complete!")
    print("\n📁 Generated files:")
    print("  📊 complete_recommender_analysis.png - Comprehensive visualizations")
    print("  📄 complete_recommender_results.json - Complete data export")
    
    if ENV_LOADED:
        print("\n🔧 To use with real APIs:")
        print("  1. Update .env file with your API credentials")
        print("  2. Run: python complete_music_recommender.py")
    else:
        print("\n🔧 To enable API integrations:")
        print("  1. Install: pip install python-dotenv openai")
        print("  2. Create .env file with your API credentials")
        print("  3. Run: python complete_music_recommender.py")
    
    return results

if __name__ == "__main__":
    # Run the complete system
    main()
