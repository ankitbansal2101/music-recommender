# 🎧 Enhanced Music Recommender System

A comprehensive music recommendation system that combines human feedback with AI-powered analysis. Users provide emoji reactions and natural language feedback to get personalized music recommendations through a beautiful web interface.

## 🎯 What This Project Does

This system creates personalized music recommendations by:

- **🎭 Collecting human feedback** through emoji reactions and natural language notes
- **🧮 Analyzing audio features** like tempo, valence, and MFCC coefficients  
- **🧠 Building user profiles** from feedback patterns and preferences
- **🤖 Generating AI playlists** with GPT-powered narratives
- **🎵 Integrating with Spotify** for real playlist creation and sharing

**Key Features:**
- Interactive web interface with audio playback
- Emoji-based mood feedback collection
- Real-time recommendation generation
- Spotify and OpenAI API integration
- Export results as JSON

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- ~7.2GB disk space for music dataset

### 1. Download Required Dataset

Download the FMA (Free Music Archive) dataset:

**Metadata (Required):**
```bash
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip
```

**Audio Files (Required):**
```bash
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip
```

This gives you 8,000 tracks (30-second clips each).

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Environment File

Create a `.env` file in the project root:

```bash
# Spotify API Credentials (get from https://developer.spotify.com/)
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here

# OpenAI API Key (get from https://platform.openai.com/)
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Get API Credentials

**Spotify API:**
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Create a new app
3. Copy Client ID and Client Secret to `.env`

**OpenAI API:**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create account and get API key
3. Add to `.env` file

### 5. Run the Application

**Web Interface (Recommended):**
```bash
python app.py
```
Then open: http://localhost:5000

**Command Line Version:**
```bash
python complete_music_recommender.py
```

---

## 📁 Project Structure

```
reccomender/
├── fma_small/                          # Audio files (8000 tracks)
│   ├── 000/                           # Organized in subdirectories
│   ├── 001/
│   └── ...
├── fma_metadata/                       # Track metadata and features  
│   ├── tracks.csv                     # Main track data
│   ├── genres.csv                     # Genre information
│   └── features.csv                   # Audio features
├── templates/                          # Flask web templates
├── static/                             # Web assets
├── app.py                              # Flask web application
├── complete_music_recommender.py       # Core recommendation engine  
├── .env                                # API credentials (create this!)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🌐 How to Use

### Web Interface:
1. **Configure** - Set your preferences and enable APIs
2. **Listen** - System selects diverse seed tracks 
3. **React** - Give feedback with emojis and comments
4. **Discover** - Get personalized recommendations
5. **Export** - Create Spotify playlists or download results

### Features Available:
- **Audio playback** with built-in HTML5 player
- **Emoji feedback** (😍 Love, 😌 Good, 😐 Meh, 👎 Hate)
- **Natural language notes** for detailed feedback
- **Real-time progress** tracking during processing
- **Spotify integration** to find and create playlists
- **AI playlist concepts** with GPT-generated descriptions
- **Export functionality** to save complete results

---

## 🛠 Troubleshooting

**Dataset Issues:**
```bash
# Alternative download method:
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

# If extraction fails:
7z x fma_metadata.zip
7z x fma_small.zip
```

**Audio Playback Issues:**
- Ensure FMA files are in correct structure: `fma_small/000/000001.mp3`
- Check file permissions for Flask serving
- Some audio files may have header issues (normal, logged as warnings)

**API Problems:**
- Verify credentials in `.env` file
- Test loading: `python -c "import os; print(os.getenv('SPOTIFY_CLIENT_ID'))"`
- OpenAI requires billing setup for API usage

**Web Interface:**
- Check Flask runs on http://127.0.0.1:5000
- Try different browsers or incognito mode
- Ensure port 5000 isn't blocked by firewall

---

## 📚 Dataset Citation


**Dataset Details:**
- **Source:** [FMA GitHub Repository](https://github.com/mdeff/fma)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Size:** 106,574 tracks total, 8,000 in small dataset
- **Format:** 30-second MP3 clips, 16 kHz, mono



*Built for music lovers who want truly personalized recommendations.*