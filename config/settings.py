"""ClipWhy V2 Configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

# ── YouTube API keys ──────────────────────────────────────────────────────────
YOUTUBE_API_KEYS = [
    v for k, v in sorted(os.environ.items())
    if k.startswith("YOUTUBE_API_KEY_") and v.strip()
]
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ── Creator criteria ──────────────────────────────────────────────────────────
MIN_SUBSCRIBERS = 10_000
MIN_LONG_VIDEOS = 10
MIN_SHORTS = 20
MIN_LONG_VIDEO_DURATION_SEC = 420   # 7 minutes
MAX_SHORT_DURATION_SEC = 60
ACTIVE_WITHIN_DAYS = 180

# ── Whisper ───────────────────────────────────────────────────────────────────
WHISPER_MODEL = "tiny"

# ── Repurpose verification ────────────────────────────────────────────────────
REPURPOSE_NUM_SHORTS = 5
REPURPOSE_NUM_LONGS = 5
REPURPOSE_MATCH_THRESHOLD = 80      # fuzzy match score
REPURPOSE_MIN_MATCHES = 2           # out of 5 Shorts

# ── Targets ───────────────────────────────────────────────────────────────────
CATEGORIES = ["tech", "education", "entertainment", "fitness", "commentary"]
TARGET_TOTAL = 1000                  # minimum goal, no cap

# ── Size brackets ─────────────────────────────────────────────────────────────
SIZE_BRACKETS = {
    "small": (10_000, 100_000),
    "medium": (100_000, 500_000),
    "large": (500_000, float("inf")),
}

# ── Country pre-filter ────────────────────────────────────────────────────────
ENGLISH_COUNTRIES = {"US", "GB", "AU", "CA", "NZ", "IE", "ZA"}
NON_ENGLISH_COUNTRIES = {
    "IN", "BD", "PK", "LK", "NP", "MM",               # South Asia
    "BR", "MX", "AR", "CL", "CO", "PE", "VE",          # Latin America
    "JP", "KR", "TW", "CN", "HK", "TH", "VN",          # East/SE Asia
    "ID", "PH", "MY", "SG",                             # SE Asia
    "SA", "AE", "EG", "TR", "IL", "IQ", "IR",           # Middle East
    "RU", "UA", "PL", "DE", "FR", "ES", "IT", "PT",     # Europe
    "NL", "GR", "RO", "HU", "CZ", "BG", "RS", "HR",    # Europe cont.
}

# ── YouTube Topic IDs (curated set, still working) ────────────────────────────
TOPIC_IDS = {
    "tech": "/m/07c1v",          # Technology
    "education": "/m/01k8wb",    # Knowledge
    "entertainment": "/m/02jjt", # Entertainment
    "fitness": "/m/0kt51",       # Health
    "commentary": "/m/098wr",    # Society
}

# Topic-to-category mapping is now in filters.py (TOPIC_WEIGHTS) with
# weighted scoring. Sports -> entertainment, Knowledge is weak signal.

# ── Rate limiting & parallelism ────────────────────────────────────────────────
DOWNLOAD_WORKERS = 4       # parallel yt-dlp threads for audio/caption downloads
PREFETCH_BATCH = 20        # channels to pre-download before processing

# ── Proxy (for yt-dlp on cloud VMs) ──────────────────────────────────────────
# YouTube blocks cloud IPs after ~2000 requests. Use Cloudflare WARP in Docker
# on the VM, then set: PROXY_URL=socks5://127.0.0.1:1080
PROXY_URL = os.environ.get("PROXY_URL", "")

# ── V1 Seeds (known good channels, same-channel Shorts+longs) ─────────────────
# Focused on PODCAST/INTERVIEW channels (highest repurpose rate ~20%)
V1_SEEDS = {
    "tech": [
        "Lex Fridman", "All-In Podcast", "Waveform Podcast MKBHD",
        "TWiT Tech Podcast Network", "Linus Tech Tips", "TechLinked",
        "Colin and Samir", "My First Million", "Starter Story",
        "Gary Vee", "The Verge", "Marques Brownlee",
        "a16z podcast", "Y Combinator", "This Week in Startups",
        "Big Think", "Chamath Palihapitiya", "Wired",
        "The Tim Ferriss Show", "Dave Lee on Investing",
        "Acquired Podcast", "No Priors AI podcast",
        "ColdFusion", "Fireship", "NetworkChuck",
        "The Diary Of A CEO", "Cleo Abram", "Riverside",
        "Sam Harris", "The Wall Street Journal",
        # Added: podcast-heavy tech channels
        "TechCrunch", "Machine Learning Street Talk",
        "Decoder with Nilay Patel", "Hard Fork New York Times",
        "Pivot Podcast Vox", "Lemonade Stand podcast",
        "Bloomberg Technology", "The AI Podcast NVIDIA",
        "Andrej Karpathy", "Two Minute Papers",
        "Patrick Boyle", "The Logan Bartlett Show",
        "20VC with Harry Stebbings", "Moonshots with Peter Diamandis",
        "Bankless podcast", "The Vergecast",
    ],
    "education": [
        "Jay Shetty Podcast", "TED", "Tom Bilyeu Impact Theory",
        "London Real", "Rich Roll Podcast", "Lewis Howes",
        "Mel Robbins", "Dr Rangan Chatterjee", "Jordan Harbinger Show",
        "After Skool", "Academy of Ideas", "Philosophize This",
        "Ali Abdaal", "Matt D'Avella", "Simon Sinek",
        "Eckhart Tolle", "Sadhguru", "Vsauce",
        "Veritasium", "Mark Rober", "Kurzgesagt",
        "3Blue1Brown", "Crash Course", "SmarterEveryDay",
        # Added: lecture/knowledge podcasts
        "The Knowledge Project Shane Parrish",
        "Naval Ravikant", "Tony Robbins", "Brendon Burchard",
        "The School of Greatness Lewis Howes",
        "Marie Forleo", "Ed Mylett Show",
        "GaryVee Audio Experience", "MedCram",
        "The Art of Manliness podcast", "Freakonomics Radio",
        "Hidden Brain NPR", "Radiolab podcast",
        "StarTalk Neil deGrasse Tyson", "Stuff You Should Know",
        "The Tim Ferriss Show", "Huberman Lab podcast",
    ],
    "entertainment": [
        "Club Shay Shay", "Hot Ones First We Feast",
        "Chicken Shop Date Amelia Dimoldenberg", "H3 Podcast",
        "Impaulsive Logan Paul", "Flagrant podcast",
        "GQ", "Kill Tony podcast", "Steve-O Wild Ride",
        "Soft White Underbelly", "Danny Jones Podcast",
        "Smartless podcast", "Call Her Daddy podcast",
        "The Breakfast Club Power 105.1", "Drink Champs",
        "Pardon My Take Barstool", "The Joe Budden Podcast",
        "Howie Mandel Does Stuff", "Matt Rife",
        "Dry Bar Comedy", "BigBoyTV", "Jubilee",
        "The Tonight Show Starring Jimmy Fallon",
        "The Graham Norton Show", "Bad Friends",
        # Interview/talk show channels
        "Dax Shepard Armchair Expert", "Conan O Brien Needs A Friend",
        "Jimmy Kimmel Live", "Seth Meyers Late Night",
        "Theo Von This Past Weekend", "Bobby Lee TigerBelly",
        "YMH Your Moms House podcast", "2 Bears 1 Cave",
        "Phil DeFranco Show", "Good Mythical Morning",
        # 2025-2026 breakouts (verified same-channel clippers)
        "Rotten Mango Stephanie Soo", "Smosh",
        "Trash Taste", "Chuckle Sandwich",
        "I've Had It Podcast", "The Shawn Ryan Show",
    ],
    "fitness": [
        "Andrew Huberman", "Dr Mike", "Jeff Nippard",
        "Mind Pump Show", "FoundMyFitness Rhonda Patrick",
        "Thomas DeLauer", "Athlean-X", "Peter Attia",
        "Mark Bell Power Project", "More Plates More Dates",
        "Dr Eric Berg", "Layne Norton Biolayne",
        "Renaissance Periodization", "The Model Health Show",
        "Ben Greenfield Life", "Dr Sten Ekberg",
        "Physionic", "Greg Doucette", "Sean Nalewanyj",
        "Noel Deyzel", "Jesse James West", "Doctor Mike",
        # Added: health/wellness podcasts
        "The Proof with Simon Hill", "Stan Efferding",
        "Stronger By Science podcast", "Barbell Medicine",
        "Dr John Campbell", "ZDoggMD",
        "MindBodyGreen podcast", "The Genius Life Max Lugavere",
        "Dhru Purohit Show", "The Doctor's Farmacy Mark Hyman",
        "Feel Better Live More Dr Chatterjee",
        "The Rich Roll Podcast", "Aubrey Marcus podcast",
        "Ben Patrick Knees Over Toes", "Chris Bumstead",
        # 2025-2026 trending health podcasts
        "The WHOOP Podcast", "Dr Mindy Pelz",
        "Dr Pradip Jamnadas", "Dr Jason Fung",
    ],
    "commentary": [
        "Piers Morgan Uncensored", "Patrick Bet-David Valuetainment",
        "Channel 5 Andrew Callaghan",
        "Triggernometry", "The Free Press", "UnHerd", "The Hill",
        "Breaking Points", "Megyn Kelly Show", "The Rubin Report",
        "Lonerbox", "Modern Wisdom Chris Williamson",
        "Ben Shapiro", "CNN", "NBC News", "VICE",
        "Bloomberg Originals", "ShxtsnGigs Podcast",
        # Commentary/news podcasts
        "Tucker Carlson", "The Rest Is History",
        "Jordan Peterson podcast", "Destiny debate",
        "The Daily Wire", "Pod Save America",
        "The Realignment podcast", "Bari Weiss Honestly",
        "Uncommon Knowledge Hoover", "The Glenn Show",
        "Reason Magazine podcast", "The Mehdi Hasan Show",
        "Democracy Now", "Secular Talk Kyle Kulinski",
        # 2025-2026 fastest growing (verified same-channel clippers)
        "MeidasTouch", "Benny Johnson",
        "Brian Tyler Cohen", "Candace Owens",
        "PBD Podcast Patrick Bet-David", "Turning Point USA",
        "Nick Shirley", "Adam Mockler",
    ],
}

# ── Search queries (60/category, podcast-focused for higher repurpose rate) ───
# Key insight from test run: podcast/interview channels repurpose at ~20%
# while tutorial/review channels are ~5%. Queries emphasize podcast format.
SEARCH_QUERIES = {
    "tech": [
        # Podcast/interview format (highest repurpose rate)
        "tech podcast clip", "tech podcast highlights",
        "technology podcast short", "tech interview clip",
        "startup podcast clip", "startup founder interview short",
        "AI podcast clip", "artificial intelligence interview",
        "developer podcast clip", "programming podcast short",
        "venture capital podcast clip", "VC interview short",
        "SaaS podcast clip", "software podcast highlight",
        "silicon valley podcast clip", "tech CEO interview",
        "data science podcast clip", "cybersecurity podcast short",
        "crypto podcast clip", "blockchain interview",
        "cloud computing podcast", "devops podcast clip",
        "machine learning podcast clip", "robotics interview short",
        "fintech podcast clip", "tech debate podcast",
        "indie hacker podcast clip", "bootstrapped founder interview",
        # Content-based queries
        "tech news highlight", "AI explained short",
        "startup tip clip", "coding tip short",
        "tech review clip", "gadget review short",
        "product review highlight", "app review short",
        "tech earnings analysis", "startup pitch highlight",
        "tech conference talk clip", "Apple news clip",
        "Google news clip", "tech layoffs discussion",
        "quantum computing explained", "semiconductor clip",
        "space tech short", "EV technology clip",
        "web development short", "open source clip",
        "tech founder story", "programming tutorial highlight",
        # Broad discovery
        "technology channel shorts", "tech YouTuber podcast",
        "best tech podcast clip", "popular tech interview",
        "trending tech short", "viral tech clip",
        "tech show highlights", "tech talk clip",
        "tech commentary short", "tech analysis clip",
    ],
    "education": [
        # Podcast/interview format
        "educational podcast clip", "education podcast short",
        "self improvement podcast clip", "personal development interview",
        "psychology podcast clip", "philosophy podcast short",
        "history podcast clip", "science podcast clip",
        "book podcast clip", "author interview short",
        "motivation podcast clip", "mindset podcast short",
        "leadership podcast clip", "business podcast clip",
        "productivity podcast clip", "learning podcast short",
        "knowledge podcast clip", "wisdom podcast clip",
        "thought leader interview", "expert interview clip",
        "intellectual conversation clip", "deep discussion short",
        "TED talk clip", "TEDx highlight",
        "lecture highlight clip", "professor explains short",
        "university talk clip", "academic discussion short",
        # Content-based queries
        "science explained short", "history explained clip",
        "economics explained short", "psychology explained clip",
        "philosophy explained short", "book summary clip",
        "study tips short", "life advice clip",
        "motivation speech short", "critical thinking clip",
        "documentary clip short", "interesting facts short",
        "master class clip", "online course highlight",
        "brain science short", "language learning clip",
        "writing tips short", "public speaking clip",
        "communication skills short", "career advice clip",
        # Broad discovery
        "educational channel shorts", "education YouTuber clip",
        "best education podcast", "popular knowledge short",
        "viral education clip", "learn something new short",
        "edutainment clip", "brainfood short",
        "teacher explains short", "mentor advice clip",
    ],
    "entertainment": [
        # Podcast/interview format
        "podcast clip viral", "podcast best moment",
        "podcast interview highlight", "podcast funny moment",
        "celebrity interview clip", "celebrity podcast short",
        "comedian podcast clip", "comedy podcast short",
        "talk show clip", "late night clip",
        "interview highlight viral", "actor interview clip",
        "musician interview short", "rapper interview clip",
        "athlete interview short", "sports podcast clip",
        "true crime podcast clip", "mystery podcast short",
        "dating podcast clip", "relationship podcast short",
        "hip hop podcast clip", "music podcast short",
        "gaming podcast clip", "esports interview",
        "food show clip", "cooking show highlight",
        "reality show clip", "drama podcast highlight",
        # Content-based queries
        "standup comedy clip", "roast clip short",
        "movie review clip", "TV show discussion short",
        "pop culture short", "celebrity story clip",
        "behind the scenes short", "award show moment",
        "red carpet clip", "bloopers clip",
        "anime discussion clip", "gaming moment viral",
        "boxing interview clip", "MMA podcast short",
        "football discussion clip", "basketball podcast short",
        "wrestling podcast clip", "improv comedy short",
        # Broad discovery
        "entertainment podcast clip", "best podcast moment",
        "trending podcast clip", "viral interview short",
        "funny clip short", "best of podcast",
        "podcast highlights compilation", "show clip viral",
        "reaction clip", "challenge clip viral",
        "gossip podcast clip", "variety show highlight",
    ],
    "fitness": [
        # Podcast/interview format (key to finding repurposers)
        "health podcast clip", "health podcast short",
        "fitness podcast clip", "fitness interview short",
        "doctor podcast clip", "medical podcast short",
        "wellness podcast clip", "nutrition podcast short",
        "biohacking podcast clip", "longevity podcast short",
        "mental health podcast clip", "therapy podcast short",
        "sports science podcast", "exercise science interview",
        "sleep expert podcast clip", "brain health podcast short",
        "gut health podcast clip", "hormone podcast short",
        "naturopath podcast clip", "functional medicine interview",
        "physiotherapy podcast clip", "chiropractor podcast short",
        "dietitian podcast clip", "nutritionist interview",
        "strength training podcast", "bodybuilding podcast clip",
        # Content-based queries
        "doctor explains short", "health tip short",
        "workout tip short", "nutrition clip",
        "gym motivation clip", "weight loss tip short",
        "supplement review clip", "sleep tip short",
        "meditation short", "breathing technique short",
        "cold plunge benefits clip", "sauna health short",
        "fasting explained clip", "protein explained short",
        "injury prevention clip", "recovery tip short",
        "stress management clip", "yoga tip short",
        # Broad discovery
        "fitness channel shorts", "health YouTuber podcast",
        "best health podcast clip", "popular fitness interview",
        "trending health short", "viral fitness clip",
        "medical expert short", "doctor interview clip",
        "wellness show clip", "health show highlight",
        "fitness expert interview", "personal trainer podcast",
    ],
    "commentary": [
        # Podcast/interview format
        "political podcast clip", "news podcast short",
        "commentary podcast clip", "opinion podcast short",
        "debate podcast clip", "discussion podcast short",
        "political interview clip", "journalist interview short",
        "analysis podcast clip", "current events podcast",
        "media podcast clip", "culture war podcast short",
        "economics podcast clip", "finance podcast short",
        "geopolitics podcast clip", "foreign policy interview",
        "legal analysis podcast", "law podcast clip",
        "history commentary podcast", "social commentary podcast",
        "libertarian podcast clip", "progressive podcast short",
        "conservative podcast clip", "centrist podcast short",
        # Content-based queries
        "news debate clip", "hot take clip",
        "social commentary short", "cultural debate clip",
        "media criticism short", "fact check clip",
        "economic analysis clip", "geopolitics short",
        "election analysis clip", "policy debate short",
        "free speech clip", "cancel culture short",
        "AI regulation clip", "housing crisis clip",
        "inflation explained short", "stock market clip",
        "legal analysis clip", "supreme court short",
        "military analysis short", "immigration debate clip",
        # Broad discovery
        "commentary channel shorts", "news YouTuber podcast",
        "best political podcast clip", "popular debate short",
        "trending opinion clip", "viral commentary short",
        "pundit interview clip", "analyst podcast short",
        "think tank clip", "foreign affairs podcast",
        "investigative journalism clip", "whistleblower short",
    ],
}

# Cross-category viral/trending queries (appended to EVERY category's search)
# Based on real search patterns people use to discover podcast clips
VIRAL_QUERIES = [
    "podcast clip viral", "podcast best moments 2025",
    "viral podcast moment", "podcast clips that went viral",
    "funniest podcast moments", "most shocking podcast moment",
    "best podcast highlights this week", "podcast shorts",
    "podcast hot takes", "podcast debate clips",
    "best podcast clips", "podcast highlights compilation",
]
