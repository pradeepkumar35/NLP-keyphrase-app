"""
Comprehensive fallback domain classifier for when both zero-shot and keyword-based methods fail.
This implements a rule-based classifier with high-precision patterns for all domains.
Focuses on the 20% of terms that cover 80% of domain-specific content (Pareto principle).
"""

import re
from typing import Dict, List, Tuple, Optional

class DomainFallbackClassifier:
    """
    A rule-based fallback classifier for domain detection when other methods fail.
    Uses high-precision patterns and context clues to identify domains with high confidence.
    Implements the 80/20 rule by focusing on the most distinctive patterns for each domain.
    """
    
    def __init__(self):
        """Initialize the fallback classifier with domain-specific patterns."""
        # High-precision patterns for each domain (regex patterns that strongly indicate a specific domain)
        # These are the 20% of patterns that will identify 80% of domain-specific content
        self.domain_patterns = {
            # TECHNOLOGY
            "technology": [
                r'\b(?:software|hardware|app|application)\s+(?:development|engineer|developer|programming|code)\b',
                r'\b(?:artificial intelligence|machine learning|deep learning|neural network)\s+(?:model|algorithm|system|application)\b',
                r'\b(?:data|cloud|server|network|internet)\s+(?:security|infrastructure|architecture|protocol|service)\b',
                r'\b(?:mobile|web|desktop|cross-platform)\s+(?:app|application|development|interface|experience)\b',
                r'\b(?:tech|technology|digital|IT)\s+(?:company|industry|sector|giant|startup|innovation)\b'
            ],
            
            # BUSINESS
            "business": [
                r'\b(?:company|corporation|firm|enterprise)\s+(?:announced|reported|launched|acquired|merged)\b',
                r'\b(?:market|stock|share|investor|shareholder)\s+(?:value|price|growth|decline|performance|return)\b',
                r'\b(?:revenue|profit|earnings|sales|income)\s+(?:growth|decline|increase|decrease|report|quarter|year)\b',
                r'\b(?:CEO|executive|board|management|leadership)\s+(?:team|decision|strategy|vision|announced|stated)\b',
                r'\b(?:merger|acquisition|partnership|deal|agreement)\s+(?:between|with|valued at|worth|announced|completed)\b'
            ],
            
            # HEALTH
            "health": [
                r'\b(?:medical|clinical|health|healthcare|patient)\s+(?:treatment|care|procedure|outcome|record|data)\b',
                r'\b(?:disease|condition|disorder|syndrome|illness)\s+(?:symptoms|diagnosis|treatment|prevention|management)\b',
                r'\b(?:doctor|physician|surgeon|specialist|nurse)\s+(?:recommended|prescribed|diagnosed|treated|examined)\b',
                r'\b(?:drug|medication|vaccine|therapy|treatment)\s+(?:approved|developed|prescribed|administered|effective)\b',
                r'\b(?:study|research|trial|experiment|analysis)\s+(?:showed|found|suggested|indicated|demonstrated)\s+(?:health|medical|clinical)\b'
            ],
            
            # POLITICS
            "politics": [
                r'\b(?:president|senator|congressman|representative|governor)\s+(?:said|announced|proposed|signed|vetoed)\b',
                r'\b(?:election|campaign|vote|ballot|poll)\s+(?:result|outcome|turnout|system|fraud|integrity)\b',
                r'\b(?:bill|legislation|law|policy|regulation)\s+(?:passed|approved|signed|proposed|introduced|amended)\b',
                r'\b(?:democrat|republican|liberal|conservative|progressive)\s+(?:party|lawmaker|voter|base|agenda|platform)\b',
                r'\b(?:government|administration|cabinet|congress|parliament)\s+(?:official|decision|policy|action|response)\b'
            ],
            
            # SPORTS
            "sports": [
                r'\b(?:team|player|coach|athlete|roster)\s+(?:won|lost|defeated|beat|signed|traded|drafted)\b',
                r'\b(?:game|match|tournament|championship|series)\s+(?:victory|defeat|win|loss|title|trophy)\b',
                r'\b(?:score|point|goal|touchdown|basket|run)\s+(?:in the|during the|late in|early in|to win|to tie)\b',
                r'\b(?:season|playoff|league|division|conference)\s+(?:record|standing|title|championship|performance)\b',
                r'\b(?:injury|contract|trade|draft|free agent|salary cap)\s+(?:report|update|news|deal|agreement|situation)\b'
            ],
            
            # ENTERTAINMENT
            "entertainment": [
                r'\b(?:movie|film|show|series|episode)\s+(?:premiered|debuted|released|directed|produced|starred)\b',
                r'\b(?:actor|actress|director|producer|celebrity)\s+(?:starred|appeared|played|portrayed|directed|produced)\b',
                r'\b(?:award|oscar|emmy|grammy|golden globe)\s+(?:winner|nominee|ceremony|nomination|category)\b',
                r'\b(?:box office|rating|review|critic|audience)\s+(?:success|failure|hit|flop|reception|response)\b',
                r'\b(?:music|song|album|track|artist)\s+(?:released|debuted|topped|chart|billboard|streaming)\b'
            ],
            
            # SCIENCE
            "science": [
                r'\b(?:scientist|researcher|study|experiment|discovery)\s+(?:found|discovered|observed|demonstrated|published)\b',
                r'\b(?:research|study|analysis|experiment|investigation)\s+(?:published in|journal|peer-reviewed|scientific|academic)\b',
                r'\b(?:theory|hypothesis|model|concept|principle)\s+(?:suggests|predicts|explains|describes|proposes)\b',
                r'\b(?:data|evidence|result|finding|observation)\s+(?:suggests|indicates|shows|demonstrates|confirms)\b',
                r'\b(?:physics|chemistry|biology|astronomy|geology)\s+(?:principle|law|theory|concept|phenomenon)\b'
            ],
            
            # ENVIRONMENT
            "environment": [
                r'\b(?:climate change|global warming|greenhouse gas|carbon emission|fossil fuel)\s+(?:impact|effect|threat|crisis|action)\b',
                r'\b(?:environmental|conservation|preservation|protection|sustainability)\s+(?:effort|initiative|program|policy|regulation)\b',
                r'\b(?:renewable|clean|green|sustainable|alternative)\s+(?:energy|power|electricity|source|technology)\b',
                r'\b(?:pollution|contamination|waste|plastic|emission)\s+(?:level|reduction|management|problem|crisis)\b',
                r'\b(?:ecosystem|biodiversity|habitat|species|wildlife)\s+(?:protection|conservation|loss|threatened|endangered)\b'
            ],
            
            # WORLD NEWS
            "world": [
                r'\b(?:country|nation|government|state|regime)\s+(?:announced|declared|imposed|condemned|responded)\b',
                r'\b(?:international|global|diplomatic|bilateral|multilateral)\s+(?:relation|agreement|cooperation|tension|conflict)\b',
                r'\b(?:foreign|diplomatic|international|global|geopolitical)\s+(?:policy|affair|crisis|conflict|tension)\b',
                r'\b(?:war|conflict|crisis|tension|dispute)\s+(?:between|involving|escalated|resolved|ongoing)\b',
                r'\b(?:UN|EU|NATO|WHO|World Bank)\s+(?:resolution|decision|statement|report|meeting|summit)\b'
            ],
            
            # EDUCATION
            "education": [
                r'\b(?:student|teacher|professor|educator|faculty)\s+(?:learning|teaching|performance|achievement|development)\b',
                r'\b(?:school|university|college|campus|institution)\s+(?:program|course|curriculum|degree|education)\b',
                r'\b(?:education|educational|academic|learning|teaching)\s+(?:system|policy|reform|standard|quality)\b',
                r'\b(?:classroom|course|curriculum|program|instruction)\s+(?:design|development|implementation|assessment|evaluation)\b',
                r'\b(?:test|exam|assessment|evaluation|grade)\s+(?:score|result|performance|standard|improvement)\b'
            ],
            
            # FOOD
            "food": [
                r'\b(?:recipe|dish|meal|cuisine|food)\s+(?:preparation|cooking|serving|tasting|pairing)\b',
                r'\b(?:ingredient|spice|herb|seasoning|flavor)\s+(?:fresh|dried|ground|chopped|minced|mixed)\b',
                r'\b(?:cook|bake|roast|grill|fry|simmer|boil)\s+(?:until|for|over|with|in)\s+(?:\d+|low|medium|high)\b',
                r'\b(?:restaurant|chef|kitchen|dining|culinary)\s+(?:experience|scene|trend|technique|tradition)\b',
                r'\b(?:cup|tablespoon|teaspoon|ounce|pound|gram)\s+(?:of)?\s+(?:\w+)\s+(?:chopped|minced|diced|sliced|grated)\b'
            ],
            
            # TRAVEL
            "travel": [
                r'\b(?:travel|trip|journey|vacation|holiday)\s+(?:destination|experience|planning|booking|package)\b',
                r'\b(?:hotel|resort|accommodation|lodging|stay)\s+(?:luxury|budget|boutique|all-inclusive|reservation)\b',
                r'\b(?:flight|airline|airport|airfare|ticket)\s+(?:booking|reservation|cancellation|delay|schedule)\b',
                r'\b(?:tourist|traveler|visitor|guest|passenger)\s+(?:attraction|experience|guide|visa|destination)\b',
                r'\b(?:destination|location|place|spot|attraction)\s+(?:popular|hidden|must-see|off-the-beaten-path|bucket-list)\b'
            ],
            
            # AUTOMOTIVE
            "automotive": [
                r'\b(?:car|vehicle|automobile|model|SUV|sedan|truck)\s+(?:manufacturer|brand|company|maker|production)\b',
                r'\b(?:engine|motor|powertrain|transmission|drivetrain)\s+(?:performance|power|efficiency|technology|system)\b',
                r'\b(?:electric|hybrid|gas|diesel|fuel)\s+(?:vehicle|car|model|engine|efficiency|economy)\b',
                r'\b(?:safety|performance|efficiency|reliability|technology)\s+(?:feature|system|rating|standard|improvement)\b',
                r'\b(?:driving|driver|ride|handling|comfort)\s+(?:experience|assistance|mode|quality|technology)\b'
            ],
            
            # REAL ESTATE
            "real estate": [
                r'\b(?:home|house|property|real estate|housing)\s+(?:market|price|value|sale|purchase)\b',
                r'\b(?:buyer|seller|agent|broker|realtor)\s+(?:market|offer|negotiation|commission|representation)\b',
                r'\b(?:mortgage|loan|financing|interest rate|down payment)\s+(?:approval|application|term|rate|option)\b',
                r'\b(?:listing|sale|purchase|transaction|closing)\s+(?:price|agreement|process|date|cost)\b',
                r'\b(?:commercial|residential|industrial|retail|office)\s+(?:property|space|building|development|market)\b'
            ],
            
            # CYBERSECURITY
            "cybersecurity": [
                r'\b(?:cyber|security|data|network|system)\s+(?:attack|breach|threat|vulnerability|protection)\b',
                r'\b(?:hacker|attacker|threat actor|cybercriminal|adversary)\s+(?:targeted|compromised|exploited|accessed|stole)\b',
                r'\b(?:malware|ransomware|virus|trojan|spyware)\s+(?:attack|infection|detection|prevention|removal)\b',
                r'\b(?:password|authentication|encryption|firewall|VPN)\s+(?:protection|security|system|solution|technology)\b',
                r'\b(?:data|information|system|network|infrastructure)\s+(?:protection|security|privacy|breach|vulnerability)\b'
            ],
            
            # ARTIFICIAL INTELLIGENCE
            "artificial intelligence": [
                r'\b(?:artificial intelligence|AI|machine learning|ML|deep learning)\s+(?:model|system|algorithm|application|technology)\b',
                r'\b(?:neural network|algorithm|model|system|framework)\s+(?:training|inference|prediction|classification|performance)\b',
                r'\b(?:data|training data|dataset|input|output)\s+(?:processing|analysis|preparation|augmentation|validation)\b',
                r'\b(?:natural language processing|NLP|computer vision|CV|speech recognition)\s+(?:model|system|application|technology)\b',
                r'\b(?:AI|artificial intelligence)\s+(?:ethics|bias|fairness|transparency|accountability|regulation)\b'
            ],
            
            # SPACE
            "space": [
                r'\b(?:space|mission|launch|rocket|spacecraft)\s+(?:exploration|program|agency|company|technology)\b',
                r'\b(?:NASA|SpaceX|ESA|Roscosmos|Blue Origin)\s+(?:mission|launch|program|spacecraft|astronaut)\b',
                r'\b(?:planet|moon|Mars|Jupiter|Saturn|asteroid|comet)\s+(?:exploration|mission|surface|atmosphere|orbit)\b',
                r'\b(?:astronaut|cosmonaut|space traveler|crew|commander)\s+(?:mission|aboard|spacewalk|returned|launched)\b',
                r'\b(?:telescope|observatory|satellite|probe|rover)\s+(?:image|data|observation|discovery|exploration)\b'
            ],
            
            # AGRICULTURE
            "agriculture": [
                r'\b(?:farm|farming|agricultural|crop|livestock)\s+(?:production|yield|management|practice|system)\b',
                r'\b(?:farmer|grower|producer|rancher|breeder)\s+(?:growing|producing|harvesting|raising|breeding)\b',
                r'\b(?:crop|harvest|yield|production|cultivation)\s+(?:season|rotation|protection|insurance|subsidy)\b',
                r'\b(?:soil|water|nutrient|fertilizer|pesticide)\s+(?:management|conservation|quality|application|runoff)\b',
                r'\b(?:sustainable|organic|conventional|precision|regenerative)\s+(?:agriculture|farming|practice|method|technique)\b'
            ],
            
            # MENTAL HEALTH
            "mental health": [
                r'\b(?:mental health|psychological|psychiatric|emotional|behavioral)\s+(?:condition|disorder|illness|wellbeing|treatment)\b',
                r'\b(?:therapy|counseling|treatment|intervention|support)\s+(?:for|of)\s+(?:mental|psychological|emotional|behavioral)\b',
                r'\b(?:depression|anxiety|trauma|PTSD|bipolar|schizophrenia)\s+(?:disorder|symptoms|diagnosis|treatment|therapy)\b',
                r'\b(?:psychiatrist|psychologist|therapist|counselor|mental health professional)\s+(?:treatment|diagnosis|therapy|practice|approach)\b',
                r'\b(?:mental health|psychological|emotional)\s+(?:awareness|stigma|support|service|resource|crisis)\b'
            ]
        }
        
        # Domain-specific entity patterns (organizations, publications, events that strongly indicate a domain)
        # These are the 20% of entities that will identify 80% of domain-specific content
        self.domain_entities = {
            "technology": [
                "Apple", "Google", "Microsoft", "Amazon", "Facebook", "Meta", "Tesla", "Intel", "AMD", "NVIDIA",
                "IBM", "Oracle", "Cisco", "Samsung", "Huawei", "TikTok", "Twitter", "LinkedIn", "GitHub", "Stack Overflow",
                "AI", "ML", "API", "IoT", "5G", "Cloud", "SaaS", "DevOps", "Blockchain", "Cryptocurrency"
            ],
            
            # BUSINESS
            "business": [
                "Wall Street", "NYSE", "NASDAQ", "Dow Jones", "S&P 500", "Fortune 500", "Forbes", "Bloomberg", "CNBC", "Financial Times",
                "CEO", "CFO", "COO", "CTO", "IPO", "M&A", "ROI", "B2B", "B2C", "GDP",
                "Federal Reserve", "Treasury", "SEC", "IMF", "World Bank", "WTO", "OPEC", "Berkshire Hathaway", "JPMorgan", "Goldman Sachs"
            ],
            
            # HEALTH
            "health": [
                "WHO", "CDC", "FDA", "NIH", "AMA", "NHS", "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins", "WebMD",
                "COVID-19", "Coronavirus", "Pandemic", "Vaccine", "Pfizer", "Moderna", "Johnson & Johnson", "Merck", "Novartis", "Roche",
                "Medicare", "Medicaid", "Affordable Care Act", "Obamacare", "Health Insurance", "Big Pharma", "Mental Health", "Wellness", "Fitness", "Nutrition"
            ],
            
            # POLITICS
            "politics": [
                "White House", "Congress", "Senate", "House of Representatives", "Supreme Court", "Capitol Hill", "Pentagon", "State Department", "United Nations", "NATO",
                "Democrat", "Republican", "GOP", "Liberal", "Conservative", "Progressive", "Election", "Campaign", "Poll", "Ballot",
                "President", "Vice President", "Secretary of State", "Attorney General", "Speaker of the House", "Majority Leader", "Minority Leader", "Filibuster", "Gerrymander", "Lobbyist"
            ],
            
            # SPORTS
            "sports": [
                "NFL", "NBA", "MLB", "NHL", "MLS", "FIFA", "UEFA", "Olympics", "Super Bowl", "World Cup",
                "ESPN", "Sports Illustrated", "Bleacher Report", "Draft", "Free Agency", "All-Star", "MVP", "Championship", "Playoff", "Tournament",
                "Manchester United", "Real Madrid", "Barcelona", "Lakers", "Yankees", "Cowboys", "Patriots", "LeBron James", "Tom Brady", "Serena Williams"
            ],
            
            # ENTERTAINMENT
            "entertainment": [
                "Hollywood", "Netflix", "Disney", "HBO", "Warner Bros", "Universal", "Paramount", "Sony Pictures", "Marvel", "DC",
                "Oscar", "Emmy", "Grammy", "Golden Globe", "Cannes", "Sundance", "Billboard", "Box Office", "Streaming", "Prime Time",
                "Celebrity", "Actor", "Actress", "Director", "Producer", "Star", "Movie", "Film", "TV Show", "Series"
            ],
            
            # SCIENCE
            "science": [
                "NASA", "CERN", "NIH", "NSF", "Nature", "Science", "Scientific American", "AAAS", "Royal Society", "Max Planck Institute",
                "Physics", "Chemistry", "Biology", "Astronomy", "Geology", "Neuroscience", "Genetics", "Quantum", "Particle", "Molecule",
                "Research", "Experiment", "Laboratory", "Hypothesis", "Theory", "Peer Review", "Publication", "Journal", "Conference", "Symposium"
            ],
            
            # ENVIRONMENT
            "environment": [
                "EPA", "IPCC", "Greenpeace", "Sierra Club", "WWF", "UNEP", "National Geographic", "Climate Change", "Global Warming", "Carbon Footprint",
                "Renewable Energy", "Solar Power", "Wind Power", "Fossil Fuel", "Greenhouse Gas", "Emission", "Pollution", "Conservation", "Sustainability", "Biodiversity",
                "Paris Agreement", "Kyoto Protocol", "COP26", "Green New Deal", "Carbon Tax", "Cap and Trade", "Recycling", "Plastic Waste", "Deforestation", "Endangered Species"
            ],
            
            # WORLD NEWS
            "world": [
                "United Nations", "European Union", "NATO", "G7", "G20", "World Bank", "IMF", "WHO", "WTO", "OPEC",
                "Foreign Policy", "Diplomacy", "International Relations", "Global Affairs", "Geopolitics", "Embassy", "Ambassador", "Treaty", "Summit", "Bilateral",
                "BBC World", "CNN International", "Al Jazeera", "Reuters", "Associated Press", "AFP", "Foreign Correspondent", "International", "Global", "Worldwide"
            ],
            
            # EDUCATION
            "education": [
                "Department of Education", "Board of Education", "School District", "University", "College", "Campus", "Faculty", "Student", "Teacher", "Professor",
                "Curriculum", "Syllabus", "Course", "Degree", "Diploma", "Certificate", "Accreditation", "SAT", "ACT", "GRE",
                "Harvard", "Stanford", "MIT", "Oxford", "Cambridge", "Yale", "Princeton", "Berkeley", "Public School", "Private School"
            ],
            
            # FOOD
            "food": [
                "Food Network", "Bon Appétit", "Epicurious", "Michelin", "James Beard", "Zagat", "Yelp", "TripAdvisor", "OpenTable", "DoorDash",
                "Chef", "Restaurant", "Cuisine", "Recipe", "Ingredient", "Menu", "Dish", "Meal", "Cooking", "Baking",
                "Vegetarian", "Vegan", "Gluten-Free", "Organic", "Farm-to-Table", "Sustainable", "Foodie", "Culinary", "Gourmet", "Gastronomy"
            ],
            
            # TRAVEL
            "travel": [
                "Expedia", "Booking.com", "Airbnb", "TripAdvisor", "Lonely Planet", "Travel + Leisure", "Condé Nast Traveler", "National Geographic Traveler", "Rick Steves", "Fodor's",
                "Hotel", "Resort", "Accommodation", "Flight", "Airline", "Airport", "Cruise", "Tour", "Vacation", "Holiday",
                "Destination", "Tourism", "Tourist", "Traveler", "Passport", "Visa", "Itinerary", "Excursion", "Adventure", "Sightseeing"
            ],
            
            # AUTOMOTIVE
            "automotive": [
                "Toyota", "Volkswagen", "Ford", "GM", "Tesla", "BMW", "Mercedes-Benz", "Honda", "Hyundai", "Kia",
                "Car and Driver", "Motor Trend", "Automotive News", "Kelley Blue Book", "Edmunds", "AutoTrader", "J.D. Power", "NHTSA", "EPA", "IIHS",
                "Electric Vehicle", "Hybrid", "SUV", "Sedan", "Truck", "Autonomous", "Self-Driving", "Horsepower", "Torque", "MPG"
            ],
            
            # REAL ESTATE
            "real estate": [
                "Zillow", "Redfin", "Realtor.com", "Trulia", "Century 21", "RE/MAX", "Keller Williams", "Coldwell Banker", "Sotheby's", "Berkshire Hathaway HomeServices",
                "MLS", "NAR", "NAHB", "Fannie Mae", "Freddie Mac", "HUD", "FHA", "VA Loan", "Mortgage", "Down Payment",
                "Real Estate Agent", "Broker", "Realtor", "Property", "Home", "House", "Condo", "Apartment", "Commercial", "Residential"
            ],
            
            # CYBERSECURITY
            "cybersecurity": [
                "CISA", "NSA", "FBI Cyber Division", "Interpol", "Europol", "NIST", "ISO", "SANS Institute", "Black Hat", "DEF CON",
                "Firewall", "Antivirus", "VPN", "Encryption", "Authentication", "Phishing", "Malware", "Ransomware", "DDoS", "Zero-day",
                "Cyber Attack", "Data Breach", "Vulnerability", "Patch", "Security", "Privacy", "Hacker", "Threat Actor", "Penetration Testing", "Security Operations Center"
            ],
            
            # ARTIFICIAL INTELLIGENCE
            "artificial intelligence": [
                "OpenAI", "DeepMind", "NVIDIA AI", "IBM Watson", "Google AI", "Microsoft AI", "Meta AI", "Anthropic", "Hugging Face", "PyTorch",
                "TensorFlow", "GPT", "BERT", "DALL-E", "Stable Diffusion", "Midjourney", "ChatGPT", "LLM", "Neural Network", "Deep Learning",
                "Machine Learning", "NLP", "Computer Vision", "Reinforcement Learning", "AI Ethics", "AGI", "Transformer", "Diffusion Model", "Generative AI", "AI Alignment"
            ],
            
            # SPACE
            "space": [
                "NASA", "SpaceX", "Blue Origin", "ESA", "Roscosmos", "ISRO", "JAXA", "Virgin Galactic", "Rocket Lab", "ULA",
                "ISS", "Hubble", "James Webb", "Perseverance", "Curiosity", "Artemis", "Apollo", "Falcon 9", "Starship", "SLS",
                "Mars", "Moon", "Jupiter", "Saturn", "Asteroid", "Comet", "Galaxy", "Exoplanet", "Black Hole", "Nebula"
            ],
            
            # AGRICULTURE
            "agriculture": [
                "USDA", "FAO", "Farm Bureau", "4-H", "FFA", "Land Grant University", "Cooperative Extension", "Monsanto", "Bayer", "John Deere",
                "Farm", "Crop", "Livestock", "Harvest", "Irrigation", "Fertilizer", "Pesticide", "GMO", "Organic", "Sustainable",
                "Corn", "Wheat", "Soybean", "Rice", "Cotton", "Cattle", "Dairy", "Poultry", "Farm Bill", "Subsidy"
            ],
            
            # MENTAL HEALTH
            "mental health": [
                "NIMH", "APA", "WHO Mental Health", "NAMI", "SAMHSA", "Mental Health America", "Psychology Today", "Crisis Text Line", "National Suicide Prevention Lifeline", "988",
                "Depression", "Anxiety", "PTSD", "Bipolar", "Schizophrenia", "OCD", "ADHD", "Autism", "Therapy", "Counseling",
                "Psychiatrist", "Psychologist", "Therapist", "Mental Illness", "Mental Wellbeing", "Stigma", "Awareness", "Self-Care", "Mindfulness", "Resilience"
            ]
        }
        
        # Domain-specific terminology frequency thresholds
        # These are calibrated based on how many terms need to appear to strongly indicate a domain
        self.domain_term_thresholds = {
            "technology": 4,
            "business": 4,
            "health": 4,
            "politics": 4,
            "sports": 3,
            "entertainment": 3,
            "science": 4,
            "environment": 3,
            "world": 4,
            "education": 4,
            "food": 3,
            "travel": 3,
            "automotive": 3,
            "real estate": 3,
            "cybersecurity": 3,
            "artificial intelligence": 3,
            "space": 3,
            "agriculture": 3,
            "mental health": 3
        }
        
        # Domain-specific terminology (common terms that appear frequently in domain-specific content)
        # These are the 20% of terms that will identify 80% of domain-specific content
        self.domain_terminology = {
            # TECHNOLOGY
            "technology": [
                "technology", "tech", "digital", "software", "hardware", "app", "device", "computer", "mobile", "internet",
                "data", "cloud", "AI", "algorithm", "code", "programming", "developer", "platform", "system", "network",
                "security", "cyber", "innovation", "startup", "gadget", "smart", "virtual", "interface", "user experience", "automation"
            ],
            
            # BUSINESS
            "business": [
                "business", "company", "market", "industry", "corporate", "firm", "enterprise", "startup", "revenue", "profit",
                "investment", "investor", "shareholder", "stock", "finance", "financial", "economy", "economic", "CEO", "executive",
                "management", "strategy", "growth", "expansion", "acquisition", "merger", "partnership", "client", "customer", "consumer"
            ],
            
            # HEALTH
            "health": [
                "health", "medical", "healthcare", "patient", "doctor", "hospital", "clinic", "treatment", "therapy", "medication",
                "disease", "condition", "symptom", "diagnosis", "prevention", "wellness", "care", "specialist", "surgery", "prescription",
                "drug", "pharmaceutical", "clinical", "research", "study", "trial", "vaccine", "immunity", "chronic", "acute"
            ],
            
            # POLITICS
            "politics": [
                "politics", "political", "government", "policy", "election", "campaign", "vote", "voter", "candidate", "president",
                "congress", "senate", "representative", "democrat", "republican", "liberal", "conservative", "legislation", "law", "bill",
                "administration", "official", "diplomat", "foreign policy", "domestic policy", "regulation", "reform", "party", "poll", "approval rating"
            ],
            
            # SPORTS
            "sports": [
                "sports", "game", "team", "player", "coach", "athlete", "championship", "tournament", "league", "season",
                "win", "loss", "score", "point", "goal", "match", "competition", "stadium", "arena", "fan",
                "draft", "trade", "contract", "injury", "performance", "record", "title", "MVP", "all-star", "playoff"
            ],
            
            # ENTERTAINMENT
            "entertainment": [
                "entertainment", "movie", "film", "show", "series", "episode", "actor", "actress", "director", "producer",
                "celebrity", "star", "Hollywood", "box office", "streaming", "music", "song", "album", "artist", "band",
                "concert", "tour", "performance", "award", "critic", "review", "rating", "audience", "fan", "premiere"
            ],
            
            # SCIENCE
            "science": [
                "science", "scientific", "research", "study", "experiment", "laboratory", "scientist", "researcher", "discovery", "innovation",
                "theory", "hypothesis", "data", "analysis", "evidence", "observation", "physics", "chemistry", "biology", "astronomy",
                "genetics", "molecule", "atom", "particle", "cell", "organism", "species", "evolution", "ecosystem", "climate"
            ],
            
            # ENVIRONMENT
            "environment": [
                "environment", "environmental", "climate", "climate change", "global warming", "carbon", "emission", "pollution", "renewable", "sustainable",
                "conservation", "ecosystem", "biodiversity", "species", "habitat", "wildlife", "forest", "ocean", "water", "air",
                "energy", "solar", "wind", "fossil fuel", "green", "recycling", "waste", "plastic", "protection", "preservation"
            ],
            
            # WORLD NEWS
            "world": [
                "international", "global", "world", "foreign", "country", "nation", "government", "leader", "president", "prime minister",
                "diplomacy", "diplomatic", "relations", "treaty", "agreement", "conflict", "crisis", "war", "peace", "security",
                "trade", "economy", "development", "aid", "refugee", "immigration", "border", "sanction", "alliance", "summit"
            ],
            
            # EDUCATION
            "education": [
                            # EDUCATION (continued)
                "education", "school", "university", "college", "student", "teacher", "professor", "classroom", "campus", "learning",
                "teaching", "academic", "curriculum", "course", "degree", "program", "study", "research", "grade", "exam",
                "test", "assignment", "scholarship", "tuition", "faculty", "administration", "board", "district", "literacy", "STEM"
            ],
            
            # FOOD
            "food": [
                "food", "recipe", "cooking", "baking", "chef", "restaurant", "cuisine", "dish", "meal", "ingredient",
                "flavor", "taste", "delicious", "kitchen", "dining", "menu", "appetizer", "entree", "dessert", "beverage",
                "vegetarian", "vegan", "organic", "fresh", "local", "seasonal", "gourmet", "culinary", "nutritious", "homemade"
            ],
            
            # TRAVEL
            "travel": [
                "travel", "trip", "vacation", "holiday", "destination", "tourism", "tourist", "hotel", "resort", "accommodation",
                "flight", "airline", "airport", "cruise", "tour", "adventure", "experience", "sightseeing", "attraction", "landmark",
                "international", "domestic", "passport", "visa", "itinerary", "booking", "reservation", "guide", "excursion", "journey"
            ],
            
            # AUTOMOTIVE
            "automotive": [
                "car", "vehicle", "automobile", "automotive", "driver", "driving", "engine", "motor", "transmission", "model",
                "brand", "manufacturer", "dealership", "sedan", "SUV", "truck", "electric", "hybrid", "gas", "diesel",
                "horsepower", "torque", "performance", "safety", "technology", "feature", "design", "interior", "exterior", "test drive"
            ],
            
            # REAL ESTATE
            "real estate": [
                "real estate", "property", "home", "house", "apartment", "condo", "townhouse", "residential", "commercial", "industrial",
                "buyer", "seller", "agent", "broker", "market", "listing", "sale", "purchase", "mortgage", "loan",
                "interest rate", "down payment", "closing", "inspection", "appraisal", "investment", "rental", "landlord", "tenant", "development"
            ],
            
            # CYBERSECURITY
            "cybersecurity": [
                "cybersecurity", "security", "cyber", "hack", "breach", "attack", "threat", "vulnerability", "malware", "ransomware",
                "phishing", "data", "protection", "encryption", "firewall", "authentication", "password", "privacy", "risk", "compliance",
                "defense", "detection", "response", "prevention", "incident", "forensics", "patch", "update", "secure", "compromise"
            ],
            
            # ARTIFICIAL INTELLIGENCE
            "artificial intelligence": [
                "artificial intelligence", "AI", "machine learning", "ML", "deep learning", "neural network", "algorithm", "model", "training", "data",
                "prediction", "classification", "recognition", "natural language processing", "NLP", "computer vision", "automation", "robotics", "intelligent", "smart",
                "GPT", "transformer", "supervised", "unsupervised", "reinforcement", "dataset", "feature", "parameter", "inference", "generative"
            ],
            
            # SPACE
            "space": [
                "space", "astronomy", "cosmos", "universe", "galaxy", "star", "planet", "moon", "Mars", "solar system",
                "rocket", "spacecraft", "satellite", "telescope", "astronaut", "mission", "launch", "orbit", "exploration", "discovery",
                "NASA", "SpaceX", "astronomical", "celestial", "cosmic", "gravitational", "interstellar", "extraterrestrial", "rover", "probe"
            ],
            
            # AGRICULTURE
            "agriculture": [
                "agriculture", "farming", "farm", "crop", "harvest", "livestock", "soil", "seed", "plant", "grow",
                "farmer", "agricultural", "cultivation", "irrigation", "fertilizer", "pesticide", "organic", "sustainable", "yield", "production",
                "rural", "field", "pasture", "grain", "dairy", "cattle", "poultry", "horticulture", "agribusiness", "food production"
            ],
            
            # MENTAL HEALTH
            "mental health": [
                "mental health", "psychological", "emotional", "behavioral", "psychiatric", "therapy", "counseling", "depression", "anxiety", "stress",
                "trauma", "disorder", "condition", "treatment", "support", "wellbeing", "wellness", "coping", "resilience", "mindfulness",
                "psychiatrist", "psychologist", "therapist", "counselor", "diagnosis", "symptom", "recovery", "self-care", "awareness", "stigma"
            ]
        }
    
    def detect_domain(self, text: str) -> Tuple[Optional[str], float]:
        """
        Detect the domain of the text using high-precision patterns and context clues.
        Implements the 80/20 rule by focusing on the most distinctive patterns for each domain.
        
        Args:
            text: The text to analyze
            
        Returns:
            Tuple of (detected_domain, confidence_score) or (None, 0.0) if no domain is detected
        """
        text_lower = text.lower()
        
        # Check for high-precision patterns (highest weight - these are the 20% that identify 80% of cases)
        pattern_matches = {}
        for domain, patterns in self.domain_patterns.items():
            pattern_matches[domain] = 0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                pattern_matches[domain] += len(matches)
        
        # Check for domain-specific entities (second highest weight)
        entity_matches = {}
        for domain, entities in self.domain_entities.items():
            entity_matches[domain] = 0
            for entity in entities:
                if entity.lower() in text_lower:
                    entity_matches[domain] += 1
                    
        # Check for domain-specific terminology frequency (lowest weight but still important)
        term_matches = {}
        for domain, terms in self.domain_terminology.items():
            term_matches[domain] = 0
            for term in terms:
                if term.lower() in text_lower:
                    term_matches[domain] += 1
        
        # Calculate combined scores with weighted importance
        domain_scores = {}
        for domain in self.domain_patterns.keys():
            # Pattern matches are weighted most heavily (60% of score)
            # These are the 20% of patterns that identify 80% of domain-specific content
            pattern_score = min(1.0, pattern_matches[domain] / 3) * 0.6
            
            # Entity matches are weighted second (30% of score)
            entity_score = min(1.0, entity_matches[domain] / 2) * 0.3
            
            # Term frequency is weighted least but still important (10% of score)
            term_threshold = self.domain_term_thresholds.get(domain, 4)
            term_score = min(1.0, term_matches[domain] / term_threshold) * 0.1
            
            # Combined score
            domain_scores[domain] = pattern_score + entity_score + term_score
        
        # Find the domain with the highest score
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            domain, score = best_domain
            
            # Only return a domain if the score is above a minimum threshold
            if score >= 0.3:
                return domain, score
            
            # Debug information about top scores
            top_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Top domain scores: {top_domains}")
            print(f"Pattern matches for top domain: {pattern_matches[domain]}")
            print(f"Entity matches for top domain: {entity_matches[domain]}")
            print(f"Term matches for top domain: {term_matches[domain]}")
        
        # No domain detected with sufficient confidence
        return None, 0.0