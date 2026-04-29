#!/usr/bin/env python3
"""
Nova - Facebook Engagement Analyzer (Professional Edition v3.0)
Developed by: Emaf-png
============================================
Hybrid approach: Graph API + ML-based fake account detection
with minimal scraping fallback and professional scoring system.

Version: 3.0.0 (Complete Rewrite)
"""

import sys
import os
import time
import json
import hashlib
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re
import math

import requests
from colorama import init, Fore, Style
from tqdm import tqdm

# Conditional imports - fall back gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import words as nltk_words
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

init(autoreset=True)

# ==================== DATA MODELS ====================
class RiskLevel(Enum):
    """Sophisticated risk classification"""
    LEGITIMATE = (0, 20, Fore.GREEN, "✅")
    LOW_RISK = (21, 40, Fore.LIGHTCYAN_EX, "🟢")
    MEDIUM_RISK = (41, 60, Fore.YELLOW, "🟡")
    HIGH_RISK = (61, 80, Fore.LIGHTRED_EX, "🟠")
    CRITICAL = (81, 100, Fore.RED, "🔴")
    
    def __init__(self, min_score: int, max_score: int, color: str, emoji: str):
        self.min_score = min_score
        self.max_score = max_score
        self.color = color
        self.emoji = emoji
    
    @classmethod
    def from_score(cls, score: int) -> 'RiskLevel':
        for level in cls:
            if level.min_score <= score <= level.max_score:
                return level
        return cls.CRITICAL


@dataclass
class ProfileFeatures:
    """Rich profile feature set for ML analysis"""
    # Basic features
    name: str = ""
    profile_id: str = ""
    profile_url: str = ""
    
    # Profile completeness features
    has_profile_pic: bool = False
    has_cover_photo: bool = False
    has_bio: bool = False
    has_education: bool = False
    has_work: bool = False
    has_location: bool = False
    has_relationship: bool = False
    
    # Activity features
    friends_count: int = 0
    followers_count: int = 0
    posts_count: int = 0
    photos_count: int = 0
    account_age_days: int = 0
    last_active_days: int = 0
    
    # Content quality features
    name_entropy: float = 0.0
    name_has_numbers: bool = False
    name_length: int = 0
    name_word_count: int = 0
    name_contains_real_words: bool = False
    
    # Network features
    friends_to_followers_ratio: float = 0.0
    posts_per_day: float = 0.0
    
    # Behavioral features
    reaction_pattern: str = ""
    comment_quality_score: float = 0.0
    activity_regularity_score: float = 0.0
    
    # Raw data for debugging
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_feature_vector(self) -> List[float]:
        """Convert to numerical feature vector for ML"""
        return [
            float(self.has_profile_pic),
            float(self.has_cover_photo),
            float(self.has_bio),
            float(self.has_education),
            float(self.has_work),
            float(self.has_location),
            float(self.has_relationship),
            float(self.friends_count),
            float(self.followers_count),
            float(self.posts_count),
            float(self.photos_count),
            float(self.account_age_days),
            float(self.last_active_days),
            self.name_entropy,
            float(self.name_has_numbers),
            float(self.name_length),
            float(self.name_word_count),
            float(self.name_contains_real_words),
            self.friends_to_followers_ratio,
            self.posts_per_day,
            self.comment_quality_score,
            self.activity_regularity_score,
        ]


@dataclass
class AnalysisResult:
    """Comprehensive analysis result"""
    profile: ProfileFeatures
    risk_score: int
    risk_level: RiskLevel
    flags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    ml_anomaly_score: Optional[float] = None
    analysis_method: str = "hybrid"


# ==================== CONFIGURATION ====================
class Config:
    """Central configuration management"""
    # API Configuration
    GRAPH_API_VERSION = "v18.0"
    GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
    
    # Analysis thresholds
    MIN_ACCOUNT_AGE_DAYS = 30
    SUSPICIOUS_AGE_DAYS = 90
    MIN_FRIENDS_FOR_LEGITIMATE = 20
    MAX_POSTS_PER_DAY_SPAM = 10
    
    # ML Configuration
    ANOMALY_CONTAMINATION = 0.1  # Expected ratio of anomalies
    CONFIDENCE_THRESHOLD = 0.7
    
    # Scoring weights (must sum to 1.0)
    WEIGHTS = {
        "profile_completeness": 0.25,
        "account_age": 0.15,
        "network_metrics": 0.20,
        "name_quality": 0.15,
        "activity_patterns": 0.15,
        "ml_anomaly": 0.10 if SKLEARN_AVAILABLE else 0.0,
    }
    
    # Redistribute ML weight if not available
    if not SKLEARN_AVAILABLE:
        remaining = 0.10
        for key in ["profile_completeness", "account_age", "network_metrics", "name_quality", "activity_patterns"]:
            WEIGHTS[key] += remaining / 5


# ==================== LOGO & UI ====================
def print_premium_logo():
    """Premium ASCII art logo"""
    logo = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗                       ║
    ║   ████╗  ██║██╔═══██╗██║   ██║██╔══██╗                      ║
    ║   ██╔██╗ ██║██║   ██║██║   ██║███████║                      ║
    ║   ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║                      ║
    ║   ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║                      ║
    ║   ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝                      ║
    ║                                                              ║
    ║   Facebook Engagement Analyzer - Professional Edition v3.0   ║
    ║   Developed by: Emaf-png | Hybrid ML + Graph API Approach    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(Fore.CYAN + Style.BRIGHT + logo)


# ==================== GRAPH API CLIENT ====================
class FacebookGraphAPI:
    """Professional Graph API client with intelligent fallback"""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.session = requests.Session()
        self.session.params = {"access_token": access_token}
        self.cache = {}
        self.rate_limit_remaining = None
    
    def get_post_reactions(self, post_id: str, limit: int = 100) -> List[Dict]:
        """Fetch reactions using Graph API"""
        url = f"{Config.GRAPH_API_BASE}/{post_id}/reactions"
        all_reactions = []
        
        params = {
            "limit": min(limit, 100),
            "fields": "id,name,pic_small,profile_type"
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            all_reactions.extend(data.get("data", []))
            
            # Paginate if needed
            while "paging" in data and "next" in data["paging"] and len(all_reactions) < limit:
                response = self.session.get(data["paging"]["next"])
                response.raise_for_status()
                data = response.json()
                all_reactions.extend(data.get("data", []))
            
            return all_reactions[:limit]
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Graph API error: {e}")
            return []
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Fetch detailed user profile"""
        if user_id in self.cache:
            return self.cache[user_id]
        
        url = f"{Config.GRAPH_API_BASE}/{user_id}"
        fields = [
            "id", "name", "picture.width(200).height(200)",
            "cover", "about", "education", "work",
            "location", "relationship_status",
            "friends.limit(5000)", "followers.limit(5000)",
            "posts.limit(100){created_time}",
            "photos.limit(100){created_time}"
        ]
        
        params = {"fields": ",".join(fields)}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            profile_data = response.json()
            self.cache[user_id] = profile_data
            return profile_data
        except requests.exceptions.RequestException as e:
            logging.warning(f"Failed to fetch profile {user_id}: {e}")
            return None
    
    def bulk_get_profiles(self, user_ids: List[str]) -> Dict[str, Dict]:
        """Fetch multiple profiles efficiently using batch requests"""
        profiles = {}
        batch_size = 50  # Facebook batch limit
        
        for i in range(0, len(user_ids), batch_size):
            batch = user_ids[i:i + batch_size]
            batch_requests = []
            
            for user_id in batch:
                relative_url = f"{user_id}?fields=id,name,picture,about,education,work,location,relationship_status,friends.limit(5000),posts.limit(50){{created_time}}"
                batch_requests.append({
                    "method": "GET",
                    "relative_url": relative_url
                })
            
            try:
                url = f"{Config.GRAPH_API_BASE}/"
                params = {
                    "batch": json.dumps(batch_requests),
                    "include_headers": "false"
                }
                response = self.session.post(url, params=params)
                response.raise_for_status()
                
                for user_id, result in zip(batch, response.json()):
                    if result.get("code") == 200:
                        body = json.loads(result["body"])
                        profiles[user_id] = body
                        self.cache[user_id] = body
                        
            except Exception as e:
                logging.error(f"Batch request failed: {e}")
        
        return profiles


# ==================== ML-BASED ANALYZER ====================
class MLAnalyzer:
    """Machine Learning based fake account detection"""
    
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE and NUMPY_AVAILABLE else None
        self.model = None
        self.trained = False
    
    def extract_features(self, profile: Dict) -> ProfileFeatures:
        """Extract rich features from profile data"""
        features = ProfileFeatures()
        
        # Basic info
        features.name = profile.get("name", "")
        features.profile_id = profile.get("id", "")
        features.profile_url = f"https://facebook.com/{features.profile_id}"
        
        # Profile completeness
        features.has_profile_pic = bool(profile.get("picture", {}).get("data", {}).get("url"))
        features.has_cover_photo = bool(profile.get("cover"))
        features.has_bio = bool(profile.get("about"))
        features.has_education = bool(profile.get("education"))
        features.has_work = bool(profile.get("work"))
        features.has_location = bool(profile.get("location"))
        features.has_relationship = bool(profile.get("relationship_status"))
        
        # Network metrics
        friends_data = profile.get("friends", {})
        features.friends_count = friends_data.get("summary", {}).get("total_count", 0)
        
        followers_data = profile.get("followers", {})
        features.followers_count = followers_data.get("summary", {}).get("total_count", 0)
        
        # Activity metrics
        posts = profile.get("posts", {}).get("data", [])
        features.posts_count = len(posts)
        
        photos = profile.get("photos", {}).get("data", [])
        features.photos_count = len(photos)
        
        # Account age
        if posts:
            first_post = min(posts, key=lambda p: p.get("created_time", ""))
            created = first_post.get("created_time", "")
            if created:
                try:
                    created_date = datetime.fromisoformat(created.replace("T", " ").replace("+0000", ""))
                    features.account_age_days = (datetime.now() - created_date).days
                except:
                    features.account_age_days = 0
        
        # Name quality analysis
        features.name_length = len(features.name)
        features.name_word_count = len(features.name.split())
        features.name_has_numbers = bool(re.search(r'\d', features.name))
        
        # Entropy calculation
        if features.name:
            char_freq = {}
            for char in features.name:
                char_freq[char] = char_freq.get(char, 0) + 1
            length = len(features.name)
            features.name_entropy = -sum((freq/length) * math.log2(freq/length) 
                                        for freq in char_freq.values())
        
        # Real words check
        if NLTK_AVAILABLE:
            try:
                english_words = set(nltk_words.words())
                name_words = features.name.lower().split()
                features.name_contains_real_words = any(
                    word in english_words for word in name_words
                )
            except:
                features.name_contains_real_words = False
        
        # Derived metrics
        if features.followers_count > 0:
            features.friends_to_followers_ratio = features.friends_count / features.followers_count
        
        if features.account_age_days > 0:
            features.posts_per_day = features.posts_count / features.account_age_days
        
        # Store raw data
        features.raw_data = profile
        
        return features
    
    def train_anomaly_detector(self, feature_vectors: List[List[float]]):
        """Train isolation forest for anomaly detection"""
        if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE or len(feature_vectors) < 10:
            return False
        
        try:
            X = np.array(feature_vectors)
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = IsolationForest(
                contamination=Config.ANOMALY_CONTAMINATION,
                random_state=42,
                n_estimators=100
            )
            self.model.fit(X_scaled)
            self.trained = True
            return True
        except Exception as e:
            logging.error(f"Failed to train model: {e}")
            return False
    
    def calculate_anomaly_score(self, feature_vector: List[float]) -> float:
        """Calculate anomaly score using trained model"""
        if not self.trained or not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
            return 0.5  # Neutral score
        
        try:
            X = np.array([feature_vector])
            X_scaled = self.scaler.transform(X)
            
            # Get decision function score
            score = self.model.decision_function(X_scaled)[0]
            
            # Normalize to 0-1 range (0 = normal, 1 = anomalous)
            normalized_score = 1 / (1 + math.exp(score))
            return normalized_score
        except Exception:
            return 0.5


# ==================== SCORING ENGINE ====================
class ScoringEngine:
    """Professional scoring system with weighted evaluation"""
    
    def __init__(self, ml_analyzer: Optional[MLAnalyzer] = None):
        self.ml_analyzer = ml_analyzer
        self.weights = Config.WEIGHTS
    
    def calculate_score(self, features: ProfileFeatures, 
                       ml_anomaly_score: Optional[float] = None) -> AnalysisResult:
        """Calculate comprehensive risk score"""
        scores = {}
        flags = []
        
        # 1. Profile Completeness Score
        completeness_points = 0
        completeness_checks = [
            features.has_profile_pic,
            features.has_cover_photo,
            features.has_bio,
            features.has_education,
            features.has_work,
            features.has_location,
            features.has_relationship
        ]
        completeness_points = sum(1 for check in completeness_checks if check)
        completeness_ratio = completeness_points / len(completeness_checks)
        scores["profile_completeness"] = (1 - completeness_ratio) * 100
        
        if completeness_ratio < 0.3:
            flags.append("Very low profile completeness")
        if not features.has_profile_pic:
            flags.append("Missing profile picture")
        if not features.has_cover_photo:
            flags.append("Missing cover photo")
        
        # 2. Account Age Score
        if features.account_age_days == 0:
            scores["account_age"] = 90  # Very suspicious
            flags.append("Cannot determine account age")
        elif features.account_age_days < Config.MIN_ACCOUNT_AGE_DAYS:
            scores["account_age"] = 100
            flags.append(f"Account only {features.account_age_days} days old")
        elif features.account_age_days < Config.SUSPICIOUS_AGE_DAYS:
            scores["account_age"] = 70
            flags.append(f"Account less than {Config.SUSPICIOUS_AGE_DAYS} days old")
        else:
            age_factor = min(features.account_age_days / 365, 5)  # Cap at 5 years
            scores["account_age"] = max(0, 50 - (age_factor * 10))
        
        # 3. Network Metrics Score
        network_score = 0
        
        if features.friends_count < 5:
            network_score += 40
            flags.append("Very few friends")
        elif features.friends_count < Config.MIN_FRIENDS_FOR_LEGITIMATE:
            network_score += 20
            flags.append("Below minimum friend threshold")
        
        if features.followers_count > 0:
            ratio = features.friends_to_followers_ratio
            if ratio < 0.1:  # Many followers, few friends
                network_score += 30
                flags.append("Suspicious friends-to-followers ratio")
            elif ratio > 10:  # Many friends, few followers
                network_score += 15
                flags.append("Unusual friends-to-followers ratio")
        
        scores["network_metrics"] = min(network_score, 100)
        
        # 4. Name Quality Score
        name_score = 0
        
        if features.name_has_numbers:
            name_score += 30
            flags.append("Name contains numbers")
        
        if features.name_entropy > 3.5:
            name_score += 40
            flags.append("Suspiciously random name")
        
        if features.name_length > 30:
            name_score += 20
            flags.append("Unusually long name")
        
        if features.name_word_count == 1 and features.name_length < 3:
            name_score += 50
            flags.append("Too short name")
        
        if not features.name_contains_real_words and NLTK_AVAILABLE:
            name_score += 35
            flags.append("Name doesn't contain real words")
        
        scores["name_quality"] = min(name_score, 100)
        
        # 5. Activity Patterns Score
        activity_score = 0
        
        if features.posts_count == 0 and features.account_age_days > 30:
            activity_score += 60
            flags.append("No posts despite old account")
        
        if features.posts_per_day > Config.MAX_POSTS_PER_DAY_SPAM:
            activity_score += 70
            flags.append("Suspicious posting frequency")
        
        if features.photos_count == 0 and features.posts_count > 10:
            activity_score += 40
            flags.append("Many posts but no photos")
        
        scores["activity_patterns"] = min(activity_score, 100)
        
        # 6. ML Anomaly Score (if available)
        if ml_anomaly_score is not None and Config.WEIGHTS["ml_anomaly"] > 0:
            scores["ml_anomaly"] = ml_anomaly_score * 100
        
        # Calculate weighted final score
        final_score = 0
        for category, score in scores.items():
            final_score += score * self.weights.get(category, 0)
        
        # Adjust for missing ML component
        if not SKLEARN_AVAILABLE:
            final_score = final_score / (1 - 0.1)  # Normalize without ML weight
        
        final_score = min(max(final_score, 0), 100)
        
        # Determine risk level
        risk_level = RiskLevel.from_score(int(final_score))
        
        # Calculate confidence
        confidence = self._calculate_confidence(scores, features)
        
        return AnalysisResult(
            profile=features,
            risk_score=int(final_score),
            risk_level=risk_level,
            flags=flags,
            confidence=confidence,
            ml_anomaly_score=ml_anomaly_score,
            analysis_method="hybrid" if ml_anomaly_score else "rule_based"
        )
    
    def _calculate_confidence(self, scores: Dict[str, float], 
                            features: ProfileFeatures) -> float:
        """Calculate confidence level of the analysis"""
        # More data points = higher confidence
        data_points = sum([
            features.account_age_days > 0,
            features.friends_count > 0,
            features.posts_count > 0,
            features.has_profile_pic,
            features.has_bio
        ])
        
        # Score consistency
        score_values = list(scores.values())
        if len(score_values) > 1:
            variance = np.var(score_values) if NUMPY_AVAILABLE else 0
            consistency = 1 / (1 + variance / 1000)
        else:
            consistency = 0.5
        
        confidence = (data_points / 5) * 0.6 + consistency * 0.4
        return min(confidence, 1.0)


# ==================== MAIN ANALYZER ====================
class NovaAnalyzer:
    """Main orchestrator for the analysis pipeline"""
    
    def __init__(self, access_token: str):
        self.api = FacebookGraphAPI(access_token)
        self.ml_analyzer = MLAnalyzer()
        self.scoring_engine = ScoringEngine(self.ml_analyzer)
        self.features_cache = {}
        
    def analyze_post_engagement(self, post_url: str, deep_scan: bool = False,
                               max_reactions: int = 500) -> Dict[str, Any]:
        """Complete engagement analysis pipeline"""
        # Extract post ID
        post_id = self._extract_post_id(post_url)
        if not post_id:
            return {"error": "Could not extract post ID"}
        
        print(Fore.CYAN + f"\n📊 Analyzing post: {post_id}")
        
        # Step 1: Fetch reactions
        print(Fore.WHITE + "🔍 Fetching reactions via Graph API...")
        reactions = self.api.get_post_reactions(post_id, limit=max_reactions)
        
        if not reactions:
            return {"error": "No reactions found or API error"}
        
        print(Fore.GREEN + f"✅ Found {len(reactions)} reactions")
        
        # Step 2: Extract user IDs
        user_ids = [r["id"] for r in reactions if "id" in r]
        
        # Step 3: Fetch profiles
        if deep_scan:
            print(Fore.WHITE + "👤 Fetching detailed profiles...")
            profiles = self.api.bulk_get_profiles(user_ids)
        else:
            # Quick analysis with minimal data
            profiles = {uid: r for uid, r in zip(user_ids, reactions)}
        
        print(Fore.GREEN + f"✅ Retrieved {len(profiles)} profiles")
        
        # Step 4: Feature extraction
        print(Fore.WHITE + "🧬 Extracting features...")
        feature_list = []
        
        for user_id, profile in profiles.items():
            features = self.ml_analyzer.extract_features(profile)
            self.features_cache[user_id] = features
            feature_list.append(features.to_feature_vector())
        
        # Step 5: Train anomaly detection if enough data
        if SKLEARN_AVAILABLE and len(feature_list) >= 10:
            print(Fore.WHITE + "🤖 Training anomaly detection model...")
            self.ml_analyzer.train_anomaly_detector(feature_list)
        
        # Step 6: Score each profile
        print(Fore.WHITE + "📈 Scoring profiles...")
        results = []
        
        for user_id, features in tqdm(self.features_cache.items(), desc="Analyzing"):
            # Calculate ML anomaly score if available
            ml_score = None
            if self.ml_analyzer.trained:
                ml_score = self.ml_analyzer.calculate_anomaly_score(
                    features.to_feature_vector()
                )
            
            # Get comprehensive score
            result = self.scoring_engine.calculate_score(features, ml_score)
            results.append(result)
        
        # Step 7: Sort by risk score
        results.sort(key=lambda x: x.risk_score, reverse=True)
        
        # Step 8: Generate report
        return self._generate_report(post_url, reactions, results)
    
    def _extract_post_id(self, url: str) -> Optional[str]:
        """Extract post ID from various URL formats"""
        # Handle different post URL formats
        patterns = [
            r'/posts/(\d+)',
            r'story_fbid=(\d+)',
            r'fbid=(\d+)',
            r'permalink/(\d+)',
            r'videos/(\d+)',
            r'(\d{15,})'  # Just a long number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _generate_report(self, post_url: str, reactions: List[Dict],
                        results: List[AnalysisResult]) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        total = len(reactions)
        
        # Statistics
        risk_distribution = {}
        for level in RiskLevel:
            count = sum(1 for r in results if r.risk_level == level)
            risk_distribution[level.name] = count
        
        suspicious = [r for r in results if r.risk_score >= 40]
        high_risk = [r for r in results if r.risk_score >= 61]
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "post_url": post_url,
            "total_reactions": total,
            "analyzed_profiles": len(results),
            "statistics": {
                "suspicious_accounts": len(suspicious),
                "high_risk_accounts": len(high_risk),
                "suspicious_percentage": (len(suspicious) / total * 100) if total > 0 else 0,
                "risk_distribution": risk_distribution,
                "average_confidence": sum(r.confidence for r in results) / len(results) if results else 0,
            },
            "suspicious_accounts": [
                {
                    "name": r.profile.name,
                    "profile_url": r.profile.profile_url,
                    "risk_score": r.risk_score,
                    "risk_level": r.risk_level.name,
                    "flags": r.flags,
                    "confidence": r.confidence,
                    "analysis_method": r.analysis_method,
                }
                for r in suspicious[:20]  # Top 20 suspicious
            ],
            "methodology": {
                "approach": "Hybrid ML + Rule-based scoring",
                "ml_available": SKLEARN_AVAILABLE and NUMPY_AVAILABLE,
                "nltk_available": NLTK_AVAILABLE,
                "weights_used": Config.WEIGHTS,
                "features_analyzed": 22,
            }
        }
        
        return report


# ==================== REPORT DISPLAY ====================
def display_interactive_report(report: Dict[str, Any]):
    """Professional interactive report display"""
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 70)
    print(Fore.CYAN + Style.BRIGHT + "     NOVA PROFESSIONAL - ENGAGEMENT ANALYSIS REPORT")
    print(Fore.CYAN + Style.BRIGHT + "=" * 70)
    
    if "error" in report:
        print(Fore.RED + f"\n❌ Error: {report['error']}")
        return
    
    stats = report["statistics"]
    methodology = report["methodology"]
    
    # Header
    print(Fore.WHITE + Style.BRIGHT + "\n📋 EXECUTIVE SUMMARY")
    print(Fore.WHITE + f"   Total Reactions: {report['total_reactions']}")
    print(Fore.WHITE + f"   Analyzed Profiles: {report['analyzed_profiles']}")
    print(Fore.WHITE + f"   Suspicious: {stats['suspicious_accounts']} ({stats['suspicious_percentage']:.1f}%)")
    print(Fore.WHITE + f"   High Risk: {stats['high_risk_accounts']}")
    print(Fore.WHITE + f"   Analysis Confidence: {stats['average_confidence']:.1%}")
    
    # Methodology
    print(Fore.CYAN + "\n🔬 METHODOLOGY")
    print(Fore.WHITE + f"   Approach: {methodology['approach']}")
    print(Fore.WHITE + f"   ML Available: {'✅' if methodology['ml_available'] else '❌'}")
    print(Fore.WHITE + f"   NLP Available: {'✅' if methodology['nltk_available'] else '❌'}")
    
    # Risk Distribution Chart (ASCII)
    print(Fore.CYAN + "\n📊 RISK DISTRIBUTION")
    max_count = max(stats["risk_distribution"].values()) if stats["risk_distribution"] else 1
    
    for level in RiskLevel:
        count = stats["risk_distribution"].get(level.name, 0)
        bar_length = int((count / max_count) * 30) if max_count > 0 else 0
        bar = "█" * bar_length
        print(f"   {level.emoji} {level.name:15} [{count:4}] {level.color}{bar}")
    
    # Suspicious Accounts
    if report["suspicious_accounts"]:
        print(Fore.YELLOW + Style.BRIGHT + f"\n⚠️  TOP SUSPICIOUS ACCOUNTS")
        print(Fore.YELLOW + "-" * 70)
        
        for i, acc in enumerate(report["suspicious_accounts"], 1):
            risk_level = RiskLevel[acc["risk_level"]]
            print(f"\n   {risk_level.emoji} #{i} | Score: {acc['risk_score']}/100 | Confidence: {acc['confidence']:.0%}")
            print(f"   {Fore.WHITE}Name: {acc['name']}")
            print(f"   {Fore.BLUE}Profile: {acc['profile_url']}")
            print(f"   {Fore.RED}Flags: {', '.join(acc['flags'])}")
    
    print(Fore.CYAN + "\n" + "=" * 70)


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(
        description="Nova Professional - Facebook Engagement Analyzer v3.0",
        epilog="Example: python nova.py --token YOUR_ACCESS_TOKEN --url POST_URL --deep",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--token", required=True, 
                       help="Facebook Graph API Access Token")
    parser.add_argument("--url", required=True,
                       help="Facebook post URL to analyze")
    parser.add_argument("--deep", action="store_true",
                       help="Perform deep profile analysis")
    parser.add_argument("--max", type=int, default=500,
                       help="Maximum reactions to analyze (default: 500)")
    parser.add_argument("--output", help="Save report as JSON file")
    
    args = parser.parse_args()
    
    # Display logo
    print_premium_logo()
    
    # Initialize analyzer
    analyzer = NovaAnalyzer(args.token)
    
    # Perform analysis
    print(Fore.CYAN + "\n🚀 Starting analysis pipeline...")
    report = analyzer.analyze_post_engagement(
        args.url,
        deep_scan=args.deep,
        max_reactions=args.max
    )
    
    # Display report
    display_interactive_report(report)
    
    # Save to file if requested
    if args.output and "error" not in report:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(Fore.GREEN + f"\n📁 Detailed report saved to: {args.output}")
    
    print(Fore.CYAN + "\n" + "=" * 70)
    print(Fore.GREEN + Style.BRIGHT + "Analysis completed successfully!")
    print(Fore.CYAN + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n⚠️ Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(Fore.RED + f"\n❌ An unexpected error occurred: {e}")
        sys.exit(1)
