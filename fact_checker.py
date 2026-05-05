"""
fact_checker.py — Real-Time Fact Verification Module for TruthLens AI

Uses the Google Fact Check Tools API (free) to cross-reference claims
against a global database of verified fact-checks from 100+ publishers.

Falls back gracefully when no API key is configured.
"""

import os
import re
import logging
import requests
from typing import Optional

logger = logging.getLogger("TruthLens.FactChecker")

# Google Fact Check Tools API endpoint
FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def extract_key_claims(text: str, max_claims: int = 3) -> list[str]:
    """
    Extract the most important claims/phrases from the text for fact-checking.
    Uses a combination of headline extraction and entity-based chunking.
    """
    claims = []
    
    # Strategy 1: Use the first sentence (often the headline/lede)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 5]
    
    if sentences:
        # The first sentence (headline/lede) is the most important claim
        first = sentences[0]
        if len(first) > 200:
            first = first[:200]
        claims.append(first)
    
    # Strategy 2: Look for quotation-based claims
    quotes = re.findall(r'"([^"]{20,150})"', text)
    for q in quotes[:1]:
        claims.append(q)
    
    # Strategy 3: Look for specific factual assertions
    # Patterns like "X said Y", "according to X", "X reported that"
    assertion_patterns = [
        r'(?:said|claimed|stated|announced|declared)\s+(?:that\s+)?(.{20,120}?)[.]',
        r'(?:according to .{3,30},)\s+(.{20,120}?)[.]',
    ]
    for pattern in assertion_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches[:1]:
            if m not in claims:
                claims.append(m.strip())
    
    return claims[:max_claims]


def query_factcheck_api(query: str, api_key: str, language: str = "en", max_age_days: int = 365) -> Optional[dict]:
    """
    Query the Google Fact Check Tools API for a given claim.
    Returns parsed results or None on failure.
    """
    try:
        params = {
            "query": query,
            "languageCode": language,
            "key": api_key,
            "pageSize": 5,
        }
        if max_age_days:
            params["maxAgeDays"] = max_age_days
        
        response = requests.get(FACTCHECK_API_URL, params=params, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Fact Check API returned status {response.status_code}: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        logger.warning("Fact Check API request timed out")
        return None
    except Exception as e:
        logger.warning(f"Fact Check API error: {e}")
        return None


def analyze_factcheck_results(api_response: dict) -> dict:
    """
    Parse the API response and extract structured fact-check information.
    Returns a summary with scores and source details.
    """
    if not api_response or "claims" not in api_response:
        return {
            "found": False,
            "matches": [],
            "credibility_signal": 0.0,  # No signal (neutral)
            "summary": "No matching fact-checks found in global databases."
        }
    
    matches = []
    total_score = 0.0
    
    # Rating keywords mapped to credibility scores
    # Positive ratings (claim is true) → lower fake probability
    # Negative ratings (claim is false) → higher fake probability
    false_keywords = ["false", "pants on fire", "mostly false", "incorrect", 
                      "misleading", "not true", "fabricated", "fake", "hoax",
                      "wrong", "inaccurate", "no evidence", "unproven", "distorts"]
    true_keywords = ["true", "mostly true", "correct", "accurate", "verified",
                     "confirmed", "supported", "factual", "real"]
    mixed_keywords = ["half true", "mixture", "partly", "partially", "mixed",
                      "needs context", "unverified", "inconclusive"]
    
    for claim_data in api_response.get("claims", [])[:5]:
        claim_text = claim_data.get("text", "")
        claimant = claim_data.get("claimant", "Unknown")
        
        for review in claim_data.get("claimReview", []):
            publisher = review.get("publisher", {}).get("name", "Unknown")
            rating = review.get("textualRating", "").lower()
            url = review.get("url", "")
            title = review.get("title", "")
            
            # Determine the credibility signal from the rating
            score = 0.0
            rating_category = "unknown"
            
            if any(kw in rating for kw in false_keywords):
                score = 0.8  # Strong fake signal
                rating_category = "FALSE"
            elif any(kw in rating for kw in true_keywords):
                score = -0.8  # Strong real signal
                rating_category = "TRUE"
            elif any(kw in rating for kw in mixed_keywords):
                score = 0.2  # Slight fake signal
                rating_category = "MIXED"
            
            total_score += score
            
            matches.append({
                "claim": claim_text[:200],
                "claimant": claimant,
                "publisher": publisher,
                "rating": review.get("textualRating", "Unknown"),
                "rating_category": rating_category,
                "url": url,
                "title": title[:150] if title else ""
            })
    
    if not matches:
        return {
            "found": False,
            "matches": [],
            "credibility_signal": 0.0,
            "summary": "No matching fact-checks found."
        }
    
    # Average the scores across all matches
    avg_signal = total_score / len(matches)
    
    # Determine overall summary
    if avg_signal > 0.3:
        summary = f"⚠️ Found {len(matches)} fact-check(s) suggesting this claim may be FALSE or MISLEADING."
    elif avg_signal < -0.3:
        summary = f"✅ Found {len(matches)} fact-check(s) confirming this claim as TRUE or VERIFIED."
    else:
        summary = f"Found {len(matches)} fact-check(s) with MIXED or INCONCLUSIVE ratings."
    
    return {
        "found": True,
        "matches": matches[:3],  # Return top 3
        "credibility_signal": round(avg_signal, 3),
        "summary": summary
    }


def check_facts(text: str, api_key: Optional[str] = None) -> dict:
    """
    Main entry point: Extract claims from text and fact-check them.
    Returns comprehensive fact-check results.
    
    If no API key is provided, returns a neutral result with no signal.
    """
    # Check for API key
    if not api_key:
        api_key = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
    
    if not api_key:
        logger.info("No Google Fact Check API key configured. Skipping fact-check.")
        return {
            "enabled": False,
            "found": False,
            "matches": [],
            "credibility_signal": 0.0,
            "summary": "Fact-check module inactive (no API key configured).",
            "claims_searched": []
        }
    
    # Extract key claims
    claims = extract_key_claims(text)
    
    if not claims:
        return {
            "enabled": True,
            "found": False,
            "matches": [],
            "credibility_signal": 0.0,
            "summary": "Could not extract verifiable claims from the text.",
            "claims_searched": []
        }
    
    # Query API for each claim and aggregate results
    all_matches = []
    total_signal = 0.0
    
    for claim in claims:
        api_result = query_factcheck_api(claim, api_key)
        if api_result:
            parsed = analyze_factcheck_results(api_result)
            if parsed["found"]:
                all_matches.extend(parsed["matches"])
                total_signal += parsed["credibility_signal"]
    
    if not all_matches:
        return {
            "enabled": True,
            "found": False,
            "matches": [],
            "credibility_signal": 0.0,
            "summary": "No matching fact-checks found for extracted claims.",
            "claims_searched": claims
        }
    
    # Deduplicate matches by URL
    seen_urls = set()
    unique_matches = []
    for m in all_matches:
        if m["url"] not in seen_urls:
            seen_urls.add(m["url"])
            unique_matches.append(m)
    
    avg_signal = total_signal / len(claims)
    
    if avg_signal > 0.3:
        summary = f"⚠️ {len(unique_matches)} fact-checker(s) flagged claims as FALSE or MISLEADING."
    elif avg_signal < -0.3:
        summary = f"✅ {len(unique_matches)} fact-checker(s) verified claims as TRUE."
    else:
        summary = f"Found {len(unique_matches)} fact-check(s) with mixed ratings."
    
    return {
        "enabled": True,
        "found": True,
        "matches": unique_matches[:5],
        "credibility_signal": round(avg_signal, 3),
        "summary": summary,
        "claims_searched": claims
    }
