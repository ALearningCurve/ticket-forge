"""
Step 2: Resume Normalizer
-------------------------
Cleans resume text by removing PII and unwanted data.

Removes:
- Phone numbers
- Email addresses  
- URLs (LinkedIn, GitHub, etc.)
- Addresses (City, State, ZIP)
- Dates (Jan 2020, 2019-Present, etc.)
- GPA

No external dependencies required (uses regex).
"""

import re
from typing import Tuple, Dict, List


class ResumeNormalizer:
    """Normalize resume text by removing PII and dates."""
    
    # Regex patterns for items to remove
    PATTERNS = {
        'phone': [
            r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}',
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
        ],
        'email': [
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        ],
        'url': [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'(?:linkedin|github|twitter|gitlab|bitbucket)\.com/[^\s]*',
        ],
        'address': [
            r'\d{1,5}\s+[\w\s]{1,30}(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|way|place|pl)\.?',
            r'[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?',
            r'[A-Z][a-zA-Z]+,\s*[A-Z]{2}(?=\s|$|•|\||\n)',
        ],
        'dates': [
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{0,2},?\s*\d{2,4}',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}\s*[-–—]\s*(?:\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)',
            r'(?:19|20)\d{2}',
        ],
        'gpa': [
            r'GPA[:\s]*\d+\.?\d*(?:\s*/\s*\d+\.?\d*)?',
            r'\d+\.\d+\s*/\s*4\.0',
        ],
    }
    
    def __init__(
        self,
        remove_phone: bool = True,
        remove_email: bool = True,
        remove_url: bool = True,
        remove_address: bool = True,
        remove_dates: bool = True,
        remove_gpa: bool = True,
    ):
        self.config = {
            'phone': remove_phone,
            'email': remove_email,
            'url': remove_url,
            'address': remove_address,
            'dates': remove_dates,
            'gpa': remove_gpa,
        }
    
    def normalize(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Normalize a single resume text.
        
        Args:
            text: Raw resume text
            
        Returns:
            Tuple of (normalized_text, removed_items_dict)
        """
        removed = {}
        normalized = text
        
        # Apply each pattern category
        for category, should_remove in self.config.items():
            if not should_remove:
                continue
                
            patterns = self.PATTERNS.get(category, [])
            category_matches = []
            
            for pattern in patterns:
                matches = re.findall(pattern, normalized, re.IGNORECASE)
                category_matches.extend(matches)
                normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
            
            if category_matches:
                removed[category] = list(set(category_matches))
        
        # Clean up formatting
        normalized = self._clean_formatting(normalized)
        
        return normalized, removed
    
    def _clean_formatting(self, text: str) -> str:
        """Clean up whitespace and special characters."""
        # Remove bullet points and special chars
        text = re.sub(r'[•▪▸►◆◇○●■□–—|·]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n+', '\n', text)
        
        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        return text.strip()
    
    def normalize_batch(self, data: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """
        Normalize multiple resumes.
        
        Args:
            data: Dict mapping engineer_id to raw text
            
        Returns:
            Tuple of:
                - Dict mapping engineer_id to normalized text
                - Dict mapping engineer_id to removed items
        """
        normalized_results = {}
        removed_results = {}
        
        for engineer_id, raw_text in data.items():
            print(f"Normalizing: {engineer_id}")
            
            normalized_text, removed = self.normalize(raw_text)
            normalized_results[engineer_id] = normalized_text
            removed_results[engineer_id] = removed
            
            removed_count = sum(len(v) for v in removed.values())
            print(f"    ✓ Removed {removed_count} PII items")
        
        return normalized_results, removed_results
