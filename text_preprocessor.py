import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import unicodedata

#function to download nltk resources
def _download_nltk():
    for pkg in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.download(pkg, quiet=False)
        except Exception as e:
            print(f"NLTK '{pkg}' download failed: {e}")
            
#function calling
_download_nltk()

#Cleaning and preparing email text before sending it to ML model
class TextPreprocessor:
    def __init__(self):
        self.stop_words       = set(stopwords.words('english'))
        self.stemmer          = PorterStemmer()
        self._url_pattern     = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        self._email_pattern   = re.compile(r'\S+@\S+\.\S+')
        self._html_pattern    = re.compile(r'<[^>]+>')
        self._special_pattern = re.compile(r'[^a-zA-Z\s]')
        self._space_pattern   = re.compile(r'\s+')

#take email text and return preprocessed email text
    def preprocess(
        self,
        text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = self._normalize_unicode(text)
        text = self._remove_html(text)
        text = self._remove_urls(text)
        text = self._remove_emails(text)
        text = text.lower()
        text = self._remove_special_chars(text)
        text = self._remove_extra_spaces(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._remove_short(tokens)
        tokens = self._stem(tokens)
        # conert word list to string
        return ' '.join(tokens)
    
#feature engineering
    def extract_metadata(
        self, 
        text: str) -> dict:
        # input validation
        if not text or not isinstance(text, str):
            return {}
        urls_found   = self._url_pattern.findall(text)
        emails_found = self._email_pattern.findall(text)
        # Text ko spaces se tor kar words ki list banata he
        words        = text.split()
        text_lower   = text.lower()
        phishing_kws = [
        # Urgency
        'urgent', 'immediately', 'asap', 'deadline', 'expire', 'expired',
        'limited time', 'act now', 'final warning', 'last chance',
        # Account security
        'verify', 'verification', 'confirm', 'confirmation', 'update',
        'validate', 'security alert', 'suspended', 'deactivated',
        'locked', 'compromised', 'unauthorized access', 'login',
        'reset password', 'account', 'billing',
        # Personal info
        'password', 'username', 'ssn', 'credit card', 'debit card',
        'bank account', 'cvv', 'pin code',
        # Prizes
        'winner', 'won', 'congratulations', 'prize', 'reward', 'bonus',
        'lottery', 'jackpot', 'million', 'free gift', 'claim now',
        # Offers
        'free', 'discount', 'offer', 'promotion', 'special offer',
        # Click
        'click here', 'click link', 'download now', 'open attachment',
        # Threats
        'warning', 'alert', 'legal action', 'lawsuit', 'fine', 'penalty',
        'overdue', 'invoice', 'virus detected', 'malware', 
        # Job scams
        'work from home', 'easy money', 'investment opportunity',
        'bitcoin', 'get rich quick', 'no experience required', 
        # Tech support
        'tech support', 'renewal', 'subscription', 'membership',
        # Shipping
        'package', 'delivery', 'shipment', 'courier',
        # General
        'action required', 'attention needed', 'important message'
        ]
        # create a list of matched kw
        found_kws  = [kw for kw in phishing_kws if kw in text_lower]
        return {
            'url_count':               
                len(urls_found),
            'email_count':             
                len(emails_found),
            'word_count':              
                len(words),
            'char_count':              
                len(text),
            'uppercase_ratio':        
                self._uppercase_ratio(text),
            'exclamation_count':       
                text.count('!'),
            'question_count':          
                text.count('?'),
            'has_urgent_keywords':     
                bool(found_kws),
            'found_phishing_keywords':
                found_kws[:7],
            }

    def _normalize_unicode(
        self, 
        text
        ):
        return unicodedata.normalize(
            'NFKD', text
            ).encode(
                'ascii', 'ignore').decode('ascii')
    def _remove_html(
        self, 
        text
        ):      
        return self._html_pattern.sub(' ', text
            )
    def _remove_urls(
        self, 
        text
        ):       
        return self._url_pattern.sub(' ', text
            )
    def _remove_emails(
        self, 
        text
        ):     
        return self._email_pattern.sub(' ', text
            )
    def _remove_special_chars(
        self, 
        text
        ): 
        return self._special_pattern.sub(' ', text
            )
    def _remove_extra_spaces(
        self, text
        ):  
        return self._space_pattern.sub(' ', text
            ).strip()
    def _tokenize(
        self, 
        text
        ):
        try:    
            return word_tokenize(text)
        except: 
            return text.split()
    def _remove_stopwords(
        self,
        tokens
        ): 
        return [
            t for t in tokens if t not in self.stop_words
            ]
    def _remove_short(
        self, 
        tokens, 
        min_len=3
        ): 
        return [
            t for t in tokens if len(t) >= min_len
            ]
    def _stem(
        self, 
        tokens
        ):            
        return [
            self.stemmer.stem(t) for t in tokens
            ]
    def _uppercase_ratio(
        self, 
        text
        ):
        letters = [
            # only collect the leers
            c for c in text if c.isalpha()
            ]
        if not letters: return 0.0
        return round(
            # counting the capitl letters
            sum(1 for c in letters if c.isupper()) / len(letters), 3
            )

if __name__ == "__main__":
    tp = TextPreprocessor()
    samples = [
        "URGENT: Your PayPal account has been SUSPENDED! Click http://fake-link.com to verify NOW!",
        "Hi John, just checking in about tomorrow's meeting at 10 AM.",
        "Congratulations! You've won $1,000,000. Send your bank details to claim@prize.net"
    ]
    print("*" * 110)
    for s in samples:
        print(f"\nOriginal : {s[:70]}")
        print(f"Processed: {tp.preprocess(s)}")
        meta = tp.extract_metadata(s)
        print(f"Metadata :")
        print(f"  URLs     : {meta['url_count']}")
        print(f"  Emails   : {meta['email_count']}")
        print(f"  Words    : {meta['word_count']}")
        print(f"  CAPS %   : {meta['uppercase_ratio']*100:.1f}%")
        print(f"  Keywords : {meta['found_phishing_keywords']}")
        