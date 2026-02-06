import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline, MarianMTModel, MarianTokenizer
from typing import List, Optional
import logging


class Translator:
    """
    Step 3: Translation of NLP text to English
    """
    
    def __init__(self, model_name: str = "facebook/m2m100_418M"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.nlp_stanza = None
        self._initialized = False
        self._translation_cache = {}  # Cache for translations
        
        # Use smaller model for faster translation
        if model_name == "facebook/m2m100_418M":
            self.model_name = "Helsinki-NLP/opus-mt-mul-en"  # Much smaller and faster
        
    def initialize(self) -> None:
        """Initialize translation models"""
        if self._initialized:
            return
            
        try:
            logging.info("Initializing translation models...")
            
            # MarianMT model and tokenizer (much faster)
            self.model = MarianMTModel.from_pretrained(self.model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            
            # Stanza for language detection
            self.nlp_stanza = stanza.Pipeline(
                lang="multilingual", 
                processors="langid",
                download_method=DownloadMethod.REUSE_RESOURCES
            )
            
            self._initialized = True
            logging.info("Translation models initialized successfully")
            
        except Exception as e:
            logging.error(f"Error during model initialization: {e}")
            raise
    
    def _map_language_code(self, lang_code: str) -> str:
        """Map language codes to standard M2M100 codes"""
        lang_mapping = {
            "fro": "fr",  # Old French
            "la": "it",   # Latin
            "nn": "no",   # Norwegian (Nynorsk)
            "kmr": "tr",  # Kurmanji
        }
        return lang_mapping.get(lang_code, lang_code)
    
    def translate_text(self, text: str) -> str:
        """Translate single text to English"""
        if not self._initialized:
            self.initialize()
            
        if text == "" or text is None:
            return text
        
        # Check cache first
        if text in self._translation_cache:
            return self._translation_cache[text]
            
        try:
            # Limit text length to avoid very long processing
            if len(text) > 500:
                text = text[:500] + "..."
                logging.warning(f"Text truncated to 500 characters for translation")
            
            # Language detection
            doc = self.nlp_stanza(text)
            detected_lang = doc.lang
            
            if detected_lang == "en":
                self._translation_cache[text] = text
                return text
            
            # Map language code
            source_lang = self._map_language_code(detected_lang)
            
            # Translation
            encoded_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            generated_tokens = self.model.generate(**encoded_text)
            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            # Cache the result
            self._translation_cache[text] = translated_text
            return translated_text
            
        except Exception as e:
            logging.warning(f"Error during text translation: {e}")
            return text  # Return original text on error
    
    def translate_batch(self, texts: List[str], batch_size: int = 10) -> List[str]:
        """Translate batch of texts"""
        if not self._initialized:
            self.initialize()
            
        translated_texts = []
        unique_texts = list(set(texts))  # Get unique texts
        translation_map = {}
        
        # First, translate unique texts only
        logging.info(f"Translating {len(unique_texts)} unique texts (from {len(texts)} total)")
        
        for i, text in enumerate(unique_texts):
            if i % 5 == 0:  # Update progress more frequently
                logging.info(f"Progress: {i}/{len(unique_texts)} unique texts translated")
            
            translated = self.translate_text(text)
            translation_map[text] = translated
        
        # Now map all original texts
        for text in texts:
            translated_texts.append(translation_map.get(text, text))
        
        logging.info(f"Translation completed. Cache size: {len(self._translation_cache)}")
        return translated_texts
    
    def translate_dataframe_column(
        self, 
        df, 
        column_name: str, 
        new_column_name: Optional[str] = None
    ):
        """Translate DataFrame column"""
        # Input validation
        if df is None or df.empty:
            logging.warning("DataFrame is empty or None")
            return df
            
        if column_name not in df.columns:
            logging.error(f"Column '{column_name}' does not exist in DataFrame")
            raise ValueError(f"Column '{column_name}' not found")
            
        if new_column_name is None:
            new_column_name = f"{column_name}_en"
            
        logging.info(f"Translating column '{column_name}' to English...")
        
        texts = df[column_name].tolist()
        translated_texts = self.translate_batch(texts)
        
        df[new_column_name] = translated_texts
        
        logging.info(f"Translation completed. New column: '{new_column_name}'")
        return df


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Sample data
    data = {
        'text': ['Hello world', 'Bonjour le monde', 'Hola mundo', '']
    }
    df = pd.DataFrame(data)
    
    translator = Translator()
    df_translated = translator.translate_dataframe_column(df, 'text')
    
    print(df_translated)
