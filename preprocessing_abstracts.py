import json
from pydantic import BaseModel, Field, AliasPath
from typing import Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Get the logger for this module
logger = logging.getLogger(__name__)


class DataModel(BaseModel):
    """
    Data model that extracts and validates fields from the JSON input using pydantic.

    This model uses AliasPath to extract nested data from a structured JSON object.

    Attributes:
    -----------
    award_id : Optional[int]
        The unique identifier for an award, extracted from 'rootTag > Award > AwardID'.
    award_title : str
        Title of the award, extracted from 'rootTag > Award > AwardTitle'.
    abstract : Optional[str]
        Abstract or description of the award, extracted from 'rootTag > Award > AbstractNarration'.
    """
    award_id: Optional[int] = Field(validation_alias=AliasPath('rootTag', 'Award', 'AwardID'))
    award_title: str = Field(validation_alias=AliasPath('rootTag', 'Award', 'AwardTitle'))
    abstract: Optional[str] = Field(validation_alias=AliasPath('rootTag', 'Award', 'AbstractNarration'))


def _download_nltk_resources() -> None:
    """
    Downloads the required NLTK resources if they are not already available.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logging.info('Downloading the nltk stopwords')
        nltk.download('stopwords')


class AbstractPreprocessor:
    """
    A class to preprocess and clean the abstracts found in JSON data files.

    Attributes:
    -----------
    json_file : str
        The path to the JSON file containing the abstract data.
    set_stopwords : set
        The set of stopwords including custom and NLTK-provided stopwords.
    phrase_to_remove : str
        A specific phrase to be removed from abstracts.
    lemmatizer : WordNetLemmatizer
        NLTK's lemmatizer to normalize tokens.

    Methods:
    --------
    create_unique_abstract_list():
        Creates a list of unique abstracts from the JSON file.
    preprocess_text(text: str):
        Cleans and preprocesses a single abstract.
    preprocess_abstracts():
        Preprocesses all abstracts in the dataset.
    """

    def __init__(self, json_file: str = 'data/data.json'):
        self.json_file = json_file

        # Define custom stop words
        custom_stop_words = ['project', 'data', 'research', 'student', 'award', 'program', 'using', 'impact', 'new',
                             'support', 'nsf', 'foundation', 'study', 'science', 'use', 'develop', 'development',
                             'model', 'used']

        # Ensure NLTK resources are downloaded
        _download_nltk_resources()
        nltk_stopwords = stopwords.words('english')
        self.set_stopwords = set(nltk_stopwords).union(custom_stop_words)

        # Define a predefined phrase to be removed
        self.phrase_to_remove = ("This award reflects NSF's statutory mission and has been deemed worthy of support "
                                 "through evaluation using the Foundation's intellectual merit and broader impacts "
                                 "review criteria")

        self.lemmatizer = WordNetLemmatizer()

    def _create_unique_abstract_list(self) -> list:
        """
        Reads JSON data, extracts the abstracts, and creates a list of unique abstracts.

        Returns:
        --------
        list of str
            A list containing unique abstracts.
        """
        logging.info('Initializing the building of a unique abstracts list.')
        try:
            with open(self.json_file, mode='r') as f:
                data = json.load(f)
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise

        # Load data using the DataModel and filter unique abstracts
        model = [DataModel(**item).dict(exclude_none=True) for item in data]
        abstracts_set = {entry.get('abstract') for entry in model if
                         entry.get('abstract') and entry.get('abstract').strip()}
        unique_abstract_list = list(abstracts_set)
        logging.info('Completed building of unique abstracts list.')
        return unique_abstract_list

    def preprocess_text(self, text: str) -> str:
        """
        Cleans and preprocesses a single abstract by removing HTML tags, phrases,
        and stopwords, and then lemmatizing the tokens.

        Parameters:
        -----------
        text : str
            The abstract text to be cleaned and preprocessed.

        Returns:
        --------
        str
            The cleaned and preprocessed text.
        """
        text = re.sub(pattern=r'<br\s*/?>', repl=' ', string=text)  # Remove HTML line breaks
        text = re.sub(pattern=r'&lt;br/&gt;', repl=' ', string=text)  # Handle HTML encoded breaks
        text = re.sub(pattern=self.phrase_to_remove, repl=' ', string=text)  # Remove the predefined phrase

        # Tokenize, lowercase, remove stopwords and lemmatize
        word_tokens = word_tokenize(text.lower())
        filtered_before_lemma = [w for w in word_tokens if w.isalpha() and w not in self.set_stopwords]
        lemmatized_text = [self.lemmatizer.lemmatize(token) for token in filtered_before_lemma]
        filtered_after_lemma = [w for w in lemmatized_text if w not in self.set_stopwords]

        return ' '.join(filtered_after_lemma)

    def preprocess_abstracts(self) -> list:
        """
        Preprocesses all abstracts by cleaning, tokenizing, removing stopwords, and lemmatizing the text.

        Returns:
        --------
        list of str
            A list of preprocessed abstracts.
        """
        unique_abstracts = self._create_unique_abstract_list()
        logging.info('Preprocessing abstracts.')
        preprocessed_abstracts = [self.preprocess_text(a) for a in unique_abstracts]
        logging.info('Finish preprocessing of abstracts.')
        return preprocessed_abstracts
