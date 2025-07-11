"""
Wikipedia Main Topics template provider.
Provides prompts for classifying questions into Wikipedia main topic categories.
"""

from typing import List
from . import TemplateProvider, TemplateRegistry


class WikiMainTopicsTemplate(TemplateProvider):
    """Template provider for Wikipedia main topics classification."""
    
    def __init__(self):
        self._wiki_main_topics = [
            "Academic disciplines",
            "Business",
            "Communication",
            "Concepts",
            "Culture",
            "Economy",
            "Education",
            "Energy",
            "Engineering",
            "Entertainment",
            "Entities",
            "Food and drink", 
            "Geography",
            "Government",
            "Health",
            "History",
            "Human behavior",
            "Humanities",
            "Information",
            "Internet",
            "Knowledge",
            "Language",
            "Law",
            "Life",
            "Lists",
            "Mass media",
            "Mathematics",
            "Military",
            "Nature",
            "People",
            "Politics",
            "Religion",
            "Science",
            "Society",
            "Sports",
            "Technology",
            "Time",
            "Universe"
        ]
        
        # Base examples for few-shot learning
        self._few_shot_base_examples = [
            ("Who is the first chairman of People's Republic of China?", 
             ["People", "History", "Politics", "Government", "Society"]),
            ("If a spacecraft travels at half the speed of light, what detailed time-related phenomena would be observed?", 
             ["Science", "Mathematics", "Knowledge", "Technology", "Engineering"]),
            ("What are the core principles of macroeconomics?",
             ["Economy", "Business", "Concepts", "Academic disciplines", "Society"])
        ]
    
    @property
    def template_id(self) -> str:
        return "wiki_main_topics"
    
    @property
    def description(self) -> str:
        return "Wikipedia main topics classification with 38 predefined categories"
    
    @property
    def choices(self) -> List[str]:
        """Get the list of valid Wikipedia main topics."""
        return self._wiki_main_topics.copy()
    
    def generate_few_shot_examples(self, topk: int) -> str:
        """Generate few-shot examples with top-k domains."""
        formatted_examples = []
        # Use 2 examples for the prompt to keep it concise but informative
        for question, all_domains in self._few_shot_base_examples[:2]:
            # Slice the domains list to match the requested topk
            topk_domains = all_domains[:topk]
            formatted_examples.append(f"[Q]: {question}\n[Topic]: {', '.join(topk_domains)}")
        
        return "\n".join(formatted_examples)
    
    def render(self, question: str, *, topk: int = 3) -> str:
        """Render template with given question and topk."""
        if topk == 1:
            # Single domain template
            return f"""Given a question, identify the most relevant main topic from Wikipedia that it pertains to. Topics are: {",".join(self._wiki_main_topics)}. 
That's to say your answer must be in those topics, and unique.
[Q]: Who is the first chairman of People's Republic of China?
[Topic]: People
[Q]: If a spacecraft travels at half the speed of light, what detailed time-related phenomena would be observed?
[Topic]: Science
[Q]: {question}
[Topic]: """
        else:
            # Multi-domain template with topk
            examples = self.generate_few_shot_examples(topk)
            return f"""Given a question, identify the top {topk} most relevant main topics from Wikipedia that it pertains to. Topics are: {",".join(self._wiki_main_topics)}. 
Your answer must be exactly {topk} topics, separated by commas, ordered by relevance (most relevant first).

{examples}

[Q]: {question}
[Topic]: """


# Create instance and register
wiki_main_topics_template = WikiMainTopicsTemplate()
TemplateRegistry.register(wiki_main_topics_template) 