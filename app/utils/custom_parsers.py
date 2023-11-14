from langchain.output_parsers import CommaSeparatedListOutputParser
from typing import List

class SemicolonSeparatedListOutputParser(CommaSeparatedListOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of comma separated values, "
            "eg: `foo; bar; baz`"
        )

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

    @property
    def _type(self) -> str:
        return "semicolon-separated-list"