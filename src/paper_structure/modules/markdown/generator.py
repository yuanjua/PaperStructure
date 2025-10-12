"""
Markdown Generation Module
Converts structured document data to Markdown format
"""

from typing import List, Dict, Any


class MarkdownGenerator:
    """Generate Markdown from structured document data"""
    
    def __init__(self):
        """Initialize markdown generator"""
        self.type_handlers = {
            'Title': self._format_title,
            'Section-header': self._format_section,
            'Text': self._format_text,
            'List-item': self._format_list_item,
            'Table': self._format_table,
            'Formula': self._format_formula,
            'Picture': self._format_picture,
            'Caption': self._format_caption,
            'Page-header': self._format_header,
            'Page-footer': self._format_footer,
            'Footnote': self._format_footnote,
        }
        
    def generate(self, elements: List[Dict[str, Any]]) -> str:
        """
        Generate markdown from structured elements
        
        Args:
            elements: List of elements with type, bbox, and content
            
        Returns:
            Markdown formatted string
        """
        markdown_parts = []
        
        for element in elements:
            element_type = element.get('type', 'Text')
            handler = self.type_handlers.get(element_type, self._format_default)
            md_text = handler(element)
            if md_text:
                markdown_parts.append(md_text)
        
        return '\n\n'.join(markdown_parts)
    
    def _format_title(self, element: Dict[str, Any]) -> str:
        """Format title as H1"""
        content = element.get('content', '')
        return f"# {content}" if content else ""
    
    def _format_section(self, element: Dict[str, Any]) -> str:
        """Format section header as H2"""
        content = element.get('content', '')
        return f"## {content}" if content else ""
    
    def _format_text(self, element: Dict[str, Any]) -> str:
        """Format regular text"""
        content = element.get('content', '')
        return content
    
    def _format_list_item(self, element: Dict[str, Any]) -> str:
        """Format list item"""
        content = element.get('content', '')
        return f"- {content}" if content else ""
    
    def _format_table(self, element: Dict[str, Any]) -> str:
        """Format table"""
        content = element.get('content', '')
        if content:
            return f"**[Table]**\n\n{content}"
        return "**[Table detected]**"
    
    def _format_formula(self, element: Dict[str, Any]) -> str:
        """Format formula"""
        content = element.get('content', '')
        if content and content.startswith('$'):
            return content
        elif content:
            return f"$$\n{content}\n$$"
        return "$$[Formula]$$"
    
    def _format_picture(self, element: Dict[str, Any]) -> str:
        """Format picture/figure"""
        return "![Figure]"
    
    def _format_caption(self, element: Dict[str, Any]) -> str:
        """Format caption"""
        content = element.get('content', '')
        return f"*{content}*" if content else ""
    
    def _format_header(self, element: Dict[str, Any]) -> str:
        """Format page header (usually skip)"""
        return ""
    
    def _format_footer(self, element: Dict[str, Any]) -> str:
        """Format page footer (usually skip)"""
        return ""
    
    def _format_footnote(self, element: Dict[str, Any]) -> str:
        """Format footnote"""
        content = element.get('content', '')
        return f"[^footnote]: {content}" if content else ""
    
    def _format_default(self, element: Dict[str, Any]) -> str:
        """Default formatting"""
        content = element.get('content', '')
        return content
    
    def __repr__(self):
        return "MarkdownGenerator()"

