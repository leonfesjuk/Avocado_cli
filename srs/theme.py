from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    String,
    Text as PygmentsText,
)
from rich.syntax import PygmentsSyntaxTheme

AVOCADO_BORDER = "#568203"
AVOCADO_LIME = "#B6D94C"
CODE_BACKGROUND = "#2b2b2b"


class AvocadoCodeStyle(Style):
    background_color = CODE_BACKGROUND
    default_style = "#e6e6e6"
    styles = {
        PygmentsText: "#e6e6e6",
        Comment: "italic #7f8c8d",
        Error: "#ff5555",
        Keyword: "bold #ff79c6",
        Keyword.Type: "#8be9fd",
        Name: "#f8f8f2",
        Name.Builtin: "#8be9fd",
        Name.Class: "bold #8be9fd",
        Name.Function: "#50fa7b",
        Name.Variable: "#f8f8f2",
        Number: "#bd93f9",
        Operator: "#ff79c6",
        String: "#f1fa8c",
        Generic.Error: "#ff5555",
        Generic.Deleted: "#ff5555",
        Generic.Inserted: "#50fa7b",
    }


AVOCADO_CODE_THEME = PygmentsSyntaxTheme(AvocadoCodeStyle)
