"""Tokenizer for the Attractor DOT DSL subset."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    DIGRAPH = auto()
    SUBGRAPH = auto()
    GRAPH = auto()
    NODE = auto()
    EDGE = auto()
    IDENTIFIER = auto()
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    DURATION = auto()
    ARROW = auto()        # ->
    LBRACE = auto()       # {
    RBRACE = auto()       # }
    LBRACKET = auto()     # [
    RBRACKET = auto()     # ]
    EQUALS = auto()       # =
    COMMA = auto()        # ,
    SEMICOLON = auto()    # ;
    DOT = auto()          # .
    EOF = auto()


KEYWORDS = {
    "digraph": TokenType.DIGRAPH,
    "subgraph": TokenType.SUBGRAPH,
    "graph": TokenType.GRAPH,
    "node": TokenType.NODE,
    "edge": TokenType.EDGE,
    "true": TokenType.BOOLEAN,
    "false": TokenType.BOOLEAN,
}

DURATION_PATTERN = re.compile(r"^-?\d+(ms|s|m|h|d)$")
FLOAT_PATTERN = re.compile(r"^-?\d*\.\d+$")
INT_PATTERN = re.compile(r"^-?\d+$")


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


class LexerError(Exception):
    pass


_COMMENT_RE = re.compile(
    r'//[^\n]*|/\*.*?\*/',
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    """Remove // line comments and /* block */ comments."""
    def _replace(m: re.Match) -> str:
        text = m.group(0)
        # preserve newlines inside block comments for line counting
        return "\n" * text.count("\n")
    return _COMMENT_RE.sub(_replace, src)


def tokenize(src: str) -> list[Token]:
    """Convert DOT source text into a list of tokens."""
    src = strip_comments(src)
    tokens: list[Token] = []
    pos = 0
    line = 1
    col = 1
    n = len(src)

    while pos < n:
        c = src[pos]

        # Whitespace
        if c in " \t\r":
            pos += 1
            col += 1
            continue
        if c == "\n":
            pos += 1
            line += 1
            col = 1
            continue

        # Arrow ->
        if c == "-" and pos + 1 < n and src[pos + 1] == ">":
            tokens.append(Token(TokenType.ARROW, "->", line, col))
            pos += 2
            col += 2
            continue

        # Single-char punctuation
        single = {
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            "=": TokenType.EQUALS,
            ",": TokenType.COMMA,
            ";": TokenType.SEMICOLON,
            ".": TokenType.DOT,
        }
        if c in single:
            tokens.append(Token(single[c], c, line, col))
            pos += 1
            col += 1
            continue

        # Quoted string
        if c == '"':
            start_line, start_col = line, col
            pos += 1
            col += 1
            buf = []
            while pos < n and src[pos] != '"':
                ch = src[pos]
                if ch == "\\":
                    pos += 1
                    col += 1
                    if pos >= n:
                        raise LexerError(f"Unterminated string escape at {start_line}:{start_col}")
                    esc = src[pos]
                    if esc == "n":
                        buf.append("\n")
                    elif esc == "t":
                        buf.append("\t")
                    elif esc == "\\":
                        buf.append("\\")
                    elif esc == '"':
                        buf.append('"')
                    else:
                        buf.append("\\" + esc)
                    pos += 1
                    col += 1
                else:
                    if ch == "\n":
                        line += 1
                        col = 1
                    else:
                        col += 1
                    buf.append(ch)
                    pos += 1
            if pos >= n:
                raise LexerError(f"Unterminated string at {start_line}:{start_col}")
            pos += 1  # closing quote
            col += 1
            tokens.append(Token(TokenType.STRING, "".join(buf), start_line, start_col))
            continue

        # Number or duration: starts with digit or minus-digit
        if c.isdigit() or (c == "-" and pos + 1 < n and src[pos + 1].isdigit()):
            start_pos, start_line, start_col = pos, line, col
            # collect the numeric part
            while pos < n and (src[pos].isdigit() or src[pos] in ".-"):
                # Check if it's a duration suffix starting
                if src[pos] == "-" and pos != start_pos:
                    break
                pos += 1
                col += 1
            # Check for duration suffix
            suffix = ""
            if pos < n and src[pos] in "smhd":
                if src[pos] == "m" and pos + 1 < n and src[pos + 1] == "s":
                    suffix = "ms"
                    pos += 2
                    col += 2
                else:
                    suffix = src[pos]
                    pos += 1
                    col += 1
            raw = src[start_pos:pos - len(suffix)] + suffix
            if suffix:
                tokens.append(Token(TokenType.DURATION, raw, start_line, start_col))
            elif "." in raw:
                tokens.append(Token(TokenType.FLOAT, raw, start_line, start_col))
            else:
                tokens.append(Token(TokenType.INTEGER, raw, start_line, start_col))
            continue

        # Identifier or keyword
        if c.isalpha() or c == "_":
            start_pos, start_line, start_col = pos, line, col
            while pos < n and (src[pos].isalnum() or src[pos] in "_"):
                pos += 1
                col += 1
            word = src[start_pos:pos]
            tt = KEYWORDS.get(word, TokenType.IDENTIFIER)
            tokens.append(Token(tt, word, start_line, start_col))
            continue

        raise LexerError(f"Unexpected character {c!r} at {line}:{col}")

    tokens.append(Token(TokenType.EOF, "", line, col))
    return tokens
