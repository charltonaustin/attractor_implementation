"""Recursive descent parser for the Attractor DOT DSL subset."""
from __future__ import annotations

import re
from typing import Any

from attractor.model.types import Edge, Graph, Node
from attractor.parser.lexer import Token, TokenType, tokenize


class ParseError(Exception):
    pass


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------
    # Token navigation helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.type != TokenType.EOF:
            self._pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._advance()
        if tok.type != tt:
            raise ParseError(
                f"Expected {tt.name} but got {tok.type.name} ({tok.value!r}) "
                f"at line {tok.line}:{tok.col}"
            )
        return tok

    def _match(self, *types: TokenType) -> Token | None:
        if self._peek().type in types:
            return self._advance()
        return None

    def _eat_optional(self, tt: TokenType) -> None:
        if self._peek().type == tt:
            self._advance()

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def parse_graph(self) -> Graph:
        self._expect(TokenType.DIGRAPH)
        name_tok = self._expect(TokenType.IDENTIFIER)
        graph_id = name_tok.value
        graph = Graph(id=graph_id)
        self._expect(TokenType.LBRACE)
        # global defaults
        node_defaults: dict[str, Any] = {}
        edge_defaults: dict[str, Any] = {}
        self._parse_stmts(graph, node_defaults, edge_defaults)
        self._expect(TokenType.RBRACE)
        self._expect(TokenType.EOF)
        return graph

    # ------------------------------------------------------------------
    # Statement list
    # ------------------------------------------------------------------

    def _parse_stmts(
        self,
        graph: Graph,
        node_defaults: dict[str, Any],
        edge_defaults: dict[str, Any],
    ) -> None:
        while self._peek().type not in (TokenType.RBRACE, TokenType.EOF):
            self._parse_stmt(graph, node_defaults, edge_defaults)
            self._eat_optional(TokenType.SEMICOLON)

    def _parse_stmt(
        self,
        graph: Graph,
        node_defaults: dict[str, Any],
        edge_defaults: dict[str, Any],
    ) -> None:
        tok = self._peek()

        # graph [ ... ]  or graph-level key=value
        if tok.type == TokenType.GRAPH:
            self._advance()
            if self._peek().type == TokenType.LBRACKET:
                attrs = self._parse_attr_block()
                graph.attrs.update(attrs)
            # else: bare "graph" keyword without brackets - ignore
            return

        # node [ ... ]  (defaults)
        if tok.type == TokenType.NODE:
            self._advance()
            if self._peek().type == TokenType.LBRACKET:
                node_defaults.update(self._parse_attr_block())
            return

        # edge [ ... ]  (defaults)
        if tok.type == TokenType.EDGE:
            self._advance()
            if self._peek().type == TokenType.LBRACKET:
                edge_defaults.update(self._parse_attr_block())
            return

        # subgraph
        if tok.type == TokenType.SUBGRAPH:
            self._parse_subgraph(graph, node_defaults, edge_defaults)
            return

        # identifier = value  (graph-level attr)
        if tok.type == TokenType.IDENTIFIER:
            next_tok = self._tokens[self._pos + 1] if self._pos + 1 < len(self._tokens) else None
            # Check for edge statement: identifier -> ...
            if next_tok and next_tok.type == TokenType.ARROW:
                self._parse_edge_stmt(graph, node_defaults, edge_defaults)
                return
            # Check for graph-level attr: identifier = value
            if next_tok and next_tok.type == TokenType.EQUALS:
                key = self._advance().value
                self._expect(TokenType.EQUALS)
                val = self._parse_value()
                graph.attrs[key] = val
                return
            # Node statement: identifier [attrs] or bare identifier
            self._parse_node_or_edge_stmt(graph, node_defaults, edge_defaults)
            return

        # Unknown token - skip to avoid infinite loop
        self._advance()

    def _parse_node_or_edge_stmt(
        self,
        graph: Graph,
        node_defaults: dict[str, Any],
        edge_defaults: dict[str, Any],
    ) -> None:
        node_id = self._expect(TokenType.IDENTIFIER).value
        if self._peek().type == TokenType.ARROW:
            # Edge chain starting with this identifier
            self._parse_edge_chain(node_id, graph, node_defaults, edge_defaults)
            return
        # Node definition
        attrs = {}
        if self._peek().type == TokenType.LBRACKET:
            attrs = self._parse_attr_block()
        merged = {**node_defaults, **attrs}
        if node_id not in graph.nodes:
            graph.nodes[node_id] = Node(id=node_id, attrs=merged)
        else:
            graph.nodes[node_id].attrs.update(merged)

    def _parse_edge_stmt(
        self,
        graph: Graph,
        node_defaults: dict[str, Any],
        edge_defaults: dict[str, Any],
    ) -> None:
        node_id = self._expect(TokenType.IDENTIFIER).value
        self._parse_edge_chain(node_id, graph, node_defaults, edge_defaults)

    def _parse_edge_chain(
        self,
        first: str,
        graph: Graph,
        node_defaults: dict[str, Any],
        edge_defaults: dict[str, Any],
    ) -> None:
        ids = [first]
        while self._peek().type == TokenType.ARROW:
            self._advance()  # consume ->
            ids.append(self._expect(TokenType.IDENTIFIER).value)

        attrs: dict[str, Any] = {}
        if self._peek().type == TokenType.LBRACKET:
            attrs = self._parse_attr_block()

        edge_attrs = {**edge_defaults, **attrs}
        # Ensure all referenced nodes exist
        for nid in ids:
            if nid not in graph.nodes:
                graph.nodes[nid] = Node(id=nid, attrs=dict(node_defaults))

        # Expand chained edges
        for i in range(len(ids) - 1):
            graph.edges.append(
                Edge(from_node=ids[i], to_node=ids[i + 1], attrs=dict(edge_attrs))
            )

    # ------------------------------------------------------------------
    # Subgraph
    # ------------------------------------------------------------------

    def _parse_subgraph(
        self,
        graph: Graph,
        parent_node_defaults: dict[str, Any],
        parent_edge_defaults: dict[str, Any],
    ) -> None:
        self._expect(TokenType.SUBGRAPH)
        # Optional subgraph ID
        sub_label = ""
        if self._peek().type == TokenType.IDENTIFIER:
            sub_label = self._advance().value
        self._expect(TokenType.LBRACE)

        # Scoped defaults inherit from parent
        sub_node_defaults = dict(parent_node_defaults)
        sub_edge_defaults = dict(parent_edge_defaults)
        sub_attrs: dict[str, Any] = {}

        # Collect nodes in this subgraph for class derivation
        nodes_before = set(graph.nodes.keys())

        # Parse inner statements
        while self._peek().type not in (TokenType.RBRACE, TokenType.EOF):
            tok = self._peek()
            if tok.type == TokenType.NODE:
                self._advance()
                if self._peek().type == TokenType.LBRACKET:
                    sub_node_defaults.update(self._parse_attr_block())
            elif tok.type == TokenType.EDGE:
                self._advance()
                if self._peek().type == TokenType.LBRACKET:
                    sub_edge_defaults.update(self._parse_attr_block())
            elif tok.type == TokenType.GRAPH:
                self._advance()
                if self._peek().type == TokenType.LBRACKET:
                    sub_attrs.update(self._parse_attr_block())
                elif self._peek().type == TokenType.IDENTIFIER:
                    # bare graph attr inside subgraph - treat as subgraph label
                    pass
            elif tok.type == TokenType.IDENTIFIER:
                next_tok = self._tokens[self._pos + 1] if self._pos + 1 < len(self._tokens) else None
                if next_tok and next_tok.type == TokenType.EQUALS:
                    key = self._advance().value
                    self._expect(TokenType.EQUALS)
                    val = self._parse_value()
                    sub_attrs[key] = val
                else:
                    self._parse_node_or_edge_stmt(graph, sub_node_defaults, sub_edge_defaults)
            elif tok.type == TokenType.SUBGRAPH:
                self._parse_subgraph(graph, sub_node_defaults, sub_edge_defaults)
            else:
                self._advance()
            self._eat_optional(TokenType.SEMICOLON)

        self._expect(TokenType.RBRACE)

        # Derive CSS class from subgraph label attribute
        label_for_class = sub_attrs.get("label", sub_label)
        if label_for_class:
            derived_class = _derive_class_from_label(label_for_class)
            # Apply to newly added nodes
            nodes_after = set(graph.nodes.keys())
            for nid in nodes_after - nodes_before:
                node = graph.nodes[nid]
                existing = node.attrs.get("class", "")
                if existing:
                    node.attrs["class"] = f"{existing},{derived_class}"
                else:
                    node.attrs["class"] = derived_class

    # ------------------------------------------------------------------
    # Attribute block
    # ------------------------------------------------------------------

    def _parse_attr_block(self) -> dict[str, Any]:
        self._expect(TokenType.LBRACKET)
        attrs: dict[str, Any] = {}
        while self._peek().type != TokenType.RBRACKET:
            if self._peek().type == TokenType.EOF:
                raise ParseError("Unterminated attribute block")
            key = self._parse_key()
            self._expect(TokenType.EQUALS)
            val = self._parse_value()
            attrs[key] = val
            if self._peek().type == TokenType.COMMA:
                self._advance()
        self._expect(TokenType.RBRACKET)
        return attrs

    def _parse_key(self) -> str:
        """Parse a key: identifier or qualified (identifier.identifier...)"""
        parts = [self._expect(TokenType.IDENTIFIER).value]
        while self._peek().type == TokenType.DOT:
            self._advance()
            parts.append(self._expect(TokenType.IDENTIFIER).value)
        return ".".join(parts)

    def _parse_value(self) -> Any:
        tok = self._peek()
        if tok.type == TokenType.STRING:
            self._advance()
            return tok.value
        if tok.type == TokenType.BOOLEAN:
            self._advance()
            return tok.value == "true"
        if tok.type == TokenType.DURATION:
            self._advance()
            return tok.value
        if tok.type == TokenType.FLOAT:
            self._advance()
            return float(tok.value)
        if tok.type == TokenType.INTEGER:
            self._advance()
            return int(tok.value)
        if tok.type == TokenType.IDENTIFIER:
            self._advance()
            return tok.value
        raise ParseError(
            f"Expected a value but got {tok.type.name} ({tok.value!r}) at {tok.line}:{tok.col}"
        )


def _derive_class_from_label(label: str) -> str:
    """Convert subgraph label to CSS class name."""
    cls = label.lower().replace(" ", "-")
    cls = re.sub(r"[^a-z0-9-]", "", cls)
    return cls


def parse(src: str) -> Graph:
    """Parse DOT source text and return a Graph model."""
    tokens = tokenize(src)
    parser = _Parser(tokens)
    return parser.parse_graph()


def parse_file(path: str) -> Graph:
    """Read a .dot file and parse it."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    return parse(src)
