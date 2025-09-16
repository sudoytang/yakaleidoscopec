use std::{error::Error, fmt};

use crate::{srcloc::{Pos, SrcLoc, SrcOrig}};


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexErrorKind {
    InvalidByte(u8),
    InvalidNumber(String),
}

impl fmt::Display for LexErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexErrorKind::InvalidByte(b) => write!(f, "Unexpected byte: \\x{:2x}", b),
            LexErrorKind::InvalidNumber(s) => write!(f, "Invalid Number: {s}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    kind: LexErrorKind,
    srcloc: SrcLoc,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "in file {}: {}", self.srcloc, self.kind)
    }
}


impl Error for LexError {}



#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenKind {
    EOF,
    Def,
    Extern,
    Ident,
    Number,
    NumberWithPoint,
    Comment,
    Punct(u8),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    kind: TokenKind,
    lexeme: String,
    srcloc: SrcLoc,
}

impl Token {
    pub fn eof(orig: SrcOrig, pos: Pos) -> Self {
        Self {
            kind: TokenKind::EOF,
            lexeme: String::new(),
            srcloc: SrcLoc::point(orig, pos)
        }    
    }
    pub fn punct(punct: u8, orig: &SrcOrig, pos: Pos) -> Self {
        Self {
            kind: TokenKind::Punct(punct),
            lexeme: String::new(),
            srcloc: SrcLoc::point(orig.clone(), pos),
        }
    }
    pub fn comment(content: String, orig: SrcOrig, start: Pos, end: Pos) -> Self {
        Self {
            kind: TokenKind::Comment,
            lexeme: content,
            srcloc: SrcLoc::new(orig, start, end),
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lexer<'a> {
    srcorig: SrcOrig,
    data: &'a [u8],
    pos: Pos,
}

impl<'a> Lexer<'a> {
    pub fn new(srcorig: SrcOrig, data: &'a [u8]) -> Self {
        Self {
            srcorig,
            data,
            pos: Pos::new(0, 1, 1),
        }
    }

    fn map_ident_to_keyword(lexeme: &str) -> TokenKind {
        match lexeme {
            "def" => TokenKind::Def,
            "extern" => TokenKind::Extern,
            _ => TokenKind::Ident,
        }
    }

}

impl<'a> Lexer<'a> {

    fn make_result(kind: TokenKind, lexeme: String, orig: &SrcOrig, start: Pos, end: Pos) -> Result<Token, LexError> {
        Ok(Token {kind, lexeme, srcloc: SrcLoc::new(orig.clone(), start, end)})
    }

    pub fn get(&mut self) -> Result<Token, LexError> {
        use TokenKind::*;
        self.skip_whitespace();
        let start = self.pos;
        let mut lexeme = String::new();
        let mut kind = match self.pop() {
            Some(b) => {
                lexeme.push(b as char);
                match b {
                    // Identifier 
                    b'A'..=b'Z' | b'a'..=b'z' | b'_' => Ident,
                    // Number
                    b'0'..=b'9' => Number,
                    b'.' => NumberWithPoint,
                    // Comment
                    b'#' => Comment,
                    // Punctuation
                    b'(' | b')' | b'{' | b'}' | b'[' | b']' | b';' | b',' | b'+' | b'-' | b'*' | b'/' | b'=' | b'<' | b'>' | b'!' => {
                        return Ok(Token::punct(b, &self.srcorig, start))
                    },
                    // Unrecognized
                    _ => return Err(LexError {
                        kind: LexErrorKind::InvalidByte(b),
                        srcloc: SrcLoc::point(self.srcorig.clone(), start),
                    })
                }
            }
            None => return Ok(Token::eof(self.srcorig.clone(), start)),
        };
        loop {

            match (kind, self.peek()) {
                // Ident + alpha, num, _ => continue
                (Ident, Some(b @ b'A'..=b'Z' | b @ b'a'..=b'z' | b @ b'_' | b @ b'0'..=b'9')) => lexeme.push(b as char),
                // Ident + Other => return, check keyword
                (Ident, _) => {
                    match Self::map_ident_to_keyword(&lexeme) {
                        Ident => return Self::make_result(kind, lexeme, &self.srcorig, start, self.pos),
                        new_kind => return Self::make_result(new_kind, lexeme, &self.srcorig, start, self.pos),
                    }
                }
                // Number | Number with Point + digit => push
                (Number | NumberWithPoint, Some(b @ b'0'..=b'9')) => lexeme.push(b as char),
                // Number + . => push (NumberWithPoint)
                (Number, Some(b'.')) => { lexeme.push('.'); kind = NumberWithPoint; }
                (NumberWithPoint, Some(b'.')) => return Self::make_result(kind, lexeme, &self.srcorig, start, self.pos),
                // Number | Number with Point + Other => return
                (Number | NumberWithPoint, _) => return Self::make_result(kind, lexeme, &self.srcorig, start, self.pos),

                // Comment + Line Break => return,
                (Comment, Some(b'\n')) => {
                    lexeme.push('\n');
                    return Self::make_result(kind, lexeme, &self.srcorig, start, self.pos);
                }
                (Comment, Some(b)) => lexeme.push(b as char),
                // Other + EOF, return
                (_, None) => return Self::make_result(kind, lexeme, &self.srcorig, start, self.pos),
                (EOF | Def | Extern | Punct(_), _) => assert!(false),
            }
            self.advance();
        }
    }
}

impl<'a> Lexer<'a> {
    fn pop(&mut self) -> Option<u8> {
        let res = self.peek();
        self.advance();
        res
    }
    fn peek(&self) -> Option<u8> {
        self.data.get(self.pos.offset).copied()
    }

    fn get_next_pos(&self) -> Option<Pos> {
        let i = self.pos.offset;
        if i == self.data.len() {
            return None;
        }
        if self.data[i] == b'\n' {
            return Some(Pos::new(i + 1, self.pos.line + 1, 1));
        }
        return Some(Pos::new(i + 1, self.pos.line, self.pos.col + 1));
    }
    fn advance(&mut self) {
        if let Some(p) = self.get_next_pos() {
            self.pos = p;
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_ascii_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_lexer(input: &'_ str) -> Lexer<'_> {
        let srcorig = SrcOrig::Stdin;
        Lexer::new(srcorig, input.as_bytes())
    }

    fn assert_token_kind(mut lexer: Lexer, expected_kind: TokenKind) {
        let token = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token.kind, expected_kind);
    }

    fn assert_token_kind_and_lexeme(mut lexer: Lexer, expected_kind: TokenKind, expected_lexeme: &str) {
        let token = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token.kind, expected_kind);
        assert_eq!(token.lexeme, expected_lexeme);
    }

    fn assert_lex_error(mut lexer: Lexer, expected_error_kind: LexErrorKind) {
        let error = lexer.get().expect_err("Expected lexer error");
        assert_eq!(error.kind, expected_error_kind);
    }

    #[test]
    fn test_eof_branch() {
        // Test empty input - should return EOF token
        let lexer = create_lexer("");
        assert_token_kind(lexer, TokenKind::EOF);
    }

    #[test]
    fn test_eof_after_whitespace() {
        // Test EOF after whitespace - should skip whitespace and return EOF
        let lexer = create_lexer(" \r  \t\n  ");
        assert_token_kind(lexer, TokenKind::EOF);
    }

    #[test]
    fn test_identifier_basic() {
        // Test basic identifier starting with lowercase letter
        let lexer = create_lexer("hello");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "hello");
    }

    #[test]
    fn test_identifier_uppercase() {
        // Test identifier starting with uppercase letter
        let lexer = create_lexer("Hello");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "Hello");
    }

    #[test]
    fn test_identifier_underscore() {
        // Test identifier starting with underscore
        let lexer = create_lexer("_test");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "_test");
    }

    #[test]
    fn test_identifier_mixed() {
        // Test identifier with mixed alphanumeric and underscore
        let lexer = create_lexer("test_var123");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "test_var123");
    }

    #[test]
    fn test_identifier_single_char() {
        // Test single character identifiers
        let lexer = create_lexer("a");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "a");
        
        let lexer = create_lexer("Z");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "Z");
        
        let lexer = create_lexer("_");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "_");
    }

    #[test]
    fn test_keyword_def() {
        // Test 'def' keyword
        let lexer = create_lexer("  def  ");
        assert_token_kind_and_lexeme(lexer, TokenKind::Def, "def");
    }

    #[test]
    fn test_non_keyword_def_() {
        // Test 'def' keyword
        let lexer = create_lexer("def_");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "def_");
    }
    #[test]
    fn test_non_keyword_def1() {
        // Test 'def' keyword
        let lexer = create_lexer("  def1   ");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "def1");
    }
    #[test]
    fn test_keyword_extern() {
        // Test 'extern' keyword
        let lexer = create_lexer("extern");
        assert_token_kind_and_lexeme(lexer, TokenKind::Extern, "extern");
    }

    #[test]
    fn test_identifier_not_keyword() {
        // Test identifiers that start with keyword but are longer
        let lexer = create_lexer("define");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "define");
        
        let lexer = create_lexer("external");
        assert_token_kind_and_lexeme(lexer, TokenKind::Ident, "external");
    }

    #[test]
    fn test_number_integer() {
        // Test basic integer
        let lexer = create_lexer("123");
        assert_token_kind_and_lexeme(lexer, TokenKind::Number, "123");
    }

    #[test]
    fn test_number_single_digit() {
        // Test single digit numbers
        let lexer = create_lexer("0");
        assert_token_kind_and_lexeme(lexer, TokenKind::Number, "0");
        
        let lexer = create_lexer("9");
        assert_token_kind_and_lexeme(lexer, TokenKind::Number, "9");
    }

    #[test]
    fn test_number_with_decimal() {
        // Test number that starts as integer and gets decimal point
        let lexer = create_lexer("123.456");
        assert_token_kind_and_lexeme(lexer, TokenKind::NumberWithPoint, "123.456");
    }

    #[test]
    fn test_number_starting_with_dot() {
        // Test number starting with decimal point
        let lexer = create_lexer(".123");
        assert_token_kind_and_lexeme(lexer, TokenKind::NumberWithPoint, ".123");
    }

    #[test]
    fn test_number_only_dot() {
        // Test just a decimal point
        let lexer = create_lexer(".");
        assert_token_kind_and_lexeme(lexer, TokenKind::NumberWithPoint, ".");
    }

    #[test]
    fn test_number_multiple_dots() {
        // Test number with multiple dots - should stop at second dot
        let mut lexer = create_lexer("123.45.67");
        let token = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token.kind, TokenKind::NumberWithPoint);
        assert_eq!(token.lexeme, "123.45");
        
        // The next token should start with the second dot
        let token2 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token2.kind, TokenKind::NumberWithPoint);
        assert_eq!(token2.lexeme, ".67");
    }

    #[test]
    fn test_comment_basic() {
        // Test basic comment
        let lexer = create_lexer("# this is a comment\n");
        assert_token_kind_and_lexeme(lexer, TokenKind::Comment, "# this is a comment\n");
    }

    #[test]
    fn test_comment_without_newline() {
        // Test comment at end of file without newline
        let lexer = create_lexer("# this is a comment");
        assert_token_kind_and_lexeme(lexer, TokenKind::Comment, "# this is a comment");
    }

    #[test]
    fn test_comment_empty() {
        // Test empty comment
        let lexer = create_lexer("#\n");
        assert_token_kind_and_lexeme(lexer, TokenKind::Comment, "#\n");
        
        let lexer = create_lexer("#");
        assert_token_kind_and_lexeme(lexer, TokenKind::Comment, "#");
    }

    #[test]
    fn test_comment_with_special_chars() {
        // Test comment containing special characters
        let lexer = create_lexer("# comment with !@#$%^&*(){}[]\n");
        assert_token_kind_and_lexeme(lexer, TokenKind::Comment, "# comment with !@#$%^&*(){}[]\n");
    }

    #[test]
    fn test_all_punctuation() {
        // Test all supported punctuation characters
        let punctuation_chars = ['(', ')', '{', '}', '[', ']', ';', ',', '+', '-', '*', '/', '=', '<', '>', '!'];
        
        for punct_char in punctuation_chars {
            let input = punct_char.to_string();
            let lexer = create_lexer(&input);
            assert_token_kind(lexer, TokenKind::Punct(punct_char as u8));
        }
    }

    #[test]
    fn test_punctuation_combinations() {
        // Test that punctuation is returned immediately, not combined
        let mut lexer = create_lexer("()");
        
        let token1 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token1.kind, TokenKind::Punct(b'('));
        
        let token2 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token2.kind, TokenKind::Punct(b')'));
    }

    #[test]
    fn test_invalid_byte_branch() {
        // Test various invalid bytes that should trigger InvalidByte error
        let invalid_inputs = ["@", "$", "%", "^", "&", "~", "`", "\"", "'", "\\", "|", ":"];
        
        for input in invalid_inputs {
            let lexer = create_lexer(input);
            assert_lex_error(lexer, LexErrorKind::InvalidByte(input.as_bytes()[0]));
        }
    }

    #[test]
    fn test_invalid_unicode() {
        // Test non-ASCII characters that should be invalid
        let lexer = create_lexer("π");
        // π in UTF-8 is [207, 128], so first byte is 207
        assert_lex_error(lexer, LexErrorKind::InvalidByte(207));
    }

    #[test]
    fn test_whitespace_skipping() {
        // Test that various whitespace characters are skipped
        let mut lexer = create_lexer("  \t\n\r  hello");
        let token = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token.kind, TokenKind::Ident);
        assert_eq!(token.lexeme, "hello");
    }

    #[test]
    fn test_whitespace_between_tokens() {
        // Test whitespace handling between tokens
        let mut lexer = create_lexer("hello   123");
        
        let token1 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token1.kind, TokenKind::Ident);
        assert_eq!(token1.lexeme, "hello");
        
        let token2 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token2.kind, TokenKind::Number);
        assert_eq!(token2.lexeme, "123");
    }

    #[test]
    fn test_identifier_followed_by_punctuation() {
        // Test identifier immediately followed by punctuation
        let mut lexer = create_lexer("hello(");
        
        let token1 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token1.kind, TokenKind::Ident);
        assert_eq!(token1.lexeme, "hello");
        
        let token2 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token2.kind, TokenKind::Punct(b'('));
    }

    #[test]
    fn test_number_followed_by_identifier() {
        // Test number immediately followed by identifier (should be separate tokens)
        let mut lexer = create_lexer("123abc");
        
        let token1 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token1.kind, TokenKind::Number);
        assert_eq!(token1.lexeme, "123");
        
        let token2 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token2.kind, TokenKind::Ident);
        assert_eq!(token2.lexeme, "abc");
    }

    #[test]
    fn test_decimal_followed_by_identifier() {
        // Test decimal number immediately followed by identifier
        let mut lexer = create_lexer("123.45abc");
        
        let token1 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token1.kind, TokenKind::NumberWithPoint);
        assert_eq!(token1.lexeme, "123.45");
        
        let token2 = lexer.get().expect("Expected successful token parsing");
        assert_eq!(token2.kind, TokenKind::Ident);
        assert_eq!(token2.lexeme, "abc");
    }

    #[test]
    fn test_complex_sequence() {
        // Test a complex sequence of different token types
        let mut lexer = create_lexer("def foo(x) { # comment\n return x + 1 + 123.45; }");
        
        let tokens = vec![
            (TokenKind::Def, "def"),
            (TokenKind::Ident, "foo"),
            (TokenKind::Punct(b'('), ""),
            (TokenKind::Ident, "x"),
            (TokenKind::Punct(b')'), ""),
            (TokenKind::Punct(b'{'), ""),
            (TokenKind::Comment, "# comment\n"),
            (TokenKind::Ident, "return"),
            (TokenKind::Ident, "x"),
            (TokenKind::Punct(b'+'), ""),
            (TokenKind::Number, "1"),
            (TokenKind::Punct(b'+'), ""),
            (TokenKind::NumberWithPoint, "123.45"),
            (TokenKind::Punct(b';'), ""),
            (TokenKind::Punct(b'}'), ""),
            (TokenKind::EOF, ""),
        ];
        
        for (expected_kind, expected_lexeme) in tokens {
            let token = lexer.get().expect("Expected successful token parsing");
            assert_eq!(token.kind, expected_kind);
            if !expected_lexeme.is_empty() {
                assert_eq!(token.lexeme, expected_lexeme);
            }
        }
    }

    #[test]
    fn test_edge_case_empty_after_whitespace() {
        // Test edge case where input becomes empty after skipping whitespace
        let lexer = create_lexer("   ");
        assert_token_kind(lexer, TokenKind::EOF);
    }

    #[test]
    fn test_single_character_tokens() {
        // Test all single character cases that should work
        let test_cases = vec![
            ("a", TokenKind::Ident, "a"),
            ("1", TokenKind::Number, "1"),
            (".", TokenKind::NumberWithPoint, "."),
            ("#", TokenKind::Comment, "#"),
            ("(", TokenKind::Punct(b'('), ""),
        ];
        
        for (input, expected_kind, expected_lexeme) in test_cases {
            let lexer = create_lexer(input);
            if expected_lexeme.is_empty() {
                assert_token_kind(lexer, expected_kind);
            } else {
                assert_token_kind_and_lexeme(lexer, expected_kind, expected_lexeme);
            }
        }
    }
}
