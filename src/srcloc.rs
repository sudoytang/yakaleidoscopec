use std::{fmt::{self}, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pos {
    pub offset: usize,
    pub line: usize,
    pub col: usize,
}

impl Pos {
    pub fn new(offset: usize, line: usize, col: usize) -> Self {
        Self { offset, line, col }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub start: Pos,
    pub end: Pos,
}

impl Span {
    pub fn point(pos: Pos) -> Self {
        Self {
            start: pos,
            end: pos,
        }
    }

    pub fn new(start: Pos, end: Pos) -> Self {
        Self {
            start,
            end,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SrcOrig {
    File(PathBuf),
    Stdin,
}

impl fmt::Display for SrcOrig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SrcOrig::File(path_buf) => write!(f, "{}", path_buf.display()),
            SrcOrig::Stdin => write!(f, "<stdin>"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrcLoc {
    pub orig: SrcOrig,
    pub span: Span,
}

impl SrcLoc {
    pub fn point(orig: SrcOrig, pos: Pos) -> Self {
        Self {
            orig,
            span: Span::point(pos),
        }
    }
    pub fn new(orig: SrcOrig, start: Pos, end: Pos) -> Self {
        Self {
            orig,
            span: Span::new(start, end),
        }
    }
}

impl fmt::Display for SrcLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let l1 = self.span.start.line;
        let l2 = self.span.end.line;
        let c1 = self.span.start.col;
        let c2 = self.span.end.col;
        if l1 == l2 {
            if c1 == c2 {
                write!(f, "{}:{}:{}", self.orig, l1, c1)
            } else {
                write!(f, "{}:{}:{}-{}", self.orig, l1, c1, c2)
            }
        } else {
            write!(f, "{}:({}:{})-({}:{})", self.orig, l1, c1, l2, c2)
        }
    }
}

