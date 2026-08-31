"""tests/test_multi_language_parser.py — Tests for multi-language symbol extraction and detection."""

import pytest
from language_detector import detect_language, is_code_file
from symbol_parser import extract_symbols_from_source, Symbol


def test_language_detector():
    assert detect_language("server.go") == "go"
    assert detect_language("lib.rs") == "rust"
    assert detect_language("App.java") == "java"
    assert detect_language("index.ts") == "typescript"
    assert detect_language("main.py") == "python"
    assert detect_language("doc.md") == "markdown"
    assert is_code_file("server.go") is True
    assert is_code_file("doc.md") is False


def test_go_symbol_extraction():
    go_code = """package main

type ServerConfig struct {
    Port int
    Host string
}

type Greeter interface {
    Greet(name string) string
}

func (s *ServerConfig) Start() error {
    return nil
}

func CalculateTotal(prices []float64) float64 {
    var sum float64
    for _, p := range prices {
        sum += p
    }
    return sum
}
"""
    symbols = extract_symbols_from_source(go_code, filename="main.go")
    names = {s.name: s for s in symbols}

    assert "ServerConfig" in names
    assert names["ServerConfig"].symbol_type == "struct"
    assert names["ServerConfig"].language == "go"

    assert "Greeter" in names
    assert names["Greeter"].symbol_type == "interface"

    assert "Start" in names
    assert names["Start"].symbol_type == "method"

    assert "CalculateTotal" in names
    assert names["CalculateTotal"].symbol_type == "function"


def test_rust_symbol_extraction():
    rust_code = """pub struct User {
    pub id: u64,
    pub name: String,
}

pub enum Role {
    Admin,
    Member,
}

pub trait Authenticatable {
    fn verify_token(&self, token: &str) -> bool;
}

pub async fn fetch_user_data(user_id: u64) -> Result<User, String> {
    Ok(User { id: user_id, name: "Alice".into() })
}
"""
    symbols = extract_symbols_from_source(rust_code, filename="auth.rs")
    names = {s.name: s for s in symbols}

    assert "User" in names
    assert names["User"].symbol_type == "struct"
    assert names["User"].language == "rust"

    assert "Role" in names
    assert names["Role"].symbol_type == "enum"

    assert "Authenticatable" in names
    assert names["Authenticatable"].symbol_type == "trait"

    assert "fetch_user_data" in names
    assert names["fetch_user_data"].symbol_type == "function"


def test_java_symbol_extraction():
    java_code = """package com.example;

public class OrderService {
    private final String dbUrl;

    public OrderService(String dbUrl) {
        this.dbUrl = dbUrl;
    }

    public Order findById(long orderId) {
        return null;
    }
}

public interface PaymentGateway {
    boolean processPayment(double amount);
}
"""
    symbols = extract_symbols_from_source(java_code, filename="OrderService.java")
    classes = [s for s in symbols if s.symbol_type == "class"]
    interfaces = [s for s in symbols if s.symbol_type == "interface"]
    methods = [s for s in symbols if s.symbol_type == "method"]

    assert any(c.name == "OrderService" for c in classes)
    assert any(i.name == "PaymentGateway" for i in interfaces)
    assert any(m.name == "findById" for m in methods)



def test_python_symbol_extraction_with_signature():
    py_code = """class DatabasePool:
    \"\"\"Connection pool manager.\"\"\"
    def __init__(self, size: int = 10):
        self.size = size

    async def acquire(self):
        return None
"""
    symbols = extract_symbols_from_source(py_code, filename="db.py")
    names = {s.name: s for s in symbols}

    assert "DatabasePool" in names
    assert names["DatabasePool"].symbol_type == "class"
    assert names["DatabasePool"].docstring == "Connection pool manager."
    assert names["DatabasePool"].language == "python"

    assert "acquire" in names
    assert names["acquire"].symbol_type == "method"
    assert names["acquire"].parent_class == "DatabasePool"
