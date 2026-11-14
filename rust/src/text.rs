//! Byte-preserving text utilities.

/// Encode UTF-8 text as bytes without normalization or lossy conversion.
pub fn encode(text: &str) -> Vec<u8> {
    text.as_bytes().to_vec()
}

/// Decode bytes as UTF-8, returning an error for invalid sequences.
pub fn decode(bytes: &[u8]) -> std::result::Result<String, std::string::FromUtf8Error> {
    String::from_utf8(bytes.to_vec())
}

/// Return a byte vocabulary in the range 0..=255.
pub fn vocabulary_size() -> usize {
    256
}
