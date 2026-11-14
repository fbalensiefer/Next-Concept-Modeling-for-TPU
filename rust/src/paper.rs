//! Paper source expansion and bibliography generation in Rust.
use serde_json::Value;
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

fn expand(base: &Path, path: &Path, seen: &mut HashSet<PathBuf>) -> Result<String, String> {
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    };
    if path
        .file_name()
        .and_then(|x| x.to_str())
        .is_some_and(|name| name.starts_with("bibliography_"))
        || !seen.insert(path.clone())
    {
        return Ok(String::new());
    }
    let text = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let mut output = String::new();
    let mut remaining = text.as_str();
    while let Some(start) = remaining.find("\\input{") {
        output.push_str(&remaining[..start]);
        let after = &remaining[start + 7..];
        let end = after.find('}').ok_or("unterminated LaTeX input")?;
        let mut child = PathBuf::from(&after[..end]);
        if child.extension().is_none() {
            child.set_extension("tex");
        }
        output.push_str(&expand(base, &child, seen)?);
        remaining = &after[end + 1..];
    }
    output.push_str(remaining);
    Ok(output)
}
fn citations(text: &str) -> Vec<String> {
    let mut keys = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = text[cursor..].find("\\cite") {
        let start = cursor + offset;
        let mut i = start + 5;
        while i < text.len() && text.as_bytes()[i].is_ascii_lowercase() {
            i += 1;
        }
        while i < text.len() && text.as_bytes()[i] == b'[' {
            i += 1;
            while i < text.len() && text.as_bytes()[i] != b']' {
                i += 1;
            }
            if i < text.len() {
                i += 1;
            }
        }
        if i < text.len() && text.as_bytes()[i] == b'{' {
            i += 1;
            if let Some(end) = text[i..].find('}') {
                for key in text[i..i + end]
                    .split(',')
                    .map(str::trim)
                    .filter(|k| !k.is_empty())
                {
                    if !keys.iter().any(|x| x == key) {
                        keys.push(key.to_owned());
                    }
                }
                cursor = i + end + 1;
                continue;
            }
        }
        cursor = start + 5;
    }
    keys
}
fn braced_commands<'a>(text: &'a str, commands: &[&str]) -> Vec<&'a str> {
    let mut values = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let Some(relative) = text[cursor..].find('\\') else {
            break;
        };
        let start = cursor + relative;
        let command = commands
            .iter()
            .find(|name| text[start..].starts_with(&format!("\\{name}{{")));
        if let Some(command) = command {
            let begin = start + command.len() + 2;
            if let Some(end) = text[begin..].find('}') {
                values.push(&text[begin..begin + end]);
                cursor = begin + end + 1;
                continue;
            }
        }
        cursor = start + 1;
    }
    values
}
fn hash_file(path: &Path) -> Result<String, String> {
    Ok(crate::corpus::sha256(
        &fs::read(path).map_err(|e| e.to_string())?,
    ))
}
fn files_under(
    root: &Path,
    folders: &[&str],
    extensions: &[&str],
    skip_document_manifest: bool,
) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for folder in folders {
        let mut pending = vec![root.join(folder)];
        while let Some(directory) = pending.pop() {
            if !directory.exists() {
                continue;
            }
            for item in fs::read_dir(directory).map_err(|e| e.to_string())? {
                let path = item.map_err(|e| e.to_string())?.path();
                if path.is_dir() {
                    if !path.file_name().is_some_and(|name| name == "__pycache__") {
                        pending.push(path);
                    }
                } else if path
                    .extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| extensions.contains(&x))
                    && !(skip_document_manifest
                        && path
                            .file_name()
                            .is_some_and(|name| name == "document_manifest.sha256"))
                {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}
fn write_manifest(root: &Path, target: &Path, files: &[PathBuf]) -> Result<(), String> {
    let mut lines = Vec::new();
    for path in files {
        let relative = path
            .strip_prefix(root)
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .replace('\\', "/");
        lines.push(format!("{}  {relative}", hash_file(path)?));
    }
    fs::write(target, format!("{}\n", lines.join("\n"))).map_err(|e| e.to_string())
}

pub fn update_manifests(root: &Path) -> Result<(), String> {
    let mut source = files_under(
        root,
        &["src/ncm", "tests", "examples", "rust"],
        &["py", "rs", "json", "txt"],
        false,
    )?;
    source.extend(
        ["pyproject.toml", "Cargo.toml", "Cargo.lock"]
            .iter()
            .map(|path| root.join(path)),
    );
    source.sort();
    source.dedup();
    let mut documents = files_under(
        root,
        &["paper", "reports"],
        &["tex", "json", "jsonl", "md", "py", "sha256", "npz"],
        true,
    )?;
    let pdf = root.join("output/pdf");
    if pdf.exists() {
        for item in fs::read_dir(pdf).map_err(|e| e.to_string())? {
            let path = item.map_err(|e| e.to_string())?.path();
            if path.extension().is_some_and(|x| x == "pdf") {
                documents.push(path);
            }
        }
    }
    documents.extend(
        ["README.md", "Research.md", "MANIFEST.in"]
            .iter()
            .map(|path| root.join(path)),
    );
    for folder in [
        "milestone3-benchmark",
        "milestone4-scale-verified",
        "scale-coverage-followup",
        "scale-balanced-coverage",
        "coarse-codebook-stability",
    ] {
        let directory = root.join("reports").join(folder);
        if directory.exists() {
            for item in fs::read_dir(directory).map_err(|e| e.to_string())? {
                let path = item.map_err(|e| e.to_string())?.path();
                if path.extension().is_some_and(|x| x == "json") {
                    documents.push(path);
                }
            }
        }
    }
    documents.sort();
    documents.dedup();
    let paper = root.join("paper");
    write_manifest(root, &paper.join("source_manifest.sha256"), &source)?;
    write_manifest(root, &paper.join("document_manifest.sha256"), &documents)
}

pub fn audit_papers(root: &Path, build_dir: Option<&Path>) -> Result<(), String> {
    let base = root.join("paper");
    let registry: Value =
        serde_json::from_slice(&fs::read(base.join("references.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let stems = ["ncm_systems_paper", "SCALE_research_paper"];
    let mut all_citations = Vec::new();
    for stem in stems {
        let source =
            fs::read_to_string(base.join(format!("{stem}.tex"))).map_err(|e| e.to_string())?;
        for common in [
            "shared_preamble.tex",
            "shared_notation.tex",
            "shared_experiments.tex",
        ] {
            if !source.contains(&format!("\\input{{{common}}}")) {
                return Err(format!("{stem} is missing shared input {common}"));
            }
        }
        let text = expand(
            &base,
            &PathBuf::from(format!("{stem}.tex")),
            &mut HashSet::new(),
        )?;
        let labels = braced_commands(&text, &["label"]);
        let label_set = labels.iter().copied().collect::<HashSet<_>>();
        if labels.len() != label_set.len() {
            return Err(format!("{stem} has duplicate labels"));
        }
        let references = braced_commands(&text, &["eqref", "ref", "pageref"]);
        if references.iter().any(|label| !label_set.contains(label)) {
            return Err(format!("{stem} has undefined references"));
        }
        let keys = citations(&text);
        if keys.iter().any(|key| registry.get(key).is_none()) {
            return Err(format!("{stem} cites a missing reference"));
        }
        all_citations.push(keys);
        let normalized = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        for stale in [
            "eleven passing",
            "11 passed",
            "no dead-code reinitialization policy is implemented",
        ] {
            if normalized.contains(stale) {
                return Err(format!("{stem} contains stale claim: {stale}"));
            }
        }
        for phrase in [
            "reliable teacher-alignment gate was distinct",
            "so that gate still failed",
            "not an external pretrained",
        ] {
            if !normalized.contains(phrase) {
                return Err(format!("{stem} is missing shared result phrase: {phrase}"));
            }
        }
        if !normalized.contains("no tpu hardware was available")
            || !normalized.contains("synthetic")
        {
            return Err(format!("{stem} omits experiment scope"));
        }
        println!(
            "{stem}: shared inputs, {} labels, references and citations resolved.",
            labels.len()
        );
    }
    if !all_citations[0].iter().any(|key| key == "scalecompanion")
        || !all_citations[1].iter().any(|key| key == "systemscompanion")
    {
        return Err("companion paper citations are missing".into());
    }
    for line in fs::read_to_string(base.join("source_manifest.sha256"))
        .map_err(|e| e.to_string())?
        .lines()
    {
        let (expected, path) = line.split_once("  ").ok_or("invalid source manifest row")?;
        if hash_file(&root.join(path))? != expected {
            return Err(format!("source hash mismatch: {path}"));
        }
    }
    let validation: Value =
        serde_json::from_slice(&fs::read(base.join("validation.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let current: Value = serde_json::from_slice(
        &fs::read(base.join("validation_current.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if validation["passed"] != 44
        || validation["failed"] != 0
        || current["passed"] != 62
        || current["failed"] != 0
    {
        return Err("paper validation counts differ from committed evidence".into());
    }
    if let Some(folder) = build_dir {
        for stem in stems {
            let log = fs::read_to_string(folder.join(format!("{stem}.log")))
                .map_err(|e| e.to_string())?
                .to_lowercase();
            for phrase in [
                "undefined",
                "overfull",
                "missing character",
                "latex error",
                "emergency stop",
                "fatal error",
                "rerun to get",
                "label(s) may have changed",
            ] {
                if log.contains(phrase) {
                    return Err(format!("{stem} build log contains {phrase}"));
                }
            }
            if fs::metadata(folder.join(format!("{stem}.pdf")))
                .map_err(|e| e.to_string())?
                .len()
                <= 10_000
            {
                return Err(format!("{stem} PDF is unexpectedly small"));
            }
        }
    }
    Ok(())
}
pub fn generate_bibliographies(root: &Path, check: bool) -> Result<(), String> {
    let base = root.join("paper");
    let entries: Value =
        serde_json::from_slice(&fs::read(base.join("references.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    for (stem, suffix) in [
        ("ncm_systems_paper", "systems"),
        ("SCALE_research_paper", "scale"),
    ] {
        let text = expand(
            &base,
            &PathBuf::from(format!("{stem}.tex")),
            &mut HashSet::new(),
        )?;
        let keys = citations(&text);
        let mut bibliography = String::from(
            "% Generated from references.json; do not hand-edit.\n\\begingroup\n\\raggedright\n\\begin{thebibliography}{99}\n",
        );
        let mut selected = Vec::new();
        for key in &keys {
            let entry = entries
                .get(key)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("missing bibliography entry: {key}"))?;
            selected.push(entry);
        }
        bibliography.push_str(&selected.join("\n\n"));
        bibliography.push_str("\n\\end{thebibliography}\n\\endgroup\n");
        let target = base
            .join("generated")
            .join(format!("bibliography_{suffix}.tex"));
        if check {
            if fs::read_to_string(&target).map_err(|e| e.to_string())? != bibliography {
                return Err(format!(
                    "generated bibliography differs: {}",
                    target.display()
                ));
            }
        } else {
            fs::write(&target, bibliography).map_err(|e| e.to_string())?;
        }
        println!(
            "{stem}: {} citations resolved from the shared registry.",
            keys.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bibliography_outputs_match_committed_manuscripts() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        generate_bibliographies(root, true).unwrap();
    }
    #[test]
    fn native_paper_audit_accepts_archived_sources_and_evidence() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        audit_papers(root, None).unwrap();
    }
}
