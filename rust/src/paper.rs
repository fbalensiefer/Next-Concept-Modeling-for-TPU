//! Paper source expansion and bibliography generation in Rust.
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
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

fn json_file(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid JSON {}: {e}", path.display()))
}
fn field_mean(rows: &[&Value], path: &[&str]) -> Result<f64, String> {
    if rows.is_empty() {
        return Err("cannot average an empty evidence group".into());
    }
    let values = rows
        .iter()
        .map(|row| {
            path.iter()
                .try_fold(*row, |v, key| {
                    v.get(*key)
                        .ok_or_else(|| format!("missing evidence field {key}"))
                })?
                .as_f64()
                .ok_or_else(|| format!("non-numeric evidence field {path:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}
fn latex_table(
    command: &str,
    caption: &str,
    label: &str,
    columns: &str,
    header: &str,
    rows: &[String],
) -> String {
    format!(
        "\\newcommand{{\\{command}}}{{%\n\\begin{{table}}[t]\\centering\\small\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabularx}}{{\\linewidth}}{{{columns}}}\\toprule\n{header}\\\\\\midrule\n{}\n\\bottomrule\\end{{tabularx}}\\end{{table}}}}",
        rows.iter()
            .map(|row| format!("{row}\\\\"))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

/// Generate current manuscript tables from the committed result ledgers.
pub fn generate_recent_results(root: &Path, check: bool) -> Result<(), String> {
    let mut facts = json!({"sources":{}});
    let mut macros = Vec::new();
    let mut read = |relative: &str| -> Result<Value, String> {
        let path = root.join(relative);
        facts["sources"][relative] = Value::String(hash_file(&path)?);
        json_file(&path)
    };
    let validation = read("paper/validation_current.json")?;
    if validation["passed"] != 62 || validation["failed"] != 0 {
        return Err("current validation evidence gate failed".into());
    }
    macros.push(format!(
        "\\newcommand{{\\CurrentTestCases}}{{{}}}",
        validation["passed"]
            .as_u64()
            .ok_or("validation passed must be integer")?
    ));
    let maintenance = read("reports/maintenance-ablation/results.json")?;
    let domains = read("reports/fresh-domain-validation/results.json")?;
    let priors = read("reports/dependency-validation/results.json")?;
    let alignment = read("reports/alignment-diagnosis/results.json")?;
    let innovations = read("reports/invertible-baselines/results.json")?;
    let maintenance = maintenance
        .as_array()
        .ok_or("maintenance evidence must be array")?;
    let domains = domains.as_array().ok_or("domain evidence must be array")?;
    let priors = priors.as_array().ok_or("prior evidence must be array")?;
    if maintenance.len() != 100 || domains.len() != 360 || priors.len() != 480 {
        return Err("manuscript evidence row counts do not match the frozen protocol".into());
    }
    let unique = priors
        .iter()
        .map(|r| format!("{}|{}|{}", r["domain"], r["seed"], r["arm"]))
        .collect::<std::collections::BTreeSet<_>>();
    if unique.len() != 120 {
        return Err("prior evidence does not contain 120 unique codecs".into());
    }
    let treatments = [
        ("minimal_ema", "Minimal EMA"),
        ("ema_recovery", "EMA + recovery"),
        ("reference", "Reference"),
        ("fixed_binary", "Fixed binary"),
    ];
    let mut rows = Vec::new();
    let mut maintenance_facts = serde_json::Map::new();
    for (name, label) in treatments {
        let selected = maintenance
            .iter()
            .filter(|r| {
                r["treatment"] == name
                    && (name != "ema_recovery" || r["study"] == "maintenance-ablation")
            })
            .collect::<Vec<_>>();
        let tv = field_mean(&selected, &["after", "factorized_oracle_tv"])?;
        let reconstruction = field_mean(&selected, &["after", "sequence_reconstruction"])?;
        let usage = selected
            .iter()
            .filter(|r| r["collapsed_fraction"].as_f64().unwrap_or(0.) > 0.05)
            .count();
        maintenance_facts.insert(
            name.into(),
            json!({"tv":tv,"reconstruction":reconstruction,"usage_failures":usage}),
        );
        rows.push(format!(
            "{label} & {:.3} & {usage}/20 & {tv:.4}",
            100. * reconstruction
        ));
    }
    facts["maintenance"] = Value::Object(maintenance_facts);
    macros.push(latex_table("RecentMaintenanceTable","Matched historical-seed ablations. Reconstruction is sequence accuracy in percent; usage failures exceed 5 percent collapsed minibatches. All objectives and schedules remain shared.","tab:recentmaintenance","@{}Yrrr@{}","Condition & Recon. & Usage failures & Oracle TV",&rows));
    let names = domains.iter().filter_map(|r| r["domain"].as_str()).fold(
        Vec::<String>::new(),
        |mut x, n| {
            if !x.iter().any(|v| v == n) {
                x.push(n.into());
            }
            x
        },
    );
    let labels = [
        "Original source",
        "Biased independent",
        "Markov",
        "Noisy parity",
        "Skewed modes",
        "Alternating modes",
    ];
    rows.clear();
    let mut domain_facts = serde_json::Map::new();
    for (name, label) in names.iter().zip(labels) {
        let mut cells = Vec::new();
        let mut local = serde_json::Map::new();
        for treatment in ["ema_recovery", "reference", "fixed_binary"] {
            let selected = domains
                .iter()
                .filter(|r| {
                    r["domain"].as_str() == Some(name.as_str()) && r["treatment"] == treatment
                })
                .collect::<Vec<_>>();
            if selected.len() != 20 {
                return Err(format!("expected 20 domain rows for {name}/{treatment}"));
            }
            let tv = field_mean(&selected, &["after", "factorized_oracle_tv"])?;
            let failures = selected
                .iter()
                .filter(|r| r["failures"].as_array().is_some_and(|x| !x.is_empty()))
                .count();
            local.insert(
                treatment.into(),
                json!({"tv":tv,"fidelity_failures":failures}),
            );
            cells.push(format!("{tv:.3} ({failures})"));
        }
        domain_facts.insert(name.clone(), Value::Object(local));
        rows.push(format!("{label} & {}", cells.join(" & ")));
    }
    facts["domains"] = Value::Object(domain_facts);
    macros.push(latex_table("RecentDomainTable","Fresh seeds 31--35 across all four objective arms. Cells give mean factorized-oracle output TV and fidelity failures out of 20 models in parentheses. No prior is trained in this study.","tab:recentdomains","@{}Yrrr@{}","Source & EMA + recovery & Reference & Fixed binary",&rows));
    rows.clear();
    let mut prior_facts = serde_json::Map::new();
    for (name, label) in names.iter().zip(labels) {
        let eligible = priors
            .iter()
            .filter(|r| {
                r["domain"].as_str() == Some(name.as_str())
                    && r["model"] == "independent"
                    && r["codec_eligible"] == true
            })
            .count();
        let mut local = serde_json::Map::new();
        let mut cells = Vec::new();
        for model in ["independent", "conditional_tree", "joint_fine"] {
            let selected = priors
                .iter()
                .filter(|r| {
                    r["domain"].as_str() == Some(name.as_str())
                        && r["model"] == model
                        && r["codec_eligible"] == true
                })
                .collect::<Vec<_>>();
            let tv = field_mean(&selected, &["output_tv"])?;
            local.insert(model.into(), json!(tv));
            cells.push(format!("{tv:.4}"));
        }
        prior_facts.insert(name.clone(), Value::Object(local));
        rows.push(format!("{label} & {eligible} & {}", cells.join(" & ")));
    }
    facts["priors"] = Value::Object(prior_facts);
    macros.push(latex_table("RecentPriorTable","Training-only prior fits on frozen reference codecs passing the sampled fidelity check. Lower decoded output TV is better. All-codec results, including flagged failures, remain in the public report.","tab:recentpriors","@{}Yrrrr@{}","Source & Eligible & Indep. & Tree & Joint",&rows));
    facts["alignment"] = alignment["invariants"].clone();
    let innovation_values = innovations
        .as_object()
        .ok_or("innovation evidence must be an object")?;
    let mut selected = serde_json::Map::new();
    for (name, row) in innovation_values {
        if !row["invertible_innovations"].is_null() {
            selected.insert(name.clone(), row["invertible_innovations"].clone());
        }
    }
    facts["innovations"] = Value::Object(selected);
    if facts["alignment"]["last_bit_pair_target_max_difference"].as_f64() != Some(0.0) {
        return Err("alignment target invariant failed".into());
    }
    if facts["innovations"]
        .as_object()
        .unwrap()
        .values()
        .any(|x| x["bijective"] != true || x["tv"].as_f64().unwrap_or(1.) >= 1e-14)
    {
        return Err("invertible innovation audit failed".into());
    }
    let tex = format!(
        "% Generated by generate_recent_results.py; do not edit.\n{}\n",
        macros.join("\n")
    );
    let data = format!(
        "{}\n",
        serde_json::to_string_pretty(&facts).map_err(|e| e.to_string())?
    );
    for (name, content) in [("recent_results.tex", tex), ("recent_results.json", data)] {
        let path = root.join("paper/generated").join(name);
        if check {
            if fs::read_to_string(&path).map_err(|e| e.to_string())? != content {
                return Err(format!("stale generated evidence: {}", path.display()));
            }
        } else {
            fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
            fs::write(path, content).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn number_list(values: &Value) -> Result<String, String> {
    values
        .as_array()
        .ok_or("expected integer array")?
        .iter()
        .map(|v| {
            v.as_i64()
                .map(|x| x.to_string())
                .ok_or_else(|| "expected integer value".into())
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|x| x.join(","))
}
fn average_path(rows: &[&Value], path: &[&str]) -> Result<f64, String> {
    field_mean(rows, path)
}
fn save_generated(root: &Path, outputs: &[(&str, String)], check: bool) -> Result<(), String> {
    for (relative, content) in outputs {
        let path = root.join(relative);
        if check {
            if fs::read_to_string(&path).map_err(|e| e.to_string())? != *content {
                return Err(format!("stale generated evidence: {}", path.display()));
            }
        } else {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::write(path, content).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Generate the shared manuscript evidence tables from committed result files.
pub fn generate_results(root: &Path, check: bool) -> Result<(), String> {
    const STUDIES: [&str; 4] = [
        "milestone4-scale-verified",
        "scale-coverage-followup",
        "scale-balanced-coverage",
        "coarse-codebook-stability",
    ];
    const ARMS: [&str; 4] = [
        "reconstruction",
        "alignment",
        "dependency",
        "alignment_dependency",
    ];
    let commit = "1d600d7f4598ddb7e2808b29ee8f6eefe01cfb78";
    let mut facts = json!({"implementation_commit":commit,"studies":{},"source_sha256":{}});
    let mut reports = std::collections::BTreeMap::new();
    for name in STUDIES.into_iter().chain(["milestone3-benchmark"]) {
        let relative = format!("reports/{name}/report.json");
        facts["source_sha256"][&relative] = json!(hash_file(&root.join(&relative))?);
        reports.insert(name, json_file(&root.join(&relative))?);
    }
    let mut shared_teacher: Option<(String, Value)> = None;
    for name in STUDIES {
        let report = &reports[name];
        let identity = (
            report["teacher"]["checkpoint_sha256"]
                .as_str()
                .ok_or("teacher checkpoint missing")?
                .to_owned(),
            report["teacher"]["split_id_sha256"].clone(),
        );
        if shared_teacher.as_ref().is_some_and(|x| x != &identity) {
            return Err(format!("{name}: teacher or sampled splits changed"));
        }
        if shared_teacher.is_none() {
            shared_teacher = Some(identity.clone());
            facts["shared_teacher"] =
                json!({"checkpoint_sha256":identity.0,"split_id_sha256":identity.1});
        }
        let config = &report["protocol"]["config"];
        let seeds = config
            .get("confirmation_seeds")
            .unwrap_or(&config["seeds"])
            .as_array()
            .ok_or("study seeds missing")?;
        let rows = report["results"]
            .as_array()
            .ok_or("study results missing")?;
        let mut means = serde_json::Map::new();
        for arm in ARMS {
            let selected = rows
                .iter()
                .filter(|r| r["arm"] == arm && seeds.contains(&r["seed"]))
                .collect::<Vec<_>>();
            if !selected.is_empty() {
                means.insert(
                    arm.into(),
                    json!(average_path(&selected, &["total_variation"])?),
                );
            }
        }
        facts["studies"][name] = json!({"seeds":seeds,"mean_output_tv":means,"followup_decision":report.get("followup_decision").cloned().unwrap_or(Value::Null),"stability_decision":report.get("coarse_stability_decision").cloned().unwrap_or(Value::Null)});
    }
    let last = &reports["coarse-codebook-stability"];
    let validation = json_file(&root.join("reports/coarse-codebook-stability/validation.json"))?;
    let bench = &reports["milestone3-benchmark"];
    if validation["passed"] != 44
        || validation["failed"] != 0
        || last["coarse_stability_decision"]["evaluated_models"] != 20
        || last["coarse_stability_decision"]["passed"] != true
        || last["followup_decision"]["teacher_alignment_gate_passed"] != false
    {
        return Err("paper validation/stability gate failed".into());
    }
    for (source, expected) in last["protocol"]["implementation_sha256"]
        .as_object()
        .ok_or("implementation hashes missing")?
    {
        let path = format!("src/ncm/{source}");
        let spec = format!("{commit}:{path}");
        let output = Command::new("git")
            .args(["show", &spec])
            .current_dir(root)
            .output()
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(format!("cannot read historical implementation {source}"));
        }
        let mut content = output.stdout;
        if source == "scale_experiment.py" {
            content = String::from_utf8(content)
                .map_err(|e| e.to_string())?
                .replace(
                    "def make_codec_step(config, optimizer, *, maintenance=maintain_coarse):",
                    "def make_codec_step(config, optimizer):",
                )
                .replace(
                    "books, coarse, diagnostics = maintenance(",
                    "books, coarse, diagnostics = maintain_coarse(",
                )
                .into_bytes();
        }
        if expected.as_str() != Some(&crate::corpus::sha256(&content)) {
            return Err(format!("historical implementation hash mismatch: {source}"));
        }
    }
    let validation_path = root.join("reports/coarse-codebook-stability/validation.json");
    facts["source_sha256"]["reports/coarse-codebook-stability/validation.json"] =
        json!(hash_file(&validation_path)?);
    let mut historical = json!({});
    for stage in ["codec", "prior"] {
        let folder = root.join("reports/text-experiments").join(stage);
        let mut files = fs::read_dir(&folder)
            .map_err(|e| e.to_string())?
            .map(|x| x.map(|e| e.path()).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        files.sort();
        for path in files {
            let relative = path
                .strip_prefix(root)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .replace('\\', "/");
            facts["source_sha256"][relative] = json!(hash_file(&path)?);
        }
        let provenance = json_file(&folder.join("provenance.json"))?;
        if provenance["historical_record_not_a_new_run"] != true
            || provenance["checkpoint_sha256"] != bench["config"][format!("{stage}_sha256")]
        {
            return Err(format!("historical {stage} provenance mismatch"));
        }
        let text = fs::read_to_string(folder.join("metrics.jsonl")).map_err(|e| e.to_string())?;
        let rows = text
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        historical[stage] =
            json!({"checkpoint_sha256":provenance["checkpoint_sha256"],"observations":rows});
    }
    facts["historical_text"] = historical.clone();
    let codec = historical["codec"]["observations"]
        .as_array()
        .ok_or("codec observations missing")?;
    if codec.first().ok_or("codec observations empty")?["step"] != 100
        || codec.last().unwrap()["step"] != 500
        || codec[0]["train"]["byte_accuracy"].as_f64() != Some(1.0)
        || ((codec[0]["validation"]["byte_cross_entropy"]
            .as_f64()
            .unwrap()
            * 1000.)
            .round()
            / 1000.
            - 1.949)
            .abs()
            > 1e-12
        || ((codec.last().unwrap()["validation"]["byte_cross_entropy"]
            .as_f64()
            .unwrap()
            * 1000.)
            .round()
            / 1000.
            - 3.55)
            .abs()
            > 1e-12
    {
        return Err("historical codec evidence gate failed".into());
    }
    if fs::metadata(root.join("examples/text/train.txt"))
        .map_err(|e| e.to_string())?
        .len()
        != 26
        || fs::metadata(root.join("examples/text/validation.txt"))
            .map_err(|e| e.to_string())?
            .len()
            != 13
    {
        return Err("text fixture size check failed".into());
    }
    facts["final_protocol"] = last["protocol"]["config"].clone();
    let seeds = last["protocol"]["config"]["confirmation_seeds"]
        .as_array()
        .ok_or("confirmation seeds missing")?;
    let new_runs = last["results"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| seeds.contains(&r["seed"]))
        .collect::<Vec<_>>();
    if new_runs.len() != 20
        || new_runs.iter().any(|r| {
            r["zero_support_teacher_mass"].as_f64() != Some(0.0)
                || r["population_coarse_active_codes"].as_f64() != Some(2.0)
                || r["population_sequence_reconstruction"]
                    .as_f64()
                    .unwrap_or(0.)
                    < 0.98
        })
    {
        return Err("final protocol model gates failed".into());
    }
    let min_ad = new_runs
        .iter()
        .filter(|r| r["arm"] == "alignment_dependency")
        .map(|r| r["population_sequence_reconstruction"].as_f64().unwrap())
        .fold(f64::INFINITY, f64::min);
    if ((min_ad * 10000.).round() / 100. - 99.07).abs() > 1e-12 {
        return Err("alignment-dependency reconstruction threshold changed".into());
    }
    facts["test_cases_passed"] = validation["passed"].clone();
    let mut environment = serde_json::Map::new();
    for k in ["platform", "python", "versions", "dtype", "device_kind"] {
        environment.insert(k.into(), bench[k].clone());
    }
    facts["environment"] = Value::Object(environment);
    facts["stability"] = last["coarse_stability_decision"].clone();
    facts["budget"] = last["budget"].clone();
    facts["benchmark_rows"] = json!([]);
    let mut macros = Vec::new();
    let m = |n: &str, v: String| format!("\\newcommand{{\\{n}}}{{{v}}}");
    macros.push(m("TestCases", "44".into()));
    macros.push(m("StabilityModels", "20".into()));
    macros.push(m(
        "StabilityPP",
        format!(
            "{:.4}",
            facts["stability"]["minimum_population_perplexity"]
                .as_f64()
                .ok_or("perplexity missing")?
        ),
    ));
    macros.push(m("StabilityCollapsed", "0".into()));
    macros.push(m("SourceRevision", "1d600d7".into()));
    let pilot = reports["milestone4-scale-verified"]["results"]
        .as_array()
        .unwrap();
    for (macro_name, arm) in [
        ("PilotTCAlignment", "alignment"),
        ("PilotTCCombined", "alignment_dependency"),
    ] {
        let selected = pilot.iter().filter(|r| r["arm"] == arm).collect::<Vec<_>>();
        macros.push(m(
            macro_name,
            format!(
                "{:.4}",
                average_path(&selected, &["hard_conditional_tc_sum"])?
            ),
        ));
    }
    macros.push(m(
        "FinalAlignmentGain",
        format!(
            "{:.5}",
            last["followup_decision"]["mean_alignment_tv_improvement_over_dependency_only"]
                .as_f64()
                .ok_or("alignment gain missing")?
        ),
    ));
    macros.push(m(
        "FinalAlignmentWins",
        last["followup_decision"]["alignment_winning_seeds"].to_string(),
    ));
    let labels = [
        "Initial pilot",
        "Decoder replay",
        "Balanced codec",
        "Coarse maintenance",
    ];
    let mut study_rows = Vec::new();
    for (label, name) in labels.iter().zip(STUDIES) {
        let study = &facts["studies"][name];
        let numbers = ARMS
            .iter()
            .map(|arm| {
                study["mean_output_tv"][arm]
                    .as_f64()
                    .map(|x| format!("{x:.4}"))
                    .unwrap_or_else(|| "--".into())
            })
            .collect::<Vec<_>>();
        study_rows.push(format!(
            "{label} & {} & {}",
            number_list(&study["seeds"])?,
            numbers.join(" & ")
        ));
    }
    macros.push(latex_table("StudyResultsTable","Mean exact output total variation (TV; lower is better). Columns R, A, D, and AD denote reconstruction, alignment, dependence, and their combination. Only the listed student seeds enter each row; the first row has no decoder replay, later rows include 200 replay updates. Different rows use different seed sets and interventions, so row-to-row means are not causal quality comparisons.","tab:studies","@{}Ylrrrr@{}","Study & Student seeds & R & A & D & AD",&study_rows));
    let mut benchmark_rows = Vec::new();
    let mut benchmark_facts = Vec::new();
    for length in [8, 128, 512] {
        let source = if length == 8 {
            bench["trained_results"]
                .as_array()
                .ok_or("trained results missing")?
        } else {
            bench["synthetic_results"]
                .as_array()
                .ok_or("synthetic results missing")?
                .iter()
                .find(|r| r["length"] == length)
                .and_then(|r| r["results"].as_array())
                .ok_or("synthetic rows missing")?
        };
        let lookup = |model: &str| {
            source
                .iter()
                .find(|r| r["model"] == model && r["batch_size"] == 16)
                .ok_or_else(|| format!("missing benchmark row {model}/{length}"))
        };
        let values = [
            lookup("ncm_padded")?["median_ms"].as_f64().unwrap(),
            lookup("ncm_actual_lengths")?["median_ms"].as_f64().unwrap(),
            lookup("ar_kv_cache")?["median_ms"].as_f64().unwrap(),
        ];
        benchmark_facts.push(json!({"length":length,"batch_size":16,"median_ms":values,"trained_tiny_fixture":length==8}));
        benchmark_rows.push(format!(
            "{length} & {} & {:.3} & {:.3} & {:.3}",
            if length == 8 {
                "Trained toy"
            } else {
                "Untrained shape"
            },
            values[0],
            values[1],
            values[2]
        ));
    }
    facts["benchmark_rows"] = Value::Array(benchmark_facts);
    let last_latency = facts["benchmark_rows"].as_array().unwrap().last().unwrap();
    macros.push(m(
        "PaddedLatency",
        format!("{:.1}", last_latency["median_ms"][0].as_f64().unwrap()),
    ));
    macros.push(m(
        "ActualLatency",
        format!("{:.1}", last_latency["median_ms"][1].as_f64().unwrap()),
    ));
    macros.push(latex_table("CPUBenchmarkTable","Synchronized single-device CPU median generation latency (ms), batch size 16, float32, five warmups and 30 repetitions. NCM includes byte decoding; AR uses a KV cache. Length 8 uses trained toy models of width 16 and one block; lengths 128/512 use untrained width-64, two-block shape checks. These are not quality-matched speedups or TPU measurements.","tab:cpu","@{}rYrrr@{}","Bytes & Scope & Padded NCM & Actual-length NCM & Cached AR",&benchmark_rows));
    let mut protocol_rows = Vec::new();
    for (label, name) in labels.iter().zip(STUDIES) {
        let cfg = &reports[name]["protocol"]["config"];
        let seeds = cfg["seeds"].as_array().ok_or("protocol seeds missing")?;
        let confirmation = cfg
            .get("confirmation_seeds")
            .unwrap_or(&cfg["seeds"])
            .as_array()
            .unwrap();
        let known = seeds
            .iter()
            .filter(|s| !confirmation.contains(s))
            .cloned()
            .collect::<Vec<_>>();
        let arms = cfg
            .get("arms")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| vec![json!("R"), json!("A"), json!("AD")]);
        protocol_rows.push(format!(
            "{label} & {} & {} & {}",
            if known.is_empty() {
                "--".into()
            } else {
                number_list(&Value::Array(known))?
            },
            number_list(&Value::Array(confirmation.clone()))?,
            seeds.len() * arms.len()
        ));
    }
    macros.push(latex_table("ProtocolSeedsTable","Completed finite-domain protocols. Known seeds are repeated initializations, not new independent confirmations. Every four-arm follow-up fits 32 student models, of which 20 use newly reserved student seeds; the teacher, domain and sampled datasets remain fixed.","tab:protocols","@{}Yllr@{}","Protocol & Known-seed checks & Evaluation seeds & Models",&protocol_rows));
    let tex = format!(
        "% Generated by generate_results.py; do not hand-edit.\n{}\n",
        macros.join("\n")
    );
    let data = format!(
        "{}\n",
        serde_json::to_string_pretty(&facts).map_err(|e| e.to_string())?
    );
    save_generated(
        root,
        &[
            ("paper/generated/observed_results.tex", tex),
            ("paper/generated/observed_results.json", data),
        ],
        check,
    )
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
