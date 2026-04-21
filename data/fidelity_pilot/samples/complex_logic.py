def process_repository(repo_path: str, config: dict) -> dict[str, Any]:
    """Process a repository and extract metrics.

    Args:
        repo_path: Path to the repository root.
        config: Processing configuration dictionary.

    Returns:
        Dictionary mapping file paths to their metrics.

    Raises:
        ValueError: If repo_path doesn't exist.
        RuntimeError: If processing fails after retries.
    """
    repo = Path(repo_path)
    if not repo.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    max_retries = config.get("max_retries", 3)
    include_patterns = config.get("include", ["**/*.py"])
    exclude_patterns = config.get("exclude", ["**/test_*", "**/__pycache__/**"])

    results: dict[str, Any] = {}
    errors: list[tuple[str, Exception]] = []

    # Collect all matching files
    all_files = [
        f for pattern in include_patterns
        for f in repo.glob(pattern)
        if not any(f.match(ep) for ep in exclude_patterns)
        and f.is_file()
        and f.stat().st_size < config.get("max_file_size", 1_000_000)
    ]

    logger.info(f"Processing {len(all_files)} files in {repo_path}")

    for file_path in sorted(all_files):
        relative = file_path.relative_to(repo)
        for attempt in range(1, max_retries + 1):
            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.split("\n")

                metrics = {
                    "path": str(relative),
                    "lines": len(lines),
                    "blank_lines": sum(1 for l in lines if not l.strip()),
                    "comment_lines": sum(1 for l in lines if l.strip().startswith("#")),
                    "max_indent": max(
                        (len(l) - len(l.lstrip())) for l in lines if l.strip()
                    ),
                    "has_type_hints": any(
                        ":" in l and "->" in l or ": " in l
                        for l in lines
                        if l.strip().startswith("def ")
                    ),
                    "classes": [
                        l.strip().split("(")[0].replace("class ", "")
                        for l in lines if l.strip().startswith("class ")
                    ],
                    "functions": [
                        l.strip().split("(")[0].replace("def ", "")
                        for l in lines if l.strip().startswith("def ")
                    ],
                    "imports": len([l for l in lines if l.startswith(("import ", "from "))]),
                    "complexity_score": _compute_complexity(content),
                }
                results[str(relative)] = metrics
                break
            except UnicodeDecodeError:
                logger.warning(f"Skipping binary file: {relative}")
                break
            except Exception as e:
                if attempt == max_retries:
                    errors.append((str(relative), e))
                    logger.error(f"Failed after {max_retries} attempts: {relative}: {e}")
                else:
                    logger.warning(f"Retry {attempt}/{max_retries} for {relative}: {e}")

    if errors:
        logger.warning(f"{len(errors)} files failed processing")

    return {
        "files": results,
        "total_files": len(all_files),
        "processed": len(results),
        "failed": len(errors),
        "errors": [(p, str(e)) for p, e in errors],
    }


def _compute_complexity(source: str) -> float:
    """Estimate cyclomatic complexity from source code."""
    keywords = ["if ", "elif ", "else:", "for ", "while ", "except ", "with ",
                 "and ", "or ", " if ", "assert "]
    return sum(source.count(kw) for kw in keywords) / max(source.count("\n"), 1)
