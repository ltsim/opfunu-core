# Shared checks for the Husky pre-commit and pre-push hooks.
#
# Runs the same trio as CI (type.yml + test.yml): ruff, mypy and pytest.
# Sourced by .husky/pre-commit and .husky/pre-push; keep the two hooks in sync
# by changing this file only.
set -e

echo "==> ruff check --fix"

# Ruff may rewrite files staged for this commit. Snapshot their worktree hashes
# before and after the fix so exactly the rewritten ones are re-staged and the
# fix lands in the same commit. Files that are not part of the commit are left
# untouched (their unstaged ruff fixes stay unstaged), and paths with spaces or
# non-ASCII names survive the -z round-trip that git status quoting would break.
staged=$(git diff --cached --name-only --diff-filter=ACM -z -- '*.py' | tr '\0' '\n')
before=$(mktemp)
after=$(mktemp)
trap 'rm -f "$before" "$after"' EXIT

if [ -n "$staged" ]; then
    printf '%s\n' "$staged" | while IFS= read -r file; do
        [ -f "$file" ] && printf '%s %s\n' "$(git hash-object -- "$file")" "$file"
    done | LC_ALL=C sort >"$before"
fi

uv run ruff check --fix . || true

if [ -n "$staged" ]; then
    printf '%s\n' "$staged" | while IFS= read -r file; do
        [ -f "$file" ] && printf '%s %s\n' "$(git hash-object -- "$file")" "$file"
    done | LC_ALL=C sort >"$after"
fi

fixed=$(LC_ALL=C comm -13 "$before" "$after" | cut -d' ' -f2-)
if [ -n "$fixed" ]; then
    printf '==> ruff fixed, staging:\n%s\n' "$fixed"
    printf '%s\n' "$fixed" | while IFS= read -r file; do
        [ -n "$file" ] && git add -- "$file"
    done
fi

echo "==> ruff check"
uv run ruff check .

echo "==> mypy"
uv run mypy

echo "==> pytest"
uv run pytest

echo "==> all checks passed"
