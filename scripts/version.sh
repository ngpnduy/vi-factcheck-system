#!/usr/bin/env bash
# Manage the repository version and create annotated, reproducible Git releases.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="${PROJECT_DIR}/VERSION"
PYPROJECT_FILE="${PROJECT_DIR}/pyproject.toml"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/version.sh show
  bash scripts/version.sh bump <patch|minor|major>
  bash scripts/version.sh tag
  bash scripts/version.sh release <patch|minor|major>
EOF
}

read_version() {
  local version
  version="$(tr -d '[:space:]' < "${VERSION_FILE}")"
  if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "VERSION must use MAJOR.MINOR.PATCH, found: ${version}" >&2
    exit 1
  fi
  printf '%s\n' "${version}"
}

require_clean_git() {
  if ! git -C "${PROJECT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "A Git repository is required for tagging." >&2
    exit 1
  fi
  if [[ -n "$(git -C "${PROJECT_DIR}" status --porcelain)" ]]; then
    echo "Working tree is not clean. Commit or stash changes before releasing." >&2
    exit 1
  fi
}

sync_pyproject_version() {
  local new_version="$1"
  local temp_file
  temp_file="$(mktemp "${PYPROJECT_FILE}.XXXXXX")"
  awk -v version="${new_version}" '
    /^version = / { print "version = \"" version "\""; next }
    { print }
  ' "${PYPROJECT_FILE}" > "${temp_file}"
  mv "${temp_file}" "${PYPROJECT_FILE}"
}

bump() {
  local level="$1"
  local current major minor patch new_version
  current="$(read_version)"
  IFS='.' read -r major minor patch <<< "${current}"
  case "${level}" in
    patch) patch=$((patch + 1)) ;;
    minor) minor=$((minor + 1)); patch=0 ;;
    major) major=$((major + 1)); minor=0; patch=0 ;;
    *) usage; exit 1 ;;
  esac
  new_version="${major}.${minor}.${patch}"
  printf '%s\n' "${new_version}" > "${VERSION_FILE}"
  sync_pyproject_version "${new_version}"
  echo "Version bumped: ${current} -> ${new_version}"
  echo "Review and commit VERSION and pyproject.toml before tagging."
}

tag() {
  local version tag_name
  require_clean_git
  version="$(read_version)"
  tag_name="v${version}"
  if git -C "${PROJECT_DIR}" rev-parse -q --verify "refs/tags/${tag_name}" >/dev/null; then
    echo "Tag already exists: ${tag_name}" >&2
    exit 1
  fi
  git -C "${PROJECT_DIR}" tag -a "${tag_name}" -m "Release ${tag_name}"
  echo "Created annotated tag ${tag_name}. Push it with: git push origin ${tag_name}"
}

command="${1:-}"
case "${command}" in
  show)
    echo "Version: $(read_version)"
    if git -C "${PROJECT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "Commit: $(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
    fi
    ;;
  bump)
    [[ $# -eq 2 ]] || { usage; exit 1; }
    bump "$2"
    ;;
  tag)
    [[ $# -eq 1 ]] || { usage; exit 1; }
    tag
    ;;
  release)
    [[ $# -eq 2 ]] || { usage; exit 1; }
    require_clean_git
    bump "$2"
    echo "Release is prepared but not tagged: commit the version bump, then run 'bash scripts/version.sh tag'."
    ;;
  *)
    usage
    exit 1
    ;;
esac
