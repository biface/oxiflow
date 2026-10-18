#!/bin/sh
# =============================================================================
# check-version.sh — package version against branch, pull request or tag
# =============================================================================
# Shared by CI (ci.yml, publish.yml) and the pre-push hook, so that both apply
# exactly the same rules (#168). cargo-semver-checks trusts the version in
# Cargo.toml to decide which changes are allowed: a wrong version would make
# it silently permissive, hence this check.
#
#   update/X.Y.Z        version X.Y.Z-rc.N
#   staging/X.Y.Z       version X.Y.Z
#   main                a release version, no pre-release suffix
#   other branches      not checked
#   PR → staging/X.Y.Z  version X.Y.Z-rc.N
#   PR → main           a release version; X.Y.Z when the head is staging/X.Y.Z
#   tag vX.Y.Z          version X.Y.Z
#
# Usage:
#   check-version.sh [-m MANIFEST] branch <branch>
#   check-version.sh [-m MANIFEST] pr <base> <head>
#   check-version.sh [-m MANIFEST] tag <tag>
#
# MANIFEST defaults to ./Cargo.toml; "-" reads it from standard input (the
# pre-push hook feeds it `git show <commit>:Cargo.toml`). POSIX sh only.
# =============================================================================

set -eu

usage() {
    echo "usage: $0 [-m MANIFEST] branch <branch> | pr <base> <head> | tag <tag>" >&2
    exit 2
}

fail() {
    echo "check-version: $*" >&2
    exit 1
}

manifest=Cargo.toml
if [ "${1:-}" = "-m" ]; then
    [ $# -ge 2 ] || usage
    manifest=$2
    shift 2
fi
[ $# -ge 1 ] || usage
context=$1
shift

# `version` key of the [package] table, ignoring every other table.
read_version() {
    awk '
        /^[[:space:]]*\[/ { in_package = ($0 ~ /^[[:space:]]*\[package\][[:space:]]*$/) ; next }
        in_package && /^[[:space:]]*version[[:space:]]*=/ {
            sub(/^[^"]*"/, ""); sub(/".*$/, ""); print; exit
        }
    ' "$1"
}

if [ "$manifest" = "-" ]; then
    version=$(read_version /dev/stdin)
else
    [ -f "$manifest" ] || fail "manifest not found: $manifest"
    version=$(read_version "$manifest")
fi
[ -n "$version" ] || fail "no version found in the [package] table of $manifest"

# X.Y.Z with no leading zeros, as SemVer requires.
num='(0|[1-9][0-9]*)'
release_re="^$num\\.$num\\.$num\$"
rc_suffix_re="-rc\\.$num\$"

is_release() { printf '%s\n' "$1" | grep -Eq "$release_re"; }

# Version part of "<prefix>/X.Y.Z", or failure if the name does not end in X.Y.Z.
# Always called as v=$(version_of ...): under set -e a failing substitution in
# an assignment stops the script, whereas one used as an argument would not.
version_of() {
    v=${1#*/}
    is_release "$v" || fail "'$1' does not end in a X.Y.Z version"
    printf '%s\n' "$v"
}

expect_rc() {
    printf '%s\n' "$version" | grep -Eq "^$(printf '%s' "$1" | sed 's/\./\\./g')$rc_suffix_re" \
        || fail "$2 requires version $1-rc.N, Cargo.toml has $version"
    echo "check-version: $version matches $2"
}

expect_exact() {
    [ "$version" = "$1" ] || fail "$2 requires version $1, Cargo.toml has $version"
    echo "check-version: $version matches $2"
}

expect_release() {
    is_release "$version" || fail "$1 requires a release version (X.Y.Z), Cargo.toml has $version"
    echo "check-version: $version matches $1"
}

case "$context" in
    branch)
        [ $# -eq 1 ] || usage
        case "$1" in
            update/*)  v=$(version_of "$1"); expect_rc "$v" "branch $1" ;;
            staging/*) v=$(version_of "$1"); expect_exact "$v" "branch $1" ;;
            main)      expect_release "branch main" ;;
            *)         echo "check-version: branch $1 is not checked (version $version)" ;;
        esac
        ;;
    pr)
        [ $# -eq 2 ] || usage
        case "$1" in
            staging/*) v=$(version_of "$1"); expect_rc "$v" "a pull request to $1" ;;
            main)
                case "$2" in
                    staging/*) v=$(version_of "$2"); expect_exact "$v" "a pull request from $2 to main" ;;
                    *)         expect_release "a pull request to main" ;;
                esac
                ;;
            *) echo "check-version: pull requests to $1 are not checked (version $version)" ;;
        esac
        ;;
    tag)
        [ $# -eq 1 ] || usage
        case "$1" in
            v*) is_release "${1#v}" || fail "tag $1 is not of the form vX.Y.Z"
                expect_exact "${1#v}" "tag $1" ;;
            *)  fail "tag $1 is not of the form vX.Y.Z" ;;
        esac
        ;;
    *) usage ;;
esac
