list:
    just --list

format:
    cargo fmt

build:
    cargo build --features scalar64
    cargo build

test:
    cargo test --features scalar64
    cargo test
  
clippy:
    cargo clippy --features scalar64
    cargo clippy

checks:
    just format
    just build
    just clippy
    just test

clean:
  find . -name target -type d -exec rm -r {} +
  just remove-lockfiles

remove-lockfiles:
    find . -name Cargo.lock -type f -exec rm {} +

list-outdated:
    cargo outdated -R -w

update:
    cargo update --aggressive

publish:
    cargo publish --no-verify
