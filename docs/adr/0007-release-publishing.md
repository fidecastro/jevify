# ADR-0007 — Releases reach PyPI from CI, by trusted publishing

- **Status:** accepted (2026-11-04).
- **Extends:** ADR-0006 D4 (build and distribution); records who builds the
  files that go to PyPI and how they get there.

## Context

0.1.0 and 0.2.0 were built by hand with `uv build` from their tags, attached
to the GitHub release with a `SHA256SUMS`, and uploaded to PyPI with a token.
The upload step needed a PyPI token on the operator's machine, and the house
rule keeps secrets out of files, logs and chat. A rebuild of `v0.2.0` from a
fresh clone produced files byte-identical to the ones attached to the
release, so the build is reproducible and a second build can check the first.

## Decision

### D1 — Publishing a GitHub release publishes to PyPI

`.github/workflows/publish.yml` runs when a release is published. It checks
out the release tag, refuses a tag that is not `v` + the version in
`pyproject.toml`, builds with `uv build`, runs `twine check --strict`, and
installs the wheel as a tool and runs `jevify --version`.

### D2 — Trusted publishing, no stored token

The upload uses PyPI trusted publishing (OIDC) from the `publish` job in the
GitHub environment `pypi`. No PyPI token exists in the repository, its
secrets or the operator's machine. The PyPI project's trusted publisher names
this repository, the workflow file `publish.yml` and the environment `pypi`;
renaming either breaks publishing until PyPI is updated.

### D3 — The CI build must match what the release shows

If the release already carries a wheel and an sdist, the CI build must be
byte-identical to them or the workflow stops before uploading. If it carries
none, the CI build is uploaded to PyPI and then attached to the release with
a `SHA256SUMS`. Either way the files on PyPI and on the GitHub release are
the same bytes. Files are attached while the release is a draft, before it
is published, so the workflow never races a half-attached release.

### D4 — Dry run

`workflow_dispatch` with a `tag` runs the same build and checks against an
existing release without publishing.

## Consequences

- Cutting a release is: bump the version, merge, tag, then publish a GitHub
  release from the tag. Publishing it is the PyPI upload; there is no
  separate step.
- Each job holds only the permission it needs: `contents: read` to build,
  `id-token: write` to publish, `contents: write` to attach.
- A non-reproducible build (a new hatchling changing metadata, say) shows up
  as a mismatch in D3 when files were attached by hand. The remedy is to let
  CI attach them.

## Rejected alternatives

- **An API token in repository secrets.** It works, but it is a long-lived
  secret that trusted publishing makes unnecessary.
- **Uploading the hand-attached files as they are.** PyPI would carry files
  that no CI run built; D3 keeps hand-built files allowed but checked.
- **Publishing on tag push.** A tag is not a release note; the GitHub release
  is the reviewed announcement, so it is the trigger.
