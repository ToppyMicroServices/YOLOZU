# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Interface Contract Terminology (重要)

- ドキュメント/README/manual/manifest では、曖昧な `contract` 単独表現は避ける。
- 原則として **`interface contract`** を使う（例: `predictions interface contract`）。
- 日本語文中でもソフトウェア用語は無理に翻訳せず、`interface contract` を保持する。

## Source of Truth / 同期ルール

CLIやworkflowを変更したら、次を**同一PRで同期**すること:

1. 実装コード（`tools/`, `yolozu/`, `scripts/`）
2. `tools/manifest.json`
3. `yolozu/data/manifest/tools_manifest.json`（packaged copy）
4. 関連docs（`README.md`, `Readme_jp.md`, `docs/`, `manual/chapters/`）

乖離を残さないこと。

## CLI Rule: `--help` 必須

- 追加/変更するCLIは **`-h/--help` を必ずサポート**する。
- 既存CLI改修時も `--help` が壊れていないことを確認する。
- shell CLI（例: `scripts/*.sh`）にも `--help` usage を実装する。
- `--help` 追加に伴い、manifestの `inputs` / `examples` / `effects` / `outputs` を更新し、挙動と一致させる。

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim               # Claim work (assignee + in_progress)
bd update <id> --status in_progress  # Alternative (no auto-claim)
bd close <id>         # Complete work
bash refresh_beads_sync.sh            # Import the remote exported snapshot
bash export_beads_snapshot.sh <path>/issues.jsonl  # Preserve remote tombstones
```

## GitHub Issues Linking

Use `external-ref` to link a Beads issue to a GitHub Issue number.

```bash
bd update <id> --external-ref gh-123
```

## Multi-environment / Team Workflow (2台開発)

- まず `git pull --rebase` でコードを更新し、`bash refresh_beads_sync.sh` で
  `beads-sync` の issue snapshot をローカルDBへimportする
- 着手するissueは `bd update <id> --claim`（同時編集を避ける）
- 共有前にもう一度refreshし、import結果の `updated_issues` /
  `tie_kept_local_ids` を確認する
- 同一時刻の競合はremote snapshotと `bd show <id>` を比較し、採用する値を
  `bd update` で明示してから再exportする
- Beadsの共有は `bash export_beads_snapshot.sh` と `beads-sync`
  worktreeへのgit pushで行う（全clone共通）。raw `bd export` でremote
  snapshotを上書きしない。実行手順は `docs/beads_github_workflow.md` を参照する
- 通常の統合で `bd import --allow-stale` は使わない。これは古いsnapshotへ
  意図的に復元する場合だけの上書きオプション

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUBLISH BEADS STATE** - Refresh, inspect the import result, then export and
   push the `beads-sync` worktree by following `docs/beads_github_workflow.md`
5. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bash refresh_beads_sync.sh
   git push
   git status  # MUST show "up to date with origin"
   ```
6. **Clean up** - Clear stashes, prune remote branches
7. **Verify** - Code and exported Beads state are committed AND pushed
8. **Hand off** - Provide context for next session

## Required Quality Gates (when code/docs/manifest changed)

```bash
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
python3 -m unittest tests.test_packaged_tools_manifest tests.test_manifest_docs_references
```

CLIを変更した場合は追加で:

```bash
# 対象CLIの --help を直接確認（python/bashどちらでも）
python3 tools/<cli>.py --help
bash scripts/<cli>.sh --help
```

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
Use 'bd' for task tracking
