# Beads + GitHub Issues 運用

このrepoでは Beads をissue trackerとして使い、必要なrecordだけを
`external_ref` でGitHub Issuesへ紐付けます。

## 確認済みのcommand surface

このrepoでサポートする Beads 1.1.0 CLI は `bd export` と `bd import` を
提供します。そのため、共有にはexportしたJSONL snapshotと
`beads-sync` branch上の通常のgit操作を使います。

ローカルのBeads databaseが作業中のlive storeです。git管理する
`.beads/issues.jsonl` はissue単位の交換用snapshotであり、database全体の
backupではありません。Dolt history、working-set state、issue以外のtableは
含まれません。`.beads/interactions.jsonl` は別のappend-only audit logで、
`bd export` の対象外です。

## 各環境で作業を始める

code branchを更新してから、remoteのissue snapshotをimportします。

```bash
git pull --rebase
bash refresh_beads_sync.sh
bd update <id> --claim
```

helperは
`refs/heads/beads-sync:refs/remotes/origin/beads-sync` を明示的にfetchするため、
`--single-branch` cloneでも動作します。一時ファイルからimportし、現在の
branchを切り替えたり、そのbranchの `.beads/issues.jsonl` を直接上書き
したりしません。

remote名、branch名、`bd` のpathが異なる場合だけ、`REMOTE`、
`SYNC_BRANCH`、`BD_BIN` を指定します。

```bash
REMOTE=upstream SYNC_BRANCH=beads-sync bash refresh_beads_sync.sh
```

## Beads stateを共有する

export前にもう一度refreshします。JSON形式のimport結果、特に
`updated_issues`、`stale_skipped_ids`、`tie_kept_local_ids` を確認します。

```bash
bash refresh_beads_sync.sh
git fetch origin +refs/heads/beads-sync:refs/remotes/origin/beads-sync

COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
BEADS_WT="${COMMON_DIR}/beads-worktrees/beads-sync"

if [[ ! -e "${BEADS_WT}/.git" ]]; then
  if git show-ref --verify --quiet refs/heads/beads-sync; then
    git worktree add "${BEADS_WT}" beads-sync
  else
    git worktree add -b beads-sync "${BEADS_WT}" origin/beads-sync
  fi
fi

git -C "${BEADS_WT}" pull --rebase origin beads-sync
bd export -o "${BEADS_WT}/.beads/issues.jsonl"
git -C "${BEADS_WT}" diff --check
git -C "${BEADS_WT}" add .beads/issues.jsonl
git -C "${BEADS_WT}" diff --cached --quiet ||
  git -C "${BEADS_WT}" commit -m "chore(beads): sync YOLOZU task state"
git -C "${BEADS_WT}" push origin beads-sync
```

`.beads/interactions.jsonl` に変更がある場合は、そのfileもstageする前に
localとremoteのappend-only logをstable `id` でmergeします。remote側の
新しいlogを短いlocal copyで置き換えないでください。

## 同時更新とpush reject

`bd import` はtimestampを考慮します。defaultでは次のように動作します。

- snapshot側が厳密に新しいrowなら、local issueを更新する
- 古いrowはskipする
- 同一timestampならlocal fieldを保持し、label、comment、dependencyはmergeする

`tie_kept_local_ids` が空でない場合は、取得したsnapshotと `bd show <id>` を
比較し、採用する値を `bd update` で明示してから再exportします。
`--allow-stale` はmergeの近道として使いません。古いsnapshotへ意図的に
復元する場合だけのoptionです。

別の環境が先に共有し、`beads-sync` のpushがrejectされた場合:

1. `beads-sync` worktreeをpull/rebaseする
2. `bash refresh_beads_sync.sh` で新しいsnapshotをimportする
3. 同一timestampとして報告されたrowを明示的に解決する
4. 再度export、commit、pushする

## 新しいRunPod checkout

RunPod wrapperも同じroot helperを使い、既存のBeads worktreeを必要としません。

```bash
bash deploy/runpod/refresh_beads_sync.sh
```

single-branch cloneの背景は `deploy/runpod/README.md` を参照してください。

## GitHub Issueへのlink

既存のGitHub Issueをlinkします。

```bash
bd update <id> --external-ref gh-123
```

repo operatorは、title完全一致によるlinkまたはGitHub Issueの作成ができます。

```bash
bd export -o .beads/issues.jsonl
python3 tools/link_beads_to_github.py --dry-run
python3 tools/link_beads_to_github.py
```

明示的にexportすることでoperatorの入力snapshotを最新にします。既に
`external_ref` があるrecordはskipされます。非dry-runの前に
`gh auth status` でGitHub CLIの認証を確認してください。
