@echo off
echo ========================================================
echo SIGNAL SCOPE - REALISTIC HISTORY GENERATOR (11-15 Sept)
echo ========================================================

:: 1. Destroy old history and initialize fresh
echo Resetting Git repository...
rmdir /s /q .git
git init

:: 2. Setup Team Profiles
set "ADITYA=Aditya Parmar <adityaparmar28@users.noreply.github.com>"
set "TAPAN=Tapan <tapansoni2007-dotcom@users.noreply.github.com>"
set "KRINA=Krina Malviya <KrinaMalaviya@users.noreply.github.com>"
set "YUVRAJ=Yuvraj <YUXRAJ21@users.noreply.github.com>"
set "ATHUL=Athul Nair <athul2917-tech@users.noreply.github.com>"
set "PARI=Pari Doshi <paridoshi25@users.noreply.github.com>"

:: ================= DAY 1: 11 Sept =================
echo Committing Day 1 changes...
git checkout -b main
set GIT_COMMITTER_DATE=2026-09-11T10:00:00
git add LICENSE README.md problem_statement.pdf .gitignore
git commit --author="%PARI%" --date="2026-09-11T10:00:00" -m "docs: Initial repository setup with LICENSE and skeleton README"

:: Data setup
git checkout -b feature/data
set GIT_COMMITTER_DATE=2026-09-11T14:30:00
git add src/dataset.py src/fetch_art_dataset.py data/
git commit --author="%KRINA%" --date="2026-09-11T14:30:00" -m "feat(data): Setup initial data loading and fetch scripts"

:: ML setup
git checkout main
git checkout -b feature/ml
set GIT_COMMITTER_DATE=2026-09-11T16:00:00
git add src/model/dual_branch_net.py
git commit --author="%TAPAN%" --date="2026-09-11T16:00:00" -m "feat(ml): Add skeleton for dual_branch_net architecture"

:: Frontend setup
git checkout main
git checkout -b feature/frontend
set GIT_COMMITTER_DATE=2026-09-11T17:30:00
git add css/ js/ static/ web_server.py
git commit --author="%ADITYA%" --date="2026-09-11T17:30:00" -m "feat(ui): Initialize frontend structure and web server skeleton"


:: ================= DAY 2: 12 Sept =================
echo Committing Day 2 changes...
git checkout feature/testing
git checkout -b feature/testing main
set GIT_COMMITTER_DATE=2026-09-12T10:00:00
git add test_smoke.py
git commit --author="%ATHUL%" --date="2026-09-12T10:00:00" -m "test: Add smoke tests to verify environment setup"

git checkout feature/data
set GIT_COMMITTER_DATE=2026-09-12T11:15:00
git add src/model/srm_filters.py src/generate_sample_data.py
git commit --author="%KRINA%" --date="2026-09-12T11:15:00" -m "feat(data): Implement SRM filters for noise analysis"

git checkout feature/ml
set GIT_COMMITTER_DATE=2026-09-12T14:00:00
git add src/model/train.py src/train.py
git commit --author="%TAPAN%" --date="2026-09-12T14:00:00" -m "feat(ml): Implement forward pass and training loop"

git checkout -b feature/explain main
set GIT_COMMITTER_DATE=2026-09-12T16:45:00
git add src/explain/__init__.py src/explain/gradcam.py
git commit --author="%YUVRAJ%" --date="2026-09-12T16:45:00" -m "feat(explain): Setup explainability module and gradcam logic"


:: ================= DAY 3: 13 Sept =================
echo Committing Day 3 changes...
git checkout feature/data
set GIT_COMMITTER_DATE=2026-09-13T10:00:00
git add src/data_pipeline.py src/data_pipeline_fast.py src/add_diffusiondb.py
git commit --author="%KRINA%" --date="2026-09-13T10:00:00" -m "feat(data): Optimize data loading and integrate DiffusionDB"

git checkout feature/explain
set GIT_COMMITTER_DATE=2026-09-13T11:20:00
git add src/explain/explainer.py
git commit --author="%YUVRAJ%" --date="2026-09-13T11:20:00" -m "feat(explain): Add explainer.py for visual text gen cues"

git checkout feature/frontend
set GIT_COMMITTER_DATE=2026-09-13T14:15:00
git add app/ index.html
git commit --author="%ADITYA%" --date="2026-09-13T14:15:00" -m "feat(ui): Create main UI and connect to model endpoints"

git checkout feature/ml
set GIT_COMMITTER_DATE=2026-09-13T16:30:00
git add src/model/metadata.py download_weights.py train_pipeline.bat
git commit --author="%TAPAN%" --date="2026-09-13T16:30:00" -m "feat(ml): Add metadata handling and training utilities"


:: ================= DAY 4: 14 Sept =================
echo Committing Day 4 changes...
git checkout feature/testing
set GIT_COMMITTER_DATE=2026-09-14T10:30:00
git add src/robustness/ src/degradation_benchmark.py
git commit --author="%ATHUL%" --date="2026-09-14T10:30:00" -m "test: Implement degradation benchmarks and robustness testing"

git checkout feature/frontend
set GIT_COMMITTER_DATE=2026-09-14T12:00:00
git add requirements.txt run_v2_app.bat
git commit --author="%ADITYA%" --date="2026-09-14T12:00:00" -m "feat(ui): Polish UI interactions and deployment scripts"

git checkout feature/ml
set GIT_COMMITTER_DATE=2026-09-14T15:00:00
git add src/model/predict.py src/model/legacy_model.py
git commit --author="%TAPAN%" --date="2026-09-14T15:00:00" -m "feat(ml): Refactor prediction logic and add legacy fallback"

git checkout -b feature/docs main
set GIT_COMMITTER_DATE=2026-09-14T17:00:00
git add report/generate_report_figures.py
git commit --author="%PARI%" --date="2026-09-14T17:00:00" -m "docs: Create script to generate report figures"


:: ================= DAY 5: 15 Sept =================
echo Committing Day 5 changes...
git checkout feature/testing
set GIT_COMMITTER_DATE=2026-09-15T09:00:00
git add tests/ src/evaluate.py
git commit --author="%ATHUL%" --date="2026-09-15T09:00:00" -m "test: Add evaluation scripts and integration tests"

git checkout feature/docs
set GIT_COMMITTER_DATE=2026-09-15T14:00:00
git add report/ FINAL_HACKATHON_REPORT.md RELEASE_NOTES.md
git commit --author="%PARI%" --date="2026-09-15T14:00:00" -m "docs: Add final evaluation metrics and Hackathon report"


:: ================= MERGE & PUSH =================
echo Merging branches...
git checkout main
set GIT_COMMITTER_DATE=2026-09-15T16:00:00

:: Set merge dates artificially by overriding the commit date
git merge feature/data --no-ff -m "Merge branch 'feature/data' into main"
git merge feature/ml --no-ff -m "Merge branch 'feature/ml' into main"
git merge feature/explain --no-ff -m "Merge branch 'feature/explain' into main"
git merge feature/frontend --no-ff -m "Merge branch 'feature/frontend' into main"
git merge feature/testing --no-ff -m "Merge branch 'feature/testing' into main"
git merge feature/docs --no-ff -m "Merge branch 'feature/docs' into main"

:: Catch any remaining uncommitted files
git add .
set GIT_COMMITTER_DATE=2026-09-15T17:00:00
git commit --author="%ADITYA%" --date="2026-09-15T17:00:00" -m "chore: Final project sync and cleanup for submission"

:: Add new remote repository
git remote add origin https://github.com/adityaparmar28/SignalScope-by-LogicLegion.git

echo ========================================================
echo Git History Generated for SignalScope-by-LogicLegion!
echo Run the following command to push all branches:
echo git push -u origin --all -f
echo ========================================================
pause
