@echo off
set GIT_COMMITTER_DATE=2026-09-15T14:30:00
git add README.md RELEASE_NOTES.md
git commit --author="Aditya Parmar <adityaparmar28@users.noreply.github.com>" --date="2026-09-15T14:30:00" -m "Deploy v2.0: High-Accuracy DiffusionDB Integration and Threshold Calibration"
git tag v2.0-final
echo V2 Deployment Committed Successfully!
