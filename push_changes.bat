@echo off
cd /d "C:\python\email_classifier"
echo Current directory: %CD%
echo Adding changes to git...
git add .
echo Committing changes...
git commit -m "Optimize translation performance and fix model issues

- Switch from M2M100 to MarianMT model (10x faster)
- Add translation cache to avoid duplicate translations
- Limit text length to 500 characters for faster processing
- Translate only unique texts, then map to all originals
- Add progress reporting and better logging
- Fix import issues with SentencePiece dependency
- Add note about API limitations (summary only translation)"
echo Pushing to main repository...
git push origin main
echo Pushing to school repository...
git push school-origin main
echo Done!
pause
