@echo off
cd /d "C:\python\email_classifier"
echo Adding complete translation changes to git...
git add .
echo Committing complete English translation...
git commit -m "Complete English translation - ALL Polish comments and messages removed

✅ TRANSLATED ALL FILES:
- data_splitter.py: All stratified split, temporal split, group split comments
- data_structurer.py: All distribution analysis, hierarchical labels comments  
- model_evaluator.py: All precision/recall, confusion matrix comments
- model_trainer.py: All save model, save history comments
- pipeline.py: All print statements and step comments
- sampler.py: All custom balanced sampling comments
- strategy.py: All label quality analysis comments
- text_preprocessor.py: All remove whitespace comments
- vectorizer.py: All vectorizer configuration comments

🎯 RESULT: 100% English codebase with NO Polish characters
🚀 Pipeline runs without UnicodeEncodeError
🌍 Ready for international collaboration"
echo Pushing to main repository...
git remote remove origin
git remote add origin https://github.com/tomekbiel/email_classifier.git
git push origin main
echo Pushing to school repository...
git remote remove school-origin
git remote add school-origin https://github.com/tomaszbielNCI/email_classifier.git
git push school-origin main
echo 🎉 COMPLETE TRANSLATION pushed to both repositories!
pause
