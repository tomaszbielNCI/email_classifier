@echo off
cd /d "C:\python\email_classifier"
echo Current directory: %CD%
echo Adding translation changes to git...
git add .
echo Committing translation changes...
git commit -m "Complete English translation of all module comments and documentation

- Translate data_structurer.py: Step 5 multi-level data handling
- Translate model_evaluator.py: Classification model evaluation  
- Translate model_trainer.py: SOTA model training and evaluation
- Translate sampler.py: Data balancing and class imbalance analysis
- Translate strategy.py: Supervised vs unsupervised decision logic
- Translate text_preprocessor.py: Regex and noise removal
- Translate vectorizer.py: TF-IDF and embeddings numerical representation
- All Polish comments converted to professional English documentation
- Complete internationalization of codebase"
echo Pushing to main repository...
git remote remove origin
git remote add origin https://github.com/tomekbiel/email_classifier.git
git push origin main
echo Pushing to school repository...
git remote remove school-origin
git remote add school-origin https://github.com/tomaszbielNCI/email_classifier.git
git push school-origin main
echo Translation changes pushed to both repositories!
pause
