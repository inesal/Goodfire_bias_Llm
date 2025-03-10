# Goodfire_bias_Llm
# README: AI Bias in Resume Screening

## Project Overview
This project investigates potential gender bias in AI-driven resume screening systems using mechanistic interpretability techniques. The study utilizes a large language model (
Meta-Llama-3.1-8B-Instruct) accessed through Goodfire's Ember API to analyze how different names (male vs. female) affect the likelihood of resume acceptance.

## Methodology
1. **Data Generation**:
   - Created synthetic resumes varying only by first name (male/female names).
   - Assigned each resume a set of predefined skills with different levels of match to job requirements.
   
2. **Model Analysis**:
   - The model was prompted with resume screening tasks.
   - Logits for "Yes" (acceptance) and "No" (rejection) responses were extracted and analyzed.

3. **Statistical Evaluation**:
   - Conducted a t-test to assess gender-based logit differences.
   - Used ANOVA to determine the effect of skill matching.
   - Performed logistic regression to quantify the impact of gender on acceptance likelihood.

## Key Findings
- Male-associated names received higher acceptance scores in ambiguous cases (1 out of 3 skill matches).
- Statistical analysis confirmed a significant bias favoring male names in these scenarios.
- Bias likely originates from pre-trained model data reflecting historical hiring inequalities.

## Mitigation Strategies
- **Anonymization**: Removing gender-identifiable information before processing.
- **Fairness Constraints**: Adjusting training data to ensure gender balance.
- **Bias Audits**: Regular monitoring using interpretability tools (e.g., Goodfire Ember API).
- **Adversarial Training**: Training models to recognize and counteract bias in decision-making.

## Files Included
- `ai_cv_bias_analysis.tex` : LaTeX document containing the full research paper.
- `male_female_test.py` : Python script for generating test data and analyzing model outputs.
- `dataset.parquet` : Processed dataset used for analysis.
- `figures/` : Directory containing plots and visualizations.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy plotly goodfire statsmodels matplotlib seaborn tqdm
   ```
2. Run the script to generate and analyze results:
   ```bash
   python male_female_test.py
   ```
3. View results in the generated figures or use the dataset for further analysis.

## Future Work
- Extend analysis to additional demographic attributes (ethnicity, age, etc.).
- Investigate mitigation methods in real-world recruitment AI applications.
- Expand dataset to include real resumes with anonymized demographic features.
