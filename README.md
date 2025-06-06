# Investigating the Impact of News on the Brazilian Stock Market: A Google Trends Analysis

## Author  
Matheus Nogueira  
📧 Contact: matnogueira@gmail.com

## About this Repository

This repository contains the final project developed for the course **IND2604 – Probability and Statistics** at PUC-Rio.

The project explores whether **public interest in financial topics**, as measured by **Google Trends**, can help explain or predict the behavior of the Brazilian stock market. Specifically, it investigates the explanatory power of search volume related to economic terms over the **Ibovespa Index** using a **multiple linear regression model**.

The study uses:
- Daily Google Trends data for 10 finance-related keywords
- Daily closing prices of the Ibovespa Index
- Macroeconomic control variables (SELIC and CDI rates)

The model tests both individual and joint statistical significance of the search trends, and diagnostic tests are conducted to assess the model specification.

### Key Findings
- Google Trends for terms like “Ibovespa” and “Stocks” showed statistically significant coefficients.
- The overall inclusion of Google Trends improved model explanatory power.
- Despite some violations in residual diagnostics (due to COVID-19 shocks), the model provided valid inferential insights.

## Academic Integrity Notice

This repository is intended strictly for academic and educational reference.  
**Plagiarism or reuse in coursework (especially in IND2604 or similar subjects) is strictly prohibited.**  
The project was made public only after the course was completed.

## Methodology Summary

- **Data Period**: Jan 2010 – Jan 2024 (weekdays only)  
- **Keywords**: "Ibovespa", "Bolsa de Valores", "Ações", "Dividendos", "Renda Fixa", "Inflação", "CDI", "Dólar", "Bitcoin", "Renda Variável"  
- **Model**: Multiple linear regression with lagged regressors  
- **Tests Used**:  
  - *t-tests* and *F-tests* for coefficient significance  
  - Jarque-Bera, Ljung-Box, and Breusch-Pagan for residual diagnostics
