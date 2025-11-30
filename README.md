Machine Learning Internship Projects
Silica Concentration Prediction & Agriculture Crop Production Forecasting

Overview:
This repository contains two real-world machine learning projects completed during my internship:
	1.	Mining Process – Silica Concentration Prediction
	2.	Agriculture Crop Production Prediction in India

Both projects focus on applying data science and machine learning techniques to analyze real industry datasets, build forecasting models, and extract actionable insights.

Project 1: Silica Concentration Prediction in Mining Process

Objective:
Predict the percentage of silica impurity in iron ore concentrate in a flotation processing plant.

Dataset:
Real industrial dataset from a mining flotation plant
Time period: March 2017 – September 2017
~700K+ time-series rows with mixture of 20-second sensor readings and hourly lab results

Key Tasks:
	•	Cleaned raw industrial data (comma-separated numeric formats)
	•	Converted timestamp to datetime and resampled to 1-minute frequency
	•	Created lag and rolling window features
	•	Built Random Forest model to forecast silica impurity
	•	Compared performance with and without % Iron Concentrate feature
	•	Evaluated multi-horizon forecasts (1, 2, 4, 8, 12, 24 hours ahead)

Highlights:
	•	Accurate short-term prediction of silica concentration
	•	Model outperformed persistence baseline for most horizons
	•	Silica can still be predicted without iron feature, but accuracy improves when included

Folder Structure:
silica_project/
├─ src/ (processing + model file)
├─ results/
└─ data/ (dataset is stored here after download)

Dataset Download:
Due to file size, the dataset is not included.
Download from Google Drive link (provided separately) and place inside the data folder.

Project 2: Agriculture Crop Production Prediction in India

Objective:
Analyze and predict agricultural production trends in India using historical government data (2001–2014).

Dataset Description:
Five CSV files containing:
	•	Crop cultivation cost and yield by state
	•	Crop production, area, and yield (2006–2011)
	•	Crop varieties, recommended zones, and seasons
	•	Agricultural index values across years
	•	National production time-series data (1993–2014)

Key Tasks:
	•	Merged all datasets into a unified format
	•	Cleaned numeric and year formats (eg: “2006-07”, “3-1999”)
	•	Converted wide tables into long time-series format
	•	Performed missing value handling and feature engineering
	•	Trained baseline Random Forest model

Insights:
	•	India agriculture production generally increased over time
	•	Yield per hectare is the strongest determinant of production
	•	Cost of cultivation does not always correlate with higher production
	•	Seasonal and regional factors significantly influence outputs

Folder Structure:
agriculture_project/
├─ src/ (processing + model pipeline)
├─ notebooks/
├─ results/
└─ data/ (place CSV files here)

Technologies Used:
Python
Pandas, NumPy
Scikit-Learn
Matplotlib
Optional: XGBoost, Statsmodels

How to Run:
	1.	Download datasets (mining dataset from Google Drive link, agriculture CSVs   provided)
	2.	Place data inside the “data” folder in respective project
	3.	Install dependencies
pip install pandas numpy scikit-learn matplotlib
	4.	Run the main script
python src/your_script_name.py

Conclusion:
These two projects represent advanced data engineering + machine learning workflows covering:
	•	Real industrial sensor data
	•	Multi-file agricultural public data
	•	Time-series modeling
	•	Feature engineering
	•	Forecasting and evaluation
	•	Real-world dataset cleaning challenges


