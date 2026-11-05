# Week 2 Logistics Inventory Preprocessing

Files:
- raw_inventory_logistics.csv: synthetic raw data with intentionally introduced quality issues.
- cleaned_inventory_logistics.csv: cleaned, capped, scaled output.
- preprocess_inventory.py: reproducible pandas/scikit-learn pipeline.
- PNG charts: visualisations for the report.

Run:
pip install pandas numpy scikit-learn
python preprocess_inventory.py

Dataset reference: UCI Machine Learning Repository, Online Retail (Chen, 2015). Its transaction fields include invoice, product code, quantity, date and unit price. This project uses the UCI dataset as a schema/reference example only; the supplied CSV is separately generated synthetic logistics data, not UCI records.

The sample data is synthetic and for education only. Median imputation and IQR capping are baseline techniques; confirm assumptions with domain experts before real operational use.
