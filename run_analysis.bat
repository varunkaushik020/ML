@echo off
echo Chronic Disease Analysis Runner
echo =============================

echo.
echo 1. Testing dataset accessibility...
python scripts/test_datasets.py

echo.
echo 2. Analyzing datasets...
python scripts/analyze_datasets.py

echo.
echo 3. Combining datasets...
python scripts/combine_datasets.py

echo.
echo Analysis complete! Check the generated plots and reports.
pause