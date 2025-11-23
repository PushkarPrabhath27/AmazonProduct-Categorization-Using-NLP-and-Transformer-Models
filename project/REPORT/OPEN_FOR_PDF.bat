@echo off
echo ========================================
echo Opening Report in Browser for PDF Export
echo ========================================
echo.
echo Instructions:
echo 1. Browser will open automatically
echo 2. Press Ctrl + P (or click Print button)
echo 3. Select "Microsoft Print to PDF" or "Save as PDF"
echo 4. Choose save location
echo 5. Click Save
echo.
echo Press any key to open the report...
pause > nul

start "" "C:\Users\pushk\OneDrive\Documents\IIT_NLP\project\REPORT\final_report.html"

echo.
echo Report opened in browser!
echo Follow the instructions above to save as PDF.
echo.
pause
