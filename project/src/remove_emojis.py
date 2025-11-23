"""
Remove ALL emojis from final_report.md
"""
import re
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_FILE = PROJECT_ROOT / "REPORT" / "final_report.md"

# Read the file
with open(REPORT_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# Define emoji patterns to remove
# This covers: ✅ ✓ ❌ ⚠️ 🔍 🚀 📝 📊 ⏰ and all other emojis
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000026FF"  # Miscellaneous Symbols (includes ✅ ✓)
    "\U00002700-\U000027BF"  # Dingbats
    "]+",
    flags=re.UNICODE
)

# Count emojis before
emojis_found = emoji_pattern.findall(content)
print(f"Found {len(emojis_found)} emoji characters to remove")

# Remove all emojis
content_clean = emoji_pattern.sub('', content)

# Also remove specific checkmark symbols that might not be caught
content_clean = content_clean.replace('✅', '')
content_clean = content_clean.replace('✓', '')
content_clean = content_clean.replace('❌', '')
content_clean = content_clean.replace('⚠️', '')
content_clean = content_clean.replace('🔍', '')
content_clean = content_clean.replace('🚀', '')
content_clean = content_clean.replace('📝', '')
content_clean = content_clean.replace('📊', '')
content_clean = content_clean.replace('⏰', '')

# Clean up any double spaces left after emoji removal
content_clean = re.sub(r'  +', ' ', content_clean)

# Clean up bullet points that start with just a dash and space
content_clean = re.sub(r'^- \*\*', r'- **', content_clean, flags=re.MULTILINE)

# Write back
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(content_clean)

print(f"\nCleaned file written to: {REPORT_FILE}")
print(f"Removed {len(emojis_found)} emojis")
print("Report is now 100% emoji-free!")
