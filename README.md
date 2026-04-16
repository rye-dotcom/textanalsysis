  🚀 Text Analysis App — Installation Guide

This guide will help you install and run the application on Windows, macOS, or Linux.

✅ Prerequisites

Before starting, make sure you have:

A terminal application:
Windows: Command Prompt or Git Bash
macOS/Linux: Terminal
Internet access
🔍 1. Check if Git is Installed

Open your terminal and run:

git --version
What this means:
If Git is installed, you’ll see a version number.
If not, you’ll get an error like: git is not recognized
📦 2. Install Git (Windows only)
Download Git for Windows:
👉 https://gitforwindows.org/
Run the installer and follow the default setup steps.
After installation, reopen your terminal and verify:
git --version
🧰 3. Install uv (Python Package Manager)

We use uv to manage Python and dependencies.

Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
macOS / Linux:
curl -Ls https://astral.sh/uv/install.sh | sh

Verify installation:

uv --version
📁 4. Clone the Repository

Choose or create a folder where you want the project:

mkdir textanalysis
cd textanalysis

Now clone the repository:

git clone https://github.com/rye-dotcom/textanalysis.git .

⚠️ Note: The trailing . installs into the current folder.

🐍 5. Create a Virtual Environment
uv venv

This creates a local Python environment in .venv/.

⚡ 6. Activate the Environment
Windows (PowerShell / Git Bash):
.venv\Scripts\activate
macOS / Linux:
source .venv/bin/activate

You should now see (.venv) in your terminal prompt.

📦 7. Install Dependencies
uv pip install -r pyproject.toml
🧠 8. Download Language Model (spaCy)
uv run python -m spacy download en_core_web_sm
▶️ 9. Run the Application
uv run streamlit run text_analysis.py

After a few seconds, Streamlit will open the app in your browser automatically.

🎉 You're Done!

You should now have the Text Analysis app running locally.

🛠 Troubleshooting
If git is not recognized → restart your terminal after installing Git
If uv is not recognized → restart your terminal after installation
If activation fails on Windows → try PowerShell instead of Git Bash
Make sure you're running commands inside the project folder
