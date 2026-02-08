# Medical Report Tracker – Setup Guide

Lab report tracker for personal use. Upload PDF/image lab reports, extract values, and view trends over time. **Runs locally on your machine** – no cloud hosting required.

## Prerequisites

- **Python 3.10+**  
- **OpenAI API key** ([platform.openai.com](https://platform.openai.com))

---

## Mac

### 1. Install Python (if needed)

```bash
brew install python@3.12
```

Or use the installer from [python.org](https://www.python.org/downloads/).

### 2. Clone or download the project

```bash
cd ~/Downloads  # or your preferred folder
git clone https://github.com/YOUR_USERNAME/Medical_Report.git
cd Medical_Report
```

Or download the ZIP from GitHub and extract it.

### 3. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

Replace with your actual OpenAI API key.

### 6. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Windows

### 1. Install Python (if needed)

Download and install from [python.org](https://www.python.org/downloads/).  
**Important:** Check **"Add Python to PATH"** during installation.

### 2. Clone or download the project

Using Git:

```cmd
cd %USERPROFILE%\Downloads
git clone https://github.com/YOUR_USERNAME/Medical_Report.git
cd Medical_Report
```

Or download the ZIP from GitHub and extract it.

### 3. Create virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies

```cmd
pip install -r requirements.txt
```

### 5. Configure API key

Create a file named `.env` in the project root with:

```
OPENAI_API_KEY=sk-your-api-key-here
```

Replace with your actual OpenAI API key.

### 6. Run the app

```cmd
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Notes

- **Data** is stored in `health_data.db` (SQLite) in the project folder.
- **Uploads** are saved in the `uploads/` folder.
- **Logs** (e.g. processing discrepancies) go to `logs/processing.log`.
- Each user runs the app on their own machine. There is no central server or cloud hosting.

---

## Publishing to GitHub (for distribution)

1. Create a new repository on [github.com](https://github.com/new). Leave it empty (no README, no .gitignore).

2. Add the remote and push:

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/Medical_Report.git
   git branch -M main
   git push -u origin main
   ```

3. Share the repo URL with friends. They can clone and follow this setup guide.
