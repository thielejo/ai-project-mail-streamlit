[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NELeUmAZ)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=23211453)
# BIS5522 AI & Machine Learning - Group Project

Welcome to the group project repository for the course **BIS5522 AI & Machine Learning**. 
This template serves as the starting point for your project work in groups of three. 
You can use this repository to collaborate, train your AI models, and document your results.

## 👥 Project Task

In this project, your group will collaboratively develop, train, and evaluate a Machine Learning or AI model based on the topics covered in the lecture. You are free to write pure Python scripts or use Jupyter Notebooks (`.ipynb`), depending on your preferred workflow.

## 🛠️ Prerequisites

To ensure reproducible environments across all team members, we use `uv` for dependency management. Before you start, make sure you have:

- [Git](https://git-scm.com/downloads) for version control and collaboration
- [Python 3.12 or newer](https://www.python.org/downloads/)
- [VS Code](https://code.visualstudio.com/) 
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for fast, reproducible setups

## 🚀 Getting Started

1. **Clone the repository:**
   *(One team member should set up the GitHub repository and invite the others)*
   ```bash
   git clone <your-group-repo-url>
   cd ai-bis5522-ml-project
   ```

2. **Open the project in VS Code:**
   ```bash
   code .
   ```

3. **Install the dependencies:**
   This command creates a virtual environment (`.venv`) and installs everything you need, including `pytorch`, `transformers`, `scikit-learn`, `pandas`, and `jupyterlab`.
   ```bash
   uv sync
   ```

4. **Verify your setup:**
   Test your environment using our built-in command:
   ```bash
   uv run main.py
   ```

## 📦 Dependencies

The project already includes the standard data science and AI stack:
- `torch` (PyTorch)
- `transformers` & `huggingface-hub`
- `scikit-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `jupyterlab`

If you need additional packages, you can add them easily using:
```bash
uv add <package-name>
```

## 🔑 Environment Variables

If your project requires API keys (e.g., using Cloud LLMs), create a local `.env` file (which is git-ignored):
```bash
STACKIT_API_KEY="your-key-here"
MISTRAL_API_KEY="your-key-here"
```

## 📖 Collaboration Guide

1. Create a branch for your feature: `git checkout -b feature/model-training`.
2. Commit your work regularly: `git commit -m "added data preprocessing"`.
3. Push to GitHub and create a Pull Request to merge your work into the `main` branch.
4. Keep your environment in sync! Run `uv sync` whenever another team member adds a new dependency.

## 🚗 Universal Pricing Agent Workflow

Build the model features:

```bash
uv run python scripts/build_features.py
```

Train the Stage 1 price model:

```bash
uv run python scripts/train_price_model.py
```

Start the Streamlit demo:

```bash
uv run streamlit run app/streamlit_app.py
```

Good luck and have fun building! 🚀
